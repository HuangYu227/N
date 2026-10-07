"""Inference-only protected initial-observation reads; no parameters or writes."""
from dataclasses import dataclass
import hashlib
import math
import torch
from .geometry import apply_complex_rope


@dataclass(frozen=True)
class SinkOptions:
    mode: str = "off"
    gain: float = .1
    position: str = "temporal-realign"
    start_chunk: int = 5

    def __post_init__(self):
        if self.mode not in ("off", "protected", "zero"):
            raise ValueError("unknown TLA sink mode")
        if self.position not in ("absolute", "temporal-realign"):
            raise ValueError("unknown TLA sink position")
        if isinstance(self.gain, bool) or not isinstance(self.gain, (int, float)) or not math.isfinite(self.gain) or not 0 <= self.gain <= 1:
            raise ValueError("sink gain must be finite and within [0, 1]")
        if isinstance(self.start_chunk, bool) or not isinstance(self.start_chunk, int) or self.start_chunk < 1:
            raise ValueError("sink start chunk must be a positive integer")

    @property
    def active(self):
        return self.mode != "off" and self.gain > 0


def add_sink_arguments(parser):
    parser.add_argument("--tla-sink", choices=("off", "protected", "zero"), default="off")
    parser.add_argument("--sink-gain", type=float, default=.1)
    parser.add_argument("--sink-position", choices=("absolute", "temporal-realign"), default="temporal-realign")
    parser.add_argument("--sink-start-chunk", type=int, default=5)


def sink_options_from_args(args):
    return SinkOptions(getattr(args, "tla_sink", "off"), getattr(args, "sink_gain", .1),
                       getattr(args, "sink_position", "temporal-realign"), getattr(args, "sink_start_chunk", 5))


def validate_sink(options, config, ablation="full", execution=None):
    if not isinstance(options, SinkOptions):
        raise TypeError("sink_options must be SinkOptions")
    if not options.active: return
    if torch.is_grad_enabled():
        raise ValueError("TLA sink is inference-only; disable gradients before creating or using the runtime")
    if config.stage != "C": raise ValueError("TLA sink requires Stage C")
    if config.camera_attention != "sana": raise ValueError("TLA sink requires native SANA camera")
    if ablation != "full": raise ValueError("TLA sink requires full TTN; combine interventions in separate experiments")
    if execution is not None and (execution.core_backend != "reference" or execution.psi_backend != "reference"):
        raise ValueError("TLA sink requires reference/reference")


def freeze_reference(state):
    """Called before atomic publication: allocation failure leaves the old runtime."""
    return state.detach().clone()


def reference_sha256(state):
    return hashlib.sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def realign_reference(reference, rope, delta):
    """K -> K D means K^T V -> D^T A; rotate rows via RoPE on A^T.

    Only SANA's actual causal time layout is supported. H/W channels retain
    unit phase. This is a time-coordinate change, not camera reprojection.
    """
    if (rope is None or not any(cls.__name__ == "CausalWanRotaryPosEmbed" for cls in type(rope).__mro__)
            or not isinstance(getattr(rope, "freqs", None), torch.Tensor)):
        raise ValueError("sink alignment requires the actual CausalWanRoPE table/layout")
    dim = getattr(rope, "attention_head_dim", None)
    table = rope.freqs
    if (reference.ndim < 3 or reference.shape[-2:] != (dim, dim) or dim % 2
            or table.ndim != 2 or table.shape[1] != dim//2 or not table.is_complex()):
        raise ValueError("sink reference and causal RoPE dimensions disagree")
    if (not isinstance(delta, torch.Tensor) or delta.ndim != 1 or delta.shape[0] != reference.shape[0]
            or delta.dtype not in (torch.int32, torch.int64) or (delta < 0).any() or (delta >= table.shape[0]).any()):
        raise ValueError("sink temporal offset is outside the causal RoPE table")
    if not delta.any(): return reference
    time_channels = dim//2 - 2*(dim//6)  # The split used by CausalWanRotaryPosEmbed.forward.
    table = table.to(reference.device)
    phase = table.new_ones((reference.shape[0], dim//2))
    phase[:, :time_channels] = table.index_select(0, delta.to(reference.device))[:, :time_channels] / table[0, :time_channels]
    if not torch.isfinite(phase).all(): raise ValueError("nonfinite sink RoPE phase")
    phase = phase.reshape(reference.shape[0], *([1] * (reference.ndim-2)), dim//2)
    return apply_complex_rope(reference.transpose(-1, -2), phase).transpose(-1, -2)


@torch.no_grad()
def sink_read_stats(fast_raw, sink_raw, reference, delta_raw, options, shift, eps, reference_hash=None, read=None):
    """Detached per [CFG branch, head] evidence; zero-denominator ratios are null."""
    def norm(value): return value.detach().float().flatten(-2).norm(dim=-1)
    if read is not None:
        valid_read = read[:, None, :, None]
        fast_raw, sink_raw, delta_raw = (torch.where(valid_read, value, 0.) for value in (fast_raw, sink_raw, delta_raw))
    fast, sink = norm(fast_raw), norm(sink_raw)
    reference_norm = norm(reference)
    valid = (fast > eps).cpu().tolist()
    ratio = (sink/fast.clamp_min(eps)).cpu().tolist()
    return {"mode": options.mode, "gain": options.gain, "position": options.position,
            "temporal_shift": shift.cpu().tolist(),
            "reference_source": "observed-prefill" if options.mode == "protected" else "zero-control",
            "reference_sha256": reference_hash,
            "per_head": {"reference_norm": reference_norm.cpu().tolist(),
                         "reference_rms": (reference_norm/reference.shape[-1]).cpu().tolist(),
                         "fast_read_norm": fast.cpu().tolist(), "sink_read_norm": sink.cpu().tolist(),
                         "sink_fast_ratio": [[x if valid[b][h] else None for h, x in enumerate(row)] for b, row in enumerate(ratio)],
                         "raw_delta_norm": norm(delta_raw).cpu().tolist()},
            "measurement": "detached FP32; [CFG branch, head]; valid read tokens only; read intervention only",
            "effective_measurement": "FP32 linear gate/projection contribution before output quantization; projection bias cancels",
            "effective_head_grouping": "output-channel groups after projection, not causal attribution to input heads"}


@torch.no_grad()
def sink_effective_stats(stats, delta_raw, gate, proj, read, heads):
    """Projection is linear in the intervention: exclude its shared bias."""
    with torch.autocast(device_type=delta_raw.device.type, enabled=False):
        gated = delta_raw.detach().float()*gate.detach().float()
        delta = torch.nn.functional.linear(gated, proj.weight.detach().float(), None) * read[..., None]
    per_head = delta.reshape(*delta.shape[:-1], heads, -1).transpose(1, 2)
    stats["per_head"]["effective_delta_norm"] = per_head.float().flatten(-2).norm(dim=-1).cpu().tolist()
    stats["effective_delta_norm"] = delta.float().norm().item()
