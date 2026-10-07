"""Observed-prefill constraints on Correct, with explicit opt-in for training."""
from dataclasses import dataclass
import hashlib
import math

import torch

from .core import correct_with_aux
from .sink import reference_sha256 as tensor_digest


@dataclass(frozen=True)
class ReplayOptions:
    mode: str = "off"
    strength: float = .25
    budget: int = 128
    start_chunk: int = 5

    def __post_init__(self):
        if self.mode not in ("off", "shrink", "observed"):
            raise ValueError("unknown TLA replay mode")
        if (isinstance(self.strength, bool) or not isinstance(self.strength, (int, float))
                or not math.isfinite(self.strength) or self.strength < 0):
            raise ValueError("replay strength must be finite and nonnegative")
        for name in ("budget", "start_chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"replay {name} must be a positive integer")

    @property
    def active(self):
        return self.mode != "off" and self.strength > 0


def add_replay_arguments(parser):
    parser.add_argument("--tla-replay", choices=("off", "shrink", "observed"), default=None,
                        help="defaults to checkpoint math; off explicitly disables replay")
    parser.add_argument("--replay-strength", type=float, default=.25)
    parser.add_argument("--replay-budget", type=int, default=128)
    parser.add_argument("--replay-start-chunk", type=int, default=5)


def replay_options_from_args(args, config=None):
    mode = getattr(args, "tla_replay", None)
    if mode is None: return trained_options(config) if config is not None else ReplayOptions()
    return ReplayOptions(mode, getattr(args, "replay_strength", .25),
                         getattr(args, "replay_budget", 128), getattr(args, "replay_start_chunk", 5))


def trained_options(config):
    return ReplayOptions("observed" if config.replay_strength else "off", config.replay_strength,
                         config.replay_budget, config.memory_start_chunk)


def validate_replay(options, config, ablation="full", execution=None, *, check_grad=True):
    if not isinstance(options, ReplayOptions): raise TypeError("replay_options must be ReplayOptions")
    if not options.active: return
    if check_grad and torch.is_grad_enabled() and options != trained_options(config):
        raise ValueError("observed replay is inference-only unless specified in the trained configuration")
    supported = ablation == "full" or (config.replay_strength and ablation in ("no-local", "no-persistent"))
    if config.stage != "C" or config.camera_attention != "sana" or not supported:
        raise ValueError("observed replay requires Full Stage C with native SANA camera")
    if execution is not None and (execution.core_backend, execution.psi_backend) != ("reference", "reference"):
        raise ValueError("observed replay requires reference/reference")


@dataclass(frozen=True)
class Observation:
    key: torch.Tensor
    value: torch.Tensor
    weight: torch.Tensor
    token_indices: torch.Tensor
    frame_ids: torch.Tensor


def capture_observation(k, v, w, write, frame_ids, budget):
    """Uniform spatial samples, shared across heads; no RNG and no noisy writes."""
    if (k.ndim != 4 or v.shape != k.shape or w.shape != k.shape[:-1]
            or write.shape != (k.shape[0], k.shape[-2]) or frame_ids.shape[0] != k.shape[0]
            or frame_ids.shape[1] != 1 or (frame_ids != 0).any()):
        raise ValueError("replay capture requires the single observed frame 0 and matching K/V/W")
    counts = write.bool().sum(-1)
    count = min(budget, int(counts.min()))
    if count < 1: raise ValueError("observed replay requires valid prefill tokens")
    indices = torch.stack([row.nonzero().flatten()[torch.linspace(0, int(n)-1, count,
        device=k.device).round().long()] for row, n in zip(write.bool(), counts)])
    gather = indices[:, None, :, None].expand(-1, k.shape[1], -1, k.shape[-1])
    tensors = (k.gather(2, gather), v.gather(2, gather),
               w.gather(2, indices[:, None].expand(-1, k.shape[1], -1)))
    if any(not torch.isfinite(t).all() for t in tensors) or (tensors[2] < 0).any():
        raise ValueError("nonfinite observation or negative replay weights")
    return Observation(*(t.detach().float().clone() for t in tensors), indices.detach().clone(),
                       frame_ids.expand(-1, count).detach().clone())


def observation_digest(observations):
    hashes = [tensor_digest(t) for item in observations for t in
              (item.key, item.value, item.weight, item.token_indices, item.frame_ids)]
    return hashlib.sha256("".join(hashes).encode("ascii")).hexdigest()


def storage_bytes(observations, transported_values):
    tensors = [t for item in observations for t in
               (item.key, item.value, item.weight, item.token_indices, item.frame_ids)]
    if transported_values is not None: tensors.append(transported_values)
    return sum(t.numel()*t.element_size() for t in tensors)


def correct_with_replay(pred, k, v, beta, mask, options, observation=None, target=None,
                        *, alpha_s=.5, eps=1e-6, collect_stats=False):
    """Convex mixture; shrink is the matched zero-history-gradient control.

    Keys keep their original absolute RoPE. Targets follow the SAME actual
    Cayley transport as the current prediction, including noisy Local psi.
    """
    aux = correct_with_aux(pred, k, v, beta, mask, alpha_s, eps)
    if not options.active: return aux.candidate, aux.w, None
    current = aux.kt_weighted_residual / aux.denom[..., None, None]
    history = torch.zeros_like(current)
    if options.mode == "observed":
        if observation is None or target is None or target.shape != observation.key.shape:
            raise ValueError("observed replay is active without a matching prefill target")
        hk, hw = observation.key, observation.weight
        hd = eps + (hw*hk.square().sum(-1)).sum(-1)
        history = hk.transpose(-1, -2) @ (hw[..., None]*(target-hk@pred)) / hd[..., None, None]
    mixed = (current + options.strength*history)/(1+options.strength)
    candidate = pred + alpha_s*mixed
    if not torch.isfinite(candidate).all(): raise FloatingPointError("nonfinite replay Correct candidate")
    stats = replay_statistics(pred, k, v, aux, history, current, mixed, candidate, options, observation,
                              target, alpha_s, eps) if collect_stats else None
    return candidate, aux.w, (stats, (candidate-aux.candidate).detach()) if stats is not None else None


@torch.no_grad()
def replay_statistics(pred, k, v, aux, history, current, mixed, candidate, options, observation, target, alpha_s, eps):
    """Detached telemetry must not add saved activations to the training graph."""
    if options.mode == "observed":
        hk, hw = observation.key, observation.weight
        hd = eps + (hw*hk.square().sum(-1)).sum(-1)
    def norm(t): return t.float().flatten(-2).norm(dim=-1)
    def values(t): return t.detach().cpu().tolist()
    def loss(key, value, weight, state, denom):
        return (weight[..., None]*(key@state-value).square()).sum((-2, -1))/(2*denom)
    cn, hn = norm(current), norm(history)
    cosine = (current*history).sum((-2, -1))/(cn*hn).clamp_min(eps)
    valid = (cn*hn > eps).cpu().tolist()
    cosine = values(cosine)
    stats = {"active": True, "mode": options.mode, "strength": options.strength,
             "current_scale": 1/(1+options.strength), "key_position": "original-absolute-rope",
             "value_transport": "actual-current-cayley", "source": "observed-prefill-only",
             "samples": observation.key.shape[-2] if observation is not None else 0,
             "per_head": {"current_gradient_norm": values(cn), "history_gradient_norm": values(hn),
                 "gradient_cosine": [[x if valid[b][h] else None for h, x in enumerate(row)]
                                     for b, row in enumerate(cosine)],
                 "mixed_update_norm": values(alpha_s*norm(mixed)),
                 "state_delta_vs_current_correct": values(norm(candidate-aux.candidate)),
                 "current_loss_before": values(loss(k, v, aux.w, pred, aux.denom)),
                 "current_loss_after": values(loss(k, v, aux.w, candidate, aux.denom))}}
    if options.mode == "observed":
        stats["per_head"].update(history_loss_before=values(loss(hk, target, hw, pred, hd)),
            history_loss_after=values(loss(hk, target, hw, candidate, hd)),
            value_transport_delta_norm=values(norm(target-observation.value)))
    stats["effective_measurement"] = "FP32 gate/projection contribution before output quantization"
    return stats
