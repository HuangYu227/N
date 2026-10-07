"""Explicit episode state and five-anchor clean transactions, independent of SANA caches."""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from .core import ANCHORS, TTNConfig, Rank2Generators, CayleyFactors, DetachedCayleySnapshot, clip_inner_gradient
from .performance import DEFAULT_EXECUTION, annotation, validate_execution
from .stability import committed_anchor_stats, matrix_scale, write_direction_stats
from .controller import TransitionController
from . import sink


class TTNSystem(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generators = Rank2Generators(config)
        self.controller = TransitionController(config)
        if config.local_update:
            # Constant initialization consumes no RNG and preserves inherited seeds.
            self.local_eta_logits = nn.Parameter(torch.full((5,), math.log(math.expm1(.01))))
        self.requires_grad_(config.stage != "A")


@dataclass
class TTNChunkContext:
    predicted: torch.Tensor
    previous: torch.Tensor
    psi: torch.Tensor
    cbase: torch.Tensor
    system: TTNSystem
    frame_ids: torch.Tensor
    poses: torch.Tensor
    read_mask: torch.Tensor
    write_mask: torch.Tensor
    revision: int
    runtime_id: int
    clean_mode: bool = False
    prefill_mode: bool = False
    candidates: dict = field(default_factory=dict)
    begin_id: int = 0
    psi_snapshot: DetachedCayleySnapshot | None = None
    write_history: tuple = ()
    live_factors: CayleyFactors | None = None
    local_stats: dict = field(default_factory=dict)
    ablation: str = "full"
    noise_info: dict = field(default_factory=dict)
    collect_local_stats: bool = False
    local_trajectory: list = field(default_factory=list)
    sink_options: sink.SinkOptions = field(default_factory=sink.SinkOptions)
    sink_reference: torch.Tensor | None = None
    sink_reference_sha256: str | None = None
    sink_shift: torch.Tensor | None = None
    sink_stats: dict = field(default_factory=dict)
    sink_trajectory: list = field(default_factory=list)
    noise_call_count: int = 0
    collect_sink_stats: bool = False

    def for_clean(self):
        return TTNChunkContext(self.predicted, self.previous, self.psi, self.cbase, self.system, self.frame_ids,
                               self.poses, self.read_mask, self.write_mask, self.revision, self.runtime_id, True,
                               self.prefill_mode, begin_id=self.begin_id, psi_snapshot=self.psi_snapshot,
                               write_history=self.write_history, live_factors=self.live_factors,
                               local_stats=self.local_stats, ablation=self.ablation,
                               collect_local_stats=self.collect_local_stats, local_trajectory=self.local_trajectory,
                               sink_options=self.sink_options, sink_reference=self.sink_reference,
                               sink_shift=self.sink_shift, sink_trajectory=self.sink_trajectory,
                               sink_reference_sha256=self.sink_reference_sha256)

    def token_masks(self, n):
        f = self.frame_ids.shape[-1]
        if n % f: raise ValueError("token count must be divisible by frame count")
        return self.read_mask.repeat_interleave(n // f, -1), self.write_mask.repeat_interleave(n // f, -1)

    def support_query_masks(self, n, hw):
        """Checkerboard in actual T/H/W order with absolute latent frame IDs."""
        if hw is None or len(hw) != 3:
            raise ValueError("Local support requires the actual token grid (T,H,W)")
        t, h, w = hw
        if any(not isinstance(i, int) or i < 1 for i in hw) or t != self.frame_ids.shape[-1] or t*h*w != n:
            raise ValueError("Local token grid, frame IDs and token count disagree")
        xy = torch.arange(h, device=self.frame_ids.device)[:, None] + torch.arange(w, device=self.frame_ids.device)
        parity = ((self.frame_ids[..., None, None] + xy) % 2 == 0).flatten(1)
        write = self.token_masks(n)[1]
        return parity & write, ~parity & write

    def stage(self, index, state, gradient, stats):
        if not self.clean_mode: raise RuntimeError("temporary contexts cannot stage commits")
        if index in self.candidates: raise RuntimeError("anchor executed twice in clean transaction")
        self.candidates[index] = (state, gradient if self.system.config.persistent_meta else gradient.detach(), stats)


@dataclass
class TTNRuntimeState:
    """Per-episode S and psi, separate from offline model weights.

    transition_fast is psi: fast coefficients of the nonlinear Cayley
    transition, not an extra MLP's dense weight matrix. The analytic gradient
    trains these coefficients on the pre-Correct innovation loss at test time.
    Predict/Correct/Read use the old psi throughout a chunk. A successful clean
    commit installs the updated psi for the NEXT chunk's Predict; it does not
    recompute the current output. S and psi are separate recurrent states.
    """
    config: TTNConfig
    world_state: torch.Tensor
    transition_fast: torch.Tensor
    previous_committed_pose: torch.Tensor
    committed_frame_ids: list
    commit_count: int = 0
    predict_count: int = 0
    revision: int = 0
    prefilled: bool = False
    last_stats: dict = field(default_factory=dict)
    # Runtime-only interventions: never saved as offline model configuration.
    ablation: str = "full"
    diagnostics: bool = False
    collect_local_stats: bool = False
    # Diagnostic-only detached writes, never persistent learned state or checkpoint content.
    write_history: tuple = ()
    sink_options: sink.SinkOptions = field(default_factory=sink.SinkOptions)
    sink_reference: torch.Tensor | None = None
    sink_reference_sha256: str | None = None
    sink_reference_verified: bool | None = None

    @classmethod
    def create(cls, config, batch_size, device, *, ablation="full", diagnostics=False, collect_local_stats=False,
               sink_options=None):
        sink_options = sink.SinkOptions() if sink_options is None else sink_options
        sink.validate_sink(sink_options, config, ablation)
        if ablation not in ("full", "no-ttt", "identity", "no-local", "no-persistent"):
            raise ValueError("unknown TTN runtime ablation")
        if ablation in ("no-local", "no-persistent") and not (config.local_update or config.persistent_meta):
            raise ValueError("Local/Persistent contribution controls require Meta-TTT")
        if ablation != "full" and torch.is_grad_enabled():
            raise ValueError("TTN runtime ablations are inference-only")
        s = torch.zeros(batch_size,
                        5,
                        config.heads,
                        config.head_dim,
                        config.head_dim,
                        device=device,
                        dtype=torch.float32)
        psi = torch.zeros(batch_size, 5, config.heads, config.generators, device=device, dtype=torch.float32)
        pose = torch.eye(4, device=device).expand(batch_size, 4, 4).clone()
        return cls(config, s, psi, pose, [set() for _ in range(batch_size)],
                   ablation=ablation, diagnostics=diagnostics, collect_local_stats=collect_local_stats,
                   sink_options=sink_options)

    def reset(self):
        self.world_state = torch.zeros_like(self.world_state)
        self.transition_fast = torch.zeros_like(self.transition_fast)
        self.previous_committed_pose = torch.eye(4, device=self.world_state.device).expand(
            self.world_state.shape[0], 4, 4).clone()
        self.committed_frame_ids = [set() for _ in self.committed_frame_ids]
        self.commit_count = 0
        self.predict_count = 0
        self.revision += 1
        self.prefilled = False
        self.last_stats = {}
        self.write_history = ()
        self.sink_reference = None
        self.sink_reference_sha256 = None
        self.sink_reference_verified = None

    def begin_chunk(self, system, poses, intrinsics, frame_ids, valid_mask, width, height, prefill=False, *, rope=None):
        if system.config != self.config: raise ValueError("runtime/system configurations differ")
        b = self.world_state.shape[0]
        if frame_ids.ndim == 1: frame_ids = frame_ids[None].expand(b, -1)
        if frame_ids.shape != valid_mask.shape or poses.shape[:2] != frame_ids.shape or b != poses.shape[0]:
            raise ValueError("batch/frame dimensions disagree")
        if any(len(set(ids.tolist())) != ids.numel() for ids in frame_ids):
            raise ValueError("duplicate frame ids in chunk")
        write = valid_mask.bool().clone()
        for batch, committed in enumerate(self.committed_frame_ids):
            write[batch] &= torch.tensor([int(i) not in committed for i in frame_ids[batch].tolist()],
                                         device=write.device)
        if prefill and (self.prefilled or self.commit_count): raise RuntimeError("prefill already completed")
        previous_pose = poses[:, 0] if prefill else self.previous_committed_pose
        options = getattr(system, "ttn_execution", DEFAULT_EXECUTION)
        sink.validate_sink(self.sink_options, self.config, self.ablation, options)
        meta = self.config.local_update or self.config.persistent_meta
        validate_execution(options, self.config.stage, self.ablation, self.diagnostics, meta=meta)
        reference = shift = None
        if self.sink_options.active and not prefill and self.commit_count >= self.sink_options.start_chunk:
            if not self.prefilled: raise RuntimeError("sink requires a successful observed-frame prefill")
            shift = torch.zeros(b, device=frame_ids.device, dtype=torch.long)
            if self.sink_options.mode == "zero":
                reference = torch.zeros_like(self.world_state)
            else:
                if self.sink_reference is None: raise RuntimeError("missing protected prefill reference")
                reference = self.sink_reference
                if self.sink_options.position == "temporal-realign":
                    shift = (frame_ids.min(-1).values - 1).clamp_min(0)
                    reference = sink.realign_reference(reference, rope, shift)
        snapshot = None
        live_factors = None
        history = self.write_history
        with annotation(options, "Predict"), torch.autocast(device_type=self.world_state.device.type, enabled=False):
            if self.config.stage == "A" or prefill or self.ablation == "identity":
                cbase = torch.zeros_like(self.transition_fast)
                predicted = self.world_state
            else:
                cbase = system.controller(poses, intrinsics, previous_pose, write, width, height)
                coeff = cbase + (self.config.delta_psi * self.transition_fast.tanh()
                                 if self.config.stage == "C" and self.ablation in ("full", "no-local") else 0)
                live_factors = CayleyFactors(system.generators.u.float(), system.generators.v.float(), coeff)
                predicted = live_factors.right(self.world_state)
                with torch.no_grad():
                    history = tuple((raw, live_factors.right(aligned).detach()) for raw, aligned in history)
                if options.core_backend != "reference" and self.config.stage == "C":
                    snapshot = DetachedCayleySnapshot.from_live(live_factors, self.transition_fast, cbase,
                                                               (id(self), self.revision, self.predict_count + 1))
        self.predict_count += 1
        return TTNChunkContext(predicted,
                               self.world_state,
                               self.transition_fast,
                               cbase,
                               system,
                               frame_ids,
                               poses,
                               valid_mask.bool(),
                               write,
                               self.revision,
                               id(self),
                               prefill_mode=prefill, begin_id=self.predict_count, psi_snapshot=snapshot,
                               write_history=history, live_factors=live_factors if meta else None,
                               ablation=self.ablation, collect_local_stats=self.collect_local_stats or self.diagnostics,
                               sink_options=self.sink_options, sink_reference=reference, sink_shift=shift,
                               sink_reference_sha256=self.sink_reference_sha256)

    def prefill(self, context):
        if self.prefilled or self.commit_count: raise RuntimeError("initial image can only be prefilled once")
        if self.sink_options.active and not context.prefill_mode:
            raise RuntimeError("sink reference requires an observed-frame prefill transaction")
        self.commit_chunk(context)
        self.prefilled = True

    def commit_chunk(self, context):
        sink.validate_sink(self.sink_options, self.config, self.ablation,
                           getattr(context.system, "ttn_execution", DEFAULT_EXECUTION))
        if not context.clean_mode or context.revision != self.revision or context.runtime_id != id(self):
            raise RuntimeError("not a current clean transaction")
        if context.psi_snapshot is not None:
            context.psi_snapshot.validate(id(self), self.revision, self.predict_count)
        if set(context.candidates) != set(range(5)):
            raise RuntimeError("all five anchors must finish before committing")
        states = []
        gradients = []
        for i in range(5):
            s, g, _ = context.candidates[i]
            if s.shape != self.world_state[:, i].shape or g.shape != self.transition_fast[:, i].shape:
                raise ValueError("candidate dimensions mismatch")
            if s.dtype != torch.float32 or not torch.isfinite(s).all() or not torch.isfinite(g).all():
                raise ValueError("nonfinite or non-FP32 candidate")
            states.append(s)
            gradients.append(g)
        new_state = torch.stack(states, 1)
        reference = self.sink_reference
        reference_hash = self.sink_reference_sha256
        if self.sink_options.active and self.sink_options.mode == "protected" and context.prefill_mode:
            if self.prefilled or self.commit_count: raise RuntimeError("sink reference can only be captured once")
            reference = sink.freeze_reference(new_state)
            reference_hash = sink.reference_sha256(reference)
        grad = torch.stack(gradients, 1)
        clipped, scale = clip_inner_gradient(grad, self.config.inner_clip, self.config.eps)
        psi = (self.transition_fast - self.config.eta_psi * clipped
               if self.config.stage == "C" and self.ablation in ("full", "no-local") else torch.zeros_like(self.transition_fast))
        if not self.config.persistent_meta: psi = psi.detach()
        if psi.dtype != torch.float32 or not torch.isfinite(psi).all():
            raise ValueError("nonfinite or non-FP32 Persistent psi; transaction not committed")
        anchor_stats = [committed_anchor_stats(context.candidates[i][2], self.transition_fast[:, i], psi[:, i],
                        grad[:, i], scale[:, i], context.cbase[:, i], ANCHORS[i], context.prefill_mode) for i in range(5)]
        if self.sink_options.active:
            for i, stats in enumerate(anchor_stats):
                stats["sink"] = context.sink_stats.get(i, {"active": False, "mode": self.sink_options.mode,
                    "gain": self.sink_options.gain, "reason": "prefill/before activation"})
                if context.sink_trajectory:
                    stats["sink_trajectory"] = [{key: value for key, value in call.items() if key != "anchors"} |
                        call["anchors"][i] for call in context.sink_trajectory if i in call["anchors"]]
        if self.config.local_update or self.config.persistent_meta:
            for i, stats in enumerate(anchor_stats):
                heads = stats["per_head"]
                stats["persistent"] = {"objective": "raw_weighted_innovation", "eta": self.config.eta_psi,
                    "meta_gradient_enabled": self.config.persistent_meta and grad.requires_grad and self.ablation in ("full", "no-local") and not context.prefill_mode,
                    "update_enabled": self.ablation in ("full", "no-local") and not context.prefill_mode,
                    "prefill": context.prefill_mode,
                    "per_head": {"raw_grad_norm": heads["inner_grad_norm"],
                                 "clipped_grad_norm": heads["inner_grad_clipped_norm"],
                                 "clip_scale": heads["inner_clip_scale"], "psi_norm": heads["psi_norm"],
                                 "update_norm": heads["psi_update_norm"],
                                 "tanh_saturation_fraction": (psi[:, i].detach().tanh().abs() > .99).float().mean(-1).cpu().tolist()}}
                stats["local"] = context.local_stats.get(i, {
                    "enabled": self.config.local_update and not context.prefill_mode and self.ablation in ("full", "no-persistent"),
                    "recorded": False, "reason": "prefill/clean-only" if context.prefill_mode else "telemetry disabled"})
                if context.local_trajectory:
                    stats["local_trajectory"] = [{key: value for key, value in call.items() if key != "anchors"} |
                        call["anchors"][i] for call in context.local_trajectory if i in call["anchors"]]
        history = context.write_history
        if not context.prefill_mode:
            correction = (new_state.detach() - context.predicted.detach()).clone()
            for i, stats in enumerate(anchor_stats):
                stats.update(write_direction_stats(correction[:, i],
                    tuple((raw[:, i], aligned[:, i]) for raw, aligned in history), self.config.eps))
            history = ((correction, correction), *history[:3])
        if self.diagnostics:
            from .stability import state_dynamics
            for i, stats in enumerate(anchor_stats):
                stats.update(state_dynamics(context.previous[:, i], context.predicted[:, i],
                                            new_state[:, i], self.config.eps),
                             inner_update_applied=self.config.stage == "C" and self.ablation in ("full", "no-local") and not context.prefill_mode)
        previous_pose = self.previous_committed_pose.clone()
        committed = [set(s) for s in self.committed_frame_ids]
        for b in range(new_state.shape[0]):
            indices = context.write_mask[b].nonzero().flatten()
            if indices.numel():
                previous_pose[b] = context.poses[b, indices[-1]].detach().float()
                committed[b].update(context.frame_ids[b, indices].tolist())
        # Assign only after every validation/computation succeeds. The context
        # keeps its old prediction/psi; the new psi is consumed next chunk.
        self.world_state = new_state
        self.sink_reference = reference
        self.sink_reference_sha256 = reference_hash
        self.write_history = history
        self.transition_fast = psi
        self.previous_committed_pose = previous_pose
        self.committed_frame_ids = committed
        self.commit_count += 1
        self.revision += 1
        self.last_stats = {
            "read_frames": context.read_mask.sum(-1).tolist(),
            "write_frames": context.write_mask.sum(-1).tolist(),
            "state_norm": float(new_state.detach().norm()),
            "state_rms": matrix_scale(new_state)["rms"],
            "psi_norm": float(psi.detach().norm()),
            "psi_update_norm": float((self.config.eta_psi * grad * scale).detach().norm())
                if self.config.stage == "C" and self.ablation in ("full", "no-local") else 0.,
            "prefill": context.prefill_mode,
            "anchors": anchor_stats
        }
        if self.diagnostics or self.ablation != "full": self.last_stats["ablation"] = self.ablation

    def detach(self):
        self.world_state = self.world_state.detach()
        self.transition_fast = self.transition_fast.detach()

    def verify_sink_reference(self):
        """Hash once at rollout end, never in each solver call."""
        if self.sink_reference is None: return None
        if sink.reference_sha256(self.sink_reference) != self.sink_reference_sha256:
            raise RuntimeError("protected sink reference changed during the episode")
        return self.sink_reference_sha256
