"""Explicit episode state and five-anchor clean transactions, independent of SANA caches."""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from .core import ANCHORS, TTNConfig, Rank2Generators, CayleyFactors, DetachedCayleySnapshot, clip_inner_gradient
from .performance import DEFAULT_EXECUTION, annotation, validate_execution
from .stability import committed_anchor_stats, matrix_scale, write_direction_stats
from .controller import TransitionController
from .event_memory import EventMemoryBank, EventMemoryOptions


class TTNSystem(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generators = Rank2Generators(config)
        self.controller = TransitionController(config)
        if config.local_update and not config.state_ttt_rank:
            # Constant initialization consumes no RNG and preserves inherited seeds.
            eta = config.local_eta_init
            logit = math.log(math.expm1(eta)) if eta < 20 else eta + math.log(-math.expm1(-eta))
            self.local_eta_logits = nn.Parameter(torch.full((5,), logit))
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
    sequence_mode: bool = False
    sequence_coefficients: tuple = ()
    camera_cached_chunks: int = 2
    checkpoint_cpu_inputs: bool = False
    measure_psi_output: bool = False
    sequence_memory_callback: object = None
    state_ttt_basis: torch.Tensor | None = None
    # Runtime-only inference state, never included in the model checkpoint.
    memory_bank: object = None
    memory_selections: dict = field(default_factory=dict)
    memory_pending: dict = field(default_factory=dict)
    memory_stats: list = field(default_factory=list)
    memory_sigma: float | None = None
    memory_call: int = 0

    def for_clean(self):
        return TTNChunkContext(self.predicted, self.previous, self.psi, self.cbase, self.system, self.frame_ids,
                               self.poses, self.read_mask, self.write_mask, self.revision, self.runtime_id, True,
                               self.prefill_mode, begin_id=self.begin_id, psi_snapshot=self.psi_snapshot,
                               write_history=self.write_history, live_factors=self.live_factors,
                               local_stats=self.local_stats, ablation=self.ablation,
                               collect_local_stats=self.collect_local_stats, local_trajectory=self.local_trajectory,
                               state_ttt_basis=self.state_ttt_basis,
                               memory_bank=self.memory_bank, memory_selections=self.memory_selections,
                               memory_stats=self.memory_stats)

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
    memory_options: object = None
    event_memory: object = None

    @classmethod
    def create(cls, config, batch_size, device, *, ablation="full", diagnostics=False,
               collect_local_stats=False, memory_options=None):
        if ablation not in ("full", "no-ttt", "identity", "no-local", "no-persistent"):
            raise ValueError("unknown TTN runtime ablation")
        if ablation in ("no-local", "no-persistent") and not (config.local_update or config.persistent_meta):
            raise ValueError("Local/Persistent contribution controls require Meta-TTT")
        if ablation != "full" and torch.is_grad_enabled():
            raise ValueError("TTN runtime ablations are inference-only")
        if memory_options is not None and not isinstance(memory_options, EventMemoryOptions):
            raise TypeError("memory_options must be EventMemoryOptions or None")
        active_memory = memory_options is not None and memory_options.active
        if active_memory and (torch.is_grad_enabled() or config.stage != 'C' or not config.state_ttt_rank
                              or ablation != 'full'):
            raise ValueError("event memory requires inference-only Stage C StateTTT with full ablation")
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
                   memory_options=memory_options,
                   event_memory=EventMemoryBank(memory_options, batch_size) if active_memory else None)

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
        if self.event_memory is not None:
            self.event_memory = EventMemoryBank(self.memory_options, self.world_state.shape[0])

    def begin_chunk(self, system, poses, intrinsics, frame_ids, valid_mask, width, height, prefill=False):
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
        meta = self.config.local_update or self.config.persistent_meta
        validate_execution(options, self.config.stage, self.ablation, self.diagnostics, meta=meta)
        snapshot = None
        live_factors = None
        memory_bank = self.event_memory
        if memory_bank is not None and (torch.is_grad_enabled() or options.core_backend != 'reference'):
            raise ValueError("event memory requires no-grad and the reference Correct/Read backend")
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
        if memory_bank is not None and live_factors is not None:
            memory_bank = memory_bank.with_transport({anchor: (live_factors.p[i], live_factors.l[:, i])
                for i, anchor in enumerate(ANCHORS) if anchor in memory_bank.options.anchors})
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
                               memory_bank=memory_bank)

    def prefill(self, context):
        if self.prefilled or self.commit_count: raise RuntimeError("initial image can only be prefilled once")
        self.commit_chunk(context)
        self.prefilled = True

    def commit_chunk(self, context):
        if not context.clean_mode or context.revision != self.revision or context.runtime_id != id(self):
            raise RuntimeError("not a current clean transaction")
        if context.psi_snapshot is not None:
            context.psi_snapshot.validate(id(self), self.revision, self.predict_count)
        if set(context.candidates) != set(range(5)):
            raise RuntimeError("all five anchors must finish before committing")
        next_memory = self.event_memory
        if next_memory is not None:
            if torch.is_grad_enabled() or context.sequence_mode or context.begin_id != self.predict_count:
                raise ValueError("event memory cannot commit training, recomputed or stale contexts")
            if context.memory_bank is None:
                raise ValueError("missing event memory snapshot")
            next_memory = context.memory_bank.with_pending(context.memory_pending,
                self.commit_count, source='observed' if context.prefill_mode else 'generated')
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
        grad = torch.stack(gradients, 1)
        clipped, scale = clip_inner_gradient(grad, self.config.inner_clip, self.config.eps)
        psi = (self.transition_fast - self.config.eta_psi * clipped
               if self.config.stage == "C" and self.ablation in ("full", "no-local") else torch.zeros_like(self.transition_fast))
        if not self.config.persistent_meta: psi = psi.detach()
        if psi.dtype != torch.float32 or not torch.isfinite(psi).all():
            raise ValueError("nonfinite or non-FP32 Persistent psi; transaction not committed")
        anchor_stats = [committed_anchor_stats(context.candidates[i][2], self.transition_fast[:, i], psi[:, i],
                        grad[:, i], scale[:, i], context.cbase[:, i], ANCHORS[i], context.prefill_mode) for i in range(5)]
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
        self.write_history = history
        self.transition_fast = psi
        self.previous_committed_pose = previous_pose
        self.committed_frame_ids = committed
        self.event_memory = next_memory
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
        if next_memory is not None:
            self.last_stats['event_memory'] = {'size': len(next_memory), 'calls': list(context.memory_stats)}

    def detach(self):
        self.world_state = self.world_state.detach()
        self.transition_fast = self.transition_fast.detach()
