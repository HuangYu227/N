"""Explicit episode state and five-anchor clean transactions, independent of SANA caches."""
from dataclasses import dataclass, field
import torch
from torch import nn
from .core import ANCHORS, TTNConfig, Rank2Generators, CayleyFactors
from .stability import committed_anchor_stats, matrix_scale
from .controller import TransitionController


class TTNSystem(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generators = Rank2Generators(config)
        self.controller = TransitionController(config)
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

    def for_clean(self):
        return TTNChunkContext(self.predicted, self.previous, self.psi, self.cbase, self.system, self.frame_ids,
                               self.poses, self.read_mask, self.write_mask, self.revision, self.runtime_id, True,
                               self.prefill_mode)

    def token_masks(self, n):
        f = self.frame_ids.shape[-1]
        if n % f: raise ValueError("token count must be divisible by frame count")
        return self.read_mask.repeat_interleave(n // f, -1), self.write_mask.repeat_interleave(n // f, -1)

    def stage(self, index, state, gradient, stats):
        if not self.clean_mode: raise RuntimeError("temporary contexts cannot stage commits")
        if index in self.candidates: raise RuntimeError("anchor executed twice in clean transaction")
        self.candidates[index] = (state, gradient.detach(), stats)


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

    @classmethod
    def create(cls, config, batch_size, device):
        s = torch.zeros(batch_size,
                        5,
                        config.heads,
                        config.head_dim,
                        config.head_dim,
                        device=device,
                        dtype=torch.float32)
        psi = torch.zeros(batch_size, 5, config.heads, config.generators, device=device, dtype=torch.float32)
        pose = torch.eye(4, device=device).expand(batch_size, 4, 4).clone()
        return cls(config, s, psi, pose, [set() for _ in range(batch_size)])

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
        with torch.autocast(device_type=self.world_state.device.type, enabled=False):
            if self.config.stage == "A" or prefill:
                cbase = torch.zeros_like(self.transition_fast)
                predicted = self.world_state
            else:
                cbase = system.controller(poses, intrinsics, previous_pose, write, width, height)
                coeff = cbase + (self.config.delta_psi * self.transition_fast.tanh() if self.config.stage == "C" else 0)
                factors = CayleyFactors(system.generators.u.float(), system.generators.v.float(), coeff)
                predicted = factors.right(self.world_state)
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
                               prefill_mode=prefill)

    def prefill(self, context):
        if self.prefilled or self.commit_count: raise RuntimeError("initial image can only be prefilled once")
        self.commit_chunk(context)
        self.prefilled = True

    def commit_chunk(self, context):
        if not context.clean_mode or context.revision != self.revision or context.runtime_id != id(self):
            raise RuntimeError("not a current clean transaction")
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
        grad = torch.stack(gradients, 1)
        scale = (self.config.inner_clip / grad.norm(dim=-1, keepdim=True).clamp_min(self.config.eps)).clamp_max(1)
        psi = (self.transition_fast -
               self.config.eta_psi * grad * scale).detach() if self.config.stage == "C" else torch.zeros_like(
                   self.transition_fast)
        anchor_stats = [committed_anchor_stats(context.candidates[i][2], self.transition_fast[:, i], psi[:, i],
                        grad[:, i], scale[:, i], context.cbase[:, i], ANCHORS[i], context.prefill_mode) for i in range(5)]
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
            "psi_norm": float(psi.norm()),
            "psi_update_norm": float((self.config.eta_psi * grad * scale).norm()) if self.config.stage == "C" else 0.,
            "prefill": context.prefill_mode,
            "anchors": anchor_stats
        }

    def detach(self):
        self.world_state = self.world_state.detach()
        self.transition_fast = self.transition_fast.detach()
