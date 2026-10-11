"""One noisy sequence loss and one backward, with no temporal gradient truncation."""
from types import SimpleNamespace
from collections import Counter
import os

import torch
from torch import nn

from .session import TTNSession
from .sequence import sequence_ranges
from .diagnostics import replay_memory_trace
from .training import activation_storage, activation_storage_stats, _memory_phase


class SequenceSession(TTNSession):
    def forward(self, x, t, y, context, cache, start, end, *args, **kwargs):
        # Scheduler index zero need not map to exactly zero sigma. The observed frame is exact.
        x = torch.cat((self.observed, x[:, :, 1:]), dim=2)
        t = t.clone()
        t[:, :, 0] = 0
        kwargs["save"] = True
        return super().forward(x, t, y, context, cache, start, end, *args, **kwargs)


class TTNSequenceTraining(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model, self.loss_fn = model, loss_fn

    def forward(self, episode):
        session, clean = episode.session, episode.clean
        context = episode.context
        losses = self.loss_fn(session, clean, episode.timesteps, episode.noise, episode.y, context,
            [[None]*10 for _ in self.model.blocks], 0, clean.shape[2], episode.mask, episode.data_info,
            episode.loss_mask, episode.on_prediction)
        return losses.mean()


def train_sequence(model, clean, y, camera_conditions, optimizer, loss_fn, timesteps, noise, *,
                   width, height, mask=None, data_info=None, extras=None, valid_mask=None, tbptt=0,
                   chunk_size=3, outer_clip=.5, on_prediction=None, window_model=None, parallel=None,
                   activation_offload="none", activation_gpu_budget_gib=0., memory_callback=None,
                   audit_update=False, history_training=None):
    if tbptt != 0 or chunk_size != 3:
        raise ValueError("noisy full-sequence training requires tbptt=0 and three-frame generation groups")
    if model.ttn_system.config.memory_granularity != "frame":
        raise ValueError("full-sequence training requires frame Direct-S memory")
    if history_training is not None:
        raise ValueError("full-sequence training does not use clean/generated training-history rollouts")
    b, _, frames, _, _ = clean.shape
    ranges = sequence_ranges(frames)
    if noise.shape != clean.shape or timesteps.shape != (b, 1, frames):
        raise ValueError("full-sequence noise/timestep dimensions mismatch")
    valid = torch.ones(b, frames, dtype=torch.bool, device=clean.device) if valid_mask is None else valid_mask.bool()
    if valid.shape != (b, frames) or not valid.all():
        raise ValueError("full-sequence training currently requires complete unpadded clips")
    if set(extras or {}) - {"chunk_plucker"} or data_info and any(
            name in data_info for name in ("image_vae_embeds", "image_embeds")):
        raise ValueError("full-sequence training supports initial-latent and UCPE/Plucker conditioning only")
    session = SequenceSession(model, camera_conditions, width, height, valid, extras, collect_local_stats=False)
    runtime = session.reset(b)
    session.observed = clean[:, :, :1]
    context = session.begin_chunk(0, frames)
    context.sequence_mode = True
    loss_mask = valid[:, None, :, None, None].float()
    loss_mask[:, :, 0] = 0
    episode = SimpleNamespace(session=session, context=context, clean=clean, noise=noise, timesteps=timesteps,
        y=y, mask=mask, data_info=data_info, loss_mask=loss_mask, on_prediction=on_prediction)
    runner = parallel.window if parallel else window_model or TTNSequenceTraining(model, loss_fn)
    if parallel:
        parallel.validate_schedule(frames, 0, len(ranges), b)
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    amp = torch.autocast(device_type=clean.device.type, enabled=torch.is_autocast_enabled(clean.device.type),
        dtype=torch.get_autocast_dtype(clean.device.type), cache_enabled=False)
    offload_audit = Counter() if memory_callback and activation_offload != "none" else None
    def phase(name, **info):
        if offload_audit is not None:
            info["offload_saved_tensors"] = activation_storage_stats(offload_audit)
        _memory_phase(memory_callback, name, **info)
    with replay_memory_trace(memory_callback if os.environ.get("TTN_REPLAY_MEMORY_TRACE") == "1" else None), amp, activation_storage(activation_offload, device=clean.device, audit=offload_audit,
                                 gpu_budget_gib=activation_gpu_budget_gib):
        phase("sequence_begin", frames=frames, tbptt=0)
        loss = runner(episode)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite full-sequence loss")
        phase("backward_begin")
        loss.backward()
        phase("backward_end")
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = (parallel.clip_grad_norm(outer_clip) if parallel else
                 torch.nn.utils.clip_grad_norm_(params, outer_clip, error_if_nonfinite=True))
    probe = None
    if audit_update:
        from .training_health import FirstUpdateProbe
        probe = FirstUpdateProbe(model)
    _memory_phase(memory_callback, "optimizer_begin")
    optimizer.step()
    _memory_phase(memory_callback, "optimizer_end")
    if set(context.candidates) != set(range(5)) or set(context.memory_stats) != set(range(5)):
        raise RuntimeError("full-sequence forward did not publish all five anchors")
    # Episode ends here. These detached states are diagnostics, never a clean inference cache.
    runtime.world_state = torch.stack([context.candidates[i][0].detach() for i in range(5)], 1)
    chunks = [{"chunk": j-1, "start": start, "end": end, "write_frames": [end-start]*b,
               "history_source": "noisy", "anchors": [context.memory_stats[i][j] for i in range(5)]}
              for j, (start, end) in enumerate(ranges) if j]
    prefill = {"chunk": -1, "start": 0, "end": 1, "write_frames": [1]*b,
               "prefill": True, "history_source": "observed", "mode": "isolated_in_each_layer",
               "live_gradient": True, "anchors": [context.memory_stats[i][0] for i in range(5)]}
    result = {"loss": float(loss.detach()), "outer_grad_norm": float(grad_norm), "runtime": runtime,
        "chunks": chunks, "prefill": prefill,
        "training_protocol": "frame-noisy-fullgrad-v1", "activation_offload": activation_offload,
        "exposure": {"clips": b, "valid_latent_frames": b*frames, "predicted_latent_frames": b*(frames-1),
                     "predicted_chunks": b*(len(ranges)-1), "model_forwards": 1,
                     "frame_state_updates_per_anchor": b*frames,
                     "clean_history_forwards": 0, "backward_calls": 1, "temporal_detaches": 0}}
    if probe is not None:
        result["optimizer_updates"] = probe.report(optimizer)
    return result
