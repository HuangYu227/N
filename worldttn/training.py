"""Causal clean-history training, one optimizer step per clip, explicit TBPTT windows."""
from contextlib import nullcontext
import weakref
from dataclasses import dataclass, field
import torch
from .session import TTNSession, carry_cache
from .cuda_debug import trace_backward


def activation_storage(mode, *, device=None):
    """Snapshot saved tensors before FSDP resharding; never replay TTN forwards.

    Native save_on_cpu copies values (including weight views) during forward,
    so backward does not retain/read freed FSDP all-gather storage. FSDP's own
    parameter hooks still unshard and reduce the actual parameter gradients.
    """
    if mode not in ("none", "cpu", "cpu-pageable"):
        raise ValueError("activation_offload must be none or cpu or cpu-pageable")
    if mode == "none": return nullcontext()
    if mode in ("cpu", "cpu-pageable"):
        # CPU tensors already live on the host. Pinned packing would only
        # change their strides and can change backward rounding on GPU hosts.
        if device is not None and torch.device(device).type == "cpu": return nullcontext()
        if mode == "cpu":
            storage = torch.autograd.graph.save_on_cpu(pin_memory=True)
        else:
            snapshots = {}
            def pack_to_cpu(tensor):
                # Frozen host-side teacher features already have stable storage.
                if tensor.device.type == "cpu" and not tensor.requires_grad:
                    return tensor.device, tensor.detach()
                root = tensor._base if tensor._base is not None else tensor
                # Shared activations can reuse snapshots; leaf/FSDP parameters must always copy.
                key = None
                if root.grad_fn is not None and not tensor.is_inference():
                    key = (id(root), tensor._version, tensor.storage_offset(), tuple(tensor.shape),
                           tuple(tensor.stride()), tensor.dtype, tensor.device)
                    cached = snapshots.get(key)
                    if cached is not None and cached[0]() is root:
                        packed = cached[1]()
                        if packed is not None: return tensor.device, packed
                # Independent contiguous snapshots survive FSDP all-gather storage release.
                packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
                                     device="cpu", pin_memory=False)
                packed.copy_(tensor)
                if key is not None: snapshots[key] = (weakref.ref(root), weakref.ref(packed))
                return tensor.device, packed
            def unpack_from_cpu(packed):
                device, tensor = packed
                return tensor.to(device, non_blocking=False)
            storage = torch.autograd.graph.saved_tensors_hooks(pack_to_cpu, unpack_from_cpu)
        return storage


def _memory_phase(callback, phase, **info):
    if callback is not None: callback(phase, **info)


def chunk_ranges(frames, chunk_size=3):
    if frames < 1 or chunk_size < 1: raise ValueError("positive frame count/chunk size required")
    starts = [0]
    end = min(frames, chunk_size + 1)
    starts.append(end)
    while end < frames:
        end = min(frames, end + chunk_size)
        starts.append(end)
    return list(zip(starts[:-1], starts[1:]))


def linear_flow_loss(session,
                     clean,
                     t,
                     noise,
                     y,
                     context,
                     cache,
                     start,
                     end,
                     mask,
                     data_info,
                     loss_mask,
                     on_prediction=None):
    """Unshifted synthetic reference for CPU tests; production uses SANAFlowLoss."""
    sigma = t.float() / 1000
    if sigma.ndim == 1: sigma = sigma[:, None, None]
    noisy = (1 - sigma[:, :, :, None, None]) * clean + sigma[:, :, :, None, None] * noise
    output, _ = session.forward(noisy, t, y, context, cache, start, end, mask, data_info)
    if on_prediction is not None: on_prediction(output)
    error = (output.float() - (noise - clean).float()).square()
    lm = loss_mask.expand_as(error)
    return (error * lm).flatten(1).sum(-1) / lm.flatten(1).sum(-1).clamp_min(1)


class SANAFlowLoss:
    """Reuse SANA q_sample, timestep mapping, flow target and masked reduction."""

    def __init__(self, config):
        from diffusion import Scheduler
        self.scheduler = Scheduler(str(config.scheduler.train_sampling_steps),
                                   noise_schedule=config.scheduler.noise_schedule,
                                   predict_flow_v=True,
                                   learn_sigma=False,
                                   pred_sigma=False,
                                   snr=False,
                                   flow_shift=config.scheduler.flow_shift)

    def __call__(self,
                 session,
                 clean,
                 t,
                 noise,
                 y,
                 context,
                 cache,
                 start,
                 end,
                 mask,
                 data_info,
                 loss_mask,
                 on_prediction=None):
        config = context.system.config  # Teacher replay intentionally has no TTN module.
        noise_info = {}
        if context.collect_local_stats and (config.local_update or config.persistent_meta):
            noise_info = {"noise_sigma": torch.as_tensor(self.scheduler.sigmas, device=clean.device,
                                                        dtype=torch.float32)[t.long()], "sampled_timestep": t}
        def model_call(x, timestep, **unused):
            result, _ = session.forward(x, timestep, y, context, cache, start, end, mask, data_info, **noise_info)
            if on_prediction is not None: on_prediction(result)
            return result

        return self.scheduler.training_losses(model_call,
                                              clean,
                                              t,
                                              model_kwargs={"data_info": {}},
                                              noise=noise,
                                              loss_mask=loss_mask)["loss"].reshape(clean.shape[0])


@dataclass
class ClipTrainingContext:
    session: TTNSession
    clean: torch.Tensor
    y: torch.Tensor
    timesteps: torch.Tensor
    noise: torch.Tensor
    loss_valid: torch.Tensor
    total: torch.Tensor
    ranges: list
    cache: list
    mask: object = None
    data_info: object = None
    records: list = field(default_factory=list)
    total_loss: float = 0.
    memory_callback: object = None
    prefill_stats: dict = field(default_factory=dict)
    full_gradient: bool = False
    cached_chunks: int = -1


class TTNTrainingWindow(torch.nn.Module):
    """One hooked module call covers Predict and all noisy/clean work in a window.

    The episode is rank-local Python state, never a parameter or buffer. In
    particular, controller/generator use must not bypass a DDP/FSDP boundary.
    Prefill also enters this boundary, using no-grad and scratch caches.
    """

    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, episode, first=0, last=0, *, prefill=False, on_prediction=None):
        if episode.full_gradient:
            if prefill: raise ValueError("full-gradient prefill belongs inside the episode forward")
            from .fullgradient import full_gradient_forward
            return full_gradient_forward(self, episode, on_prediction)
        session = episode.session
        if prefill:
            _memory_phase(episode.memory_callback, "prefill_begin")
            session.prefill(episode.clean[:, :, :1], episode.y, episode.mask, episode.data_info)
            episode.prefill_stats = dict(session.runtime.last_stats, chunk=-1, start=0, end=1)
            _memory_phase(episode.memory_callback, "prefill_end")
            return episode.clean.new_zeros(())
        losses_in_window = []
        from .performance import DEFAULT_EXECUTION, annotation
        execution = getattr(self.model.ttn_system, "ttn_execution", DEFAULT_EXECUTION)
        for index in range(first, last):
            start, end = episode.ranges[index]
            context = session.begin_chunk(start, end)
            lm = episode.loss_valid[:, None, start:end, None, None].to(torch.float32)
            callback = (lambda output, i=index: on_prediction(i, output)) if on_prediction else None
            _memory_phase(episode.memory_callback, "noisy_begin", chunk=index, start=start, end=end)
            with annotation(execution, "NoisyForward"):
                losses = self.loss_fn(session, episode.clean[:, :, start:end], episode.timesteps[:, :, start:end],
                                      episode.noise[:, :, start:end], episode.y, context, episode.cache, start, end,
                                      episode.mask, episode.data_info, lm, callback)
            loss = (losses * (episode.loss_valid[:, start:end].sum(-1) / episode.total)).mean()
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite flow loss")
            losses_in_window.append(loss)
            episode.total_loss += float(loss.detach())
            _memory_phase(episode.memory_callback, "noisy_end", chunk=index, start=start, end=end)
            # The current GT clean chunk becomes history only AFTER its noisy prediction.
            _memory_phase(episode.memory_callback, "clean_begin", chunk=index, start=start, end=end)
            with annotation(execution, "CleanForward"):
                _, episode.cache = session.clean_forward(episode.clean[:, :, start:end], episode.y, context,
                                                          episode.cache, start, end, episode.mask, episode.data_info)
            episode.records.append(dict(session.runtime.last_stats, chunk=index, start=start, end=end))
            _memory_phase(episode.memory_callback, "clean_end", chunk=index, start=start, end=end)
        return torch.stack(losses_in_window).sum()


def train_clip(model,
               clean,
               y,
               camera_conditions,
               optimizer,
               loss_fn,
               timesteps,
               noise,
               *,
               width,
               height,
               mask=None,
               data_info=None,
               extras=None,
               valid_mask=None,
               tbptt=2,
               chunk_size=3,
               outer_clip=.5,
               on_prediction=None,
               window_model=None,
               parallel=None,
               activation_offload="none",
               memory_callback=None,
               audit_update=False,
               cached_chunks=-1):
    if getattr(model, 'ttn_training_protocol', '') == 'dual-psi-clean-fullgrad-v1':
        from .legacy_sequence_training import train_legacy_sequence
        return train_legacy_sequence(model, clean, y, camera_conditions, optimizer, loss_fn, timesteps, noise,
            width=width, height=height, mask=mask, data_info=data_info, extras=extras, valid_mask=valid_mask,
            tbptt=tbptt, chunk_size=chunk_size, outer_clip=outer_clip, on_prediction=on_prediction,
            window_model=window_model, parallel=parallel, activation_offload=activation_offload,
            memory_callback=memory_callback, audit_update=audit_update, cached_chunks=cached_chunks)
    if type(tbptt) is not int or tbptt not in (0, 1, 2, 4): raise ValueError("TBPTT supports 0 (full episode),1,2,4")
    if tbptt == 0 and parallel is not None and parallel.mode == 'fsdp2':
        raise ValueError('full-gradient FSDP2 requires dual-psi-clean-fullgrad-v1 layer scan')
    if activation_offload not in ("none", "cpu", "cpu-pageable"): raise ValueError("invalid activation_offload")
    if type(cached_chunks) is not int or cached_chunks not in (-1, 1, 2): raise ValueError("cached_chunks must be -1,1,2")
    b, _, frames, _, _ = clean.shape
    if timesteps.ndim == 1: timesteps = timesteps[:, None, None].expand(b, 1, frames).clone()
    if timesteps.shape != (b, 1, frames) or noise.shape != clean.shape:
        raise ValueError("noise/timestep dimensions mismatch")
    timesteps = timesteps.clone()
    timesteps[:, :, 0] = 0
    valid = torch.ones(b, frames, dtype=torch.bool, device=clean.device) if valid_mask is None else valid_mask.bool()
    if valid.shape != (b, frames): raise ValueError("valid mask dimensions mismatch")
    loss_valid = valid.clone()
    loss_valid[:, 0] = False
    total = loss_valid.sum(-1).clamp_min(1)
    session = TTNSession(model, camera_conditions, width, height, valid, extras, collect_local_stats=True)
    runtime = session.reset(b)
    optimizer.zero_grad(set_to_none=True)
    model.eval()  # disables stochastic conditioning; does NOT disable differentiation
    episode = ClipTrainingContext(session, clean, y, timesteps, noise, loss_valid, total,
                                  chunk_ranges(frames, chunk_size), [[None] * 10 for _ in model.blocks], mask, data_info,
                                  memory_callback=memory_callback, full_gradient=tbptt == 0,
                                  cached_chunks=cached_chunks)
    model.ttn_fullgrad_cpu_inputs = clean.is_cuda and activation_offload in ("cpu", "cpu-pageable")
    runner = parallel.window if parallel else window_model
    if runner is None: runner = TTNTrainingWindow(model, loss_fn)
    if parallel: parallel.validate_schedule(frames, tbptt, len(episode.ranges), b)
    # PyTorch 2.9 caches FP32 -> BF16 weights without grad_fn when their first
    # use is under no_grad. Prefill and training share the caller's AMP scope,
    # so keep prefill out of that weight cache; training still uses caching.
    if tbptt != 0:
      with torch.no_grad(), torch.autocast(device_type=clean.device.type,
                                        enabled=torch.is_autocast_enabled(clean.device.type),
                                        dtype=torch.get_autocast_dtype(clean.device.type),
                                        cache_enabled=False):
        runner(episode, prefill=True)
      if parallel: parallel.reshard()
      runtime.detach()  # legacy prefill boundary
    window = tbptt or len(episode.ranges)
    for first in range(0, len(episode.ranges), window):
        last = min(first + window, len(episode.ranges))
        sync = parallel.accumulation(last == len(episode.ranges)) if parallel else nullcontext()
        # FSDP swaps full parameter storage between chunk forwards. Cached AMP
        # weight casts share one backward node across those forwards and can
        # defer weight gradients past FSDP's per-forward reduction hooks.
        amp = torch.autocast(device_type=clean.device.type,
                             enabled=torch.is_autocast_enabled(clean.device.type),
                             dtype=torch.get_autocast_dtype(clean.device.type),
                             cache_enabled=False) if tbptt == 0 or parallel is not None and parallel.mode == "fsdp2" else nullcontext()
        # DDP no_sync still encloses both forward and backward. CPU storage
        # preserves the full window's S/optional Persistent-meta graph.
        with sync, amp, activation_storage(activation_offload):
            loss = runner(episode, first, last, on_prediction=on_prediction)
            _memory_phase(memory_callback, "backward_begin", first=first, last=last)
            from .performance import DEFAULT_EXECUTION, annotation
            execution = getattr(model.ttn_system, "ttn_execution", DEFAULT_EXECUTION)
            with annotation(execution, "Backward"), trace_backward(loss):
                loss.backward()
            _memory_phase(memory_callback, "backward_end", first=first, last=last)
        runtime.detach()
        episode.cache = carry_cache(episode.cache, model.ttn_system.config.camera_attention)
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = parallel.clip_grad_norm(outer_clip) if parallel else torch.nn.utils.clip_grad_norm_(
        params, outer_clip, error_if_nonfinite=True)
    _memory_phase(memory_callback, "optimizer_begin")
    probe = None
    if audit_update:
        from .training_health import FirstUpdateProbe
        probe = FirstUpdateProbe(model)
    with annotation(execution, "Optimizer"):
        optimizer.step()  # slow parameters stay fixed throughout every clip's windows
    updates = probe.report(optimizer) if probe is not None else None
    _memory_phase(memory_callback, "optimizer_end")
    result = {"loss": episode.total_loss, "outer_grad_norm": float(grad_norm), "runtime": runtime,
              "chunks": episode.records, "prefill": episode.prefill_stats,
              "exposure": {"clips": b, "valid_latent_frames": int(valid.sum()),
                  "predicted_latent_frames": int(loss_valid.sum()),
                  "predicted_chunks": sum(int((loss_valid[:, start:end].any(-1)).sum()) for start, end in episode.ranges)}}
    if tbptt == 0:
        result["sequence"] = {"protocol": "dual-psi-clean-fullgrad-v1", "backward_calls": 1,
                              "temporal_detaches": 0, "cached_chunks": cached_chunks,
                              "live_prefill": True}
    if updates is not None: result["optimizer_updates"] = updates
    return result
