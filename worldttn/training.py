"""Causal clean-history training, one optimizer step per clip, explicit TBPTT windows."""
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass, field, fields, replace
import math
import torch
from .session import TTNSession, carry_cache
from .cuda_debug import trace_backward


def history_settings(value=None):
    """An explicit training protocol, included verbatim in exact-resume identity."""
    defaults = {"source": "gt", "steps": 4, "flow_shift": 9.8, "cached_chunks": -1, "cfg_scale": 1.}
    value = {} if value is None else value
    if not isinstance(value, dict): raise ValueError("history_training must be a mapping")
    if set(value) - set(defaults): raise ValueError("unknown history_training setting")
    result = defaults | value
    if result["source"] not in ("gt", "generated"): raise ValueError("history source must be gt or generated")
    if isinstance(result["steps"], bool) or not isinstance(result["steps"], int) or result["steps"] < 1:
        raise ValueError("history steps must be a positive integer")
    if (isinstance(result["flow_shift"], bool) or not isinstance(result["flow_shift"], (int, float))
            or not math.isfinite(result["flow_shift"]) or result["flow_shift"] <= 0):
        raise ValueError("history flow shift must be positive and finite")
    if (isinstance(result["cached_chunks"], bool) or not isinstance(result["cached_chunks"], int)
            or result["cached_chunks"] not in (-1, 1, 2)):
        raise ValueError("history cached_chunks must be -1, 1 or 2")
    if isinstance(result["cfg_scale"], bool) or result["cfg_scale"] != 1.:
        raise ValueError("generated-history training currently requires conditional CFG=1")
    return result


def _history_scheduler(steps, shift, device):
    # Same installed scheduler and per-token sign convention as the native sampler.
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


@torch.no_grad()
def generate_history_chunk(session, initial_noise, observed, y, context, cache, start, end,
                           mask, data_info, protocol):
    """Stop-gradient rollout; only the later clean transaction changes history.

    No future GT argument exists. The live training context and native caches
    are isolated so solver calls cannot overwrite their tensors or telemetry.
    """
    from .history import _clone_tree
    probe = replace(context, **{f.name: _clone_tree(getattr(context, f.name))
                                for f in fields(context) if f.name != "system"})
    probe.collect_local_stats = False
    scratch = _clone_tree(cache)
    x = initial_noise.detach().clone()
    if start == 0: x[:, :, :1] = observed
    scheduler = _history_scheduler(protocol["steps"], protocol["flow_shift"], x.device)
    b, c, frames, h, w = x.shape
    # A no-grad cast must never poison the following live AMP weight cache.
    with torch.autocast(device_type=x.device.type, enabled=torch.is_autocast_enabled(x.device.type),
                        dtype=torch.get_autocast_dtype(x.device.type), cache_enabled=False):
        for i, timestep in enumerate(scheduler.timesteps):
            t = timestep.to(x.device).float().expand(b, 1, frames).clone()
            if start == 0: t[:, :, 0] = 0
            sigma = scheduler.sigmas[i].to(x.device).expand_as(t).clone()
            if start == 0: sigma[:, :, 0] = 0
            prediction, _ = session.forward(x, t, y, probe, [list(s) for s in scratch], start, end,
                                            mask, data_info, noise_sigma=sigma)
            token_t = t[:, 0, :, None].expand(b, frames, h*w).reshape(b, -1)
            x = scheduler.step(-prediction.float().flatten(2).transpose(1, 2), timestep,
                               x.float().flatten(2).transpose(1, 2), per_token_timesteps=token_t,
                               return_dict=False)[0].transpose(1, 2).reshape(b, c, frames, h, w)
            if start == 0: x[:, :, :1] = observed
            if not torch.isfinite(x).all(): raise FloatingPointError("nonfinite generated training history")
    return x


def activation_storage(mode, *, device=None, audit=None, gpu_budget_gib=0.):
    """Snapshot saved tensors before FSDP resharding; never replay TTN forwards.

    Native save_on_cpu copies values (including weight views) during forward,
    so backward does not retain/read freed FSDP all-gather storage. FSDP's own
    parameter hooks still unshard and reduce the actual parameter gradients.
    A GPU budget keeps independent copies with the same contiguous layout as
    native pinned CPU packing. Its cumulative cap resets each TBPTT window;
    allocator, communication and backward workspace are outside this budget.
    """
    if mode not in ("none", "cpu"): raise ValueError("activation_offload must be none or cpu")
    if (isinstance(gpu_budget_gib, bool) or not isinstance(gpu_budget_gib, (int, float))
            or not math.isfinite(gpu_budget_gib) or gpu_budget_gib < 0):
        raise ValueError("activation GPU budget must be finite and nonnegative")
    if gpu_budget_gib and mode != "cpu":
        raise ValueError("activation GPU budget requires cpu offload")
    if mode == "none": return nullcontext()
    if mode == "cpu":
        # CPU tensors already live on the host. Pinned packing would only
        # change their strides and can change backward rounding on GPU hosts.
        if device is not None and torch.device(device).type == "cpu": return nullcontext()
        storage = torch.autograd.graph.save_on_cpu(pin_memory=True)
        if audit is not None or gpu_budget_gib:
            pack = storage.pack_hook
            budget = (int(gpu_budget_gib) * (1 << 30)
                      + int((gpu_budget_gib % 1) * (1 << 30)))
            spent = 0
            def record(tensor):
                nonlocal spent
                size = tensor.numel() * tensor.element_size()
                route = "cpu"
                if (gpu_budget_gib and tensor.device.type == "cuda" and tensor.layout == torch.strided
                        and size <= budget - spent):
                    # Match native pinned CPU packing's layout; snapshot FSDP weight views before reshard.
                    packed = (tensor.device, tensor.detach().clone(memory_format=torch.contiguous_format))
                    spent += size
                    route = "gpu"
                else:
                    packed = pack(tensor)
                if audit is not None:
                    key = (str(tensor.device), str(tensor.dtype), tuple(tensor.shape),
                           tuple(tensor.stride()), tensor.element_size())
                    if gpu_budget_gib: key += (route,)
                    audit[key] += 1
                return packed
            storage.pack_hook = record
        return storage


def activation_storage_stats(audit):
    """Cumulative saved-tensor copy payload per window, not live bytes or host RSS."""
    sizes = [(math.prod(key[2]) * key[4] * count, key, count) for key, count in audit.items()]
    gpu_sizes = [row for row in sizes if len(row[1]) > 5 and row[1][5] == "gpu"]
    gpu_bytes = sum(row[0] for row in gpu_sizes)
    total_bytes = sum(row[0] for row in sizes)
    return {"pack_calls": sum(audit.values()), "packed_tensor_bytes": total_bytes,
            "gpu_packed_tensor_bytes": gpu_bytes, "cpu_packed_tensor_bytes": total_bytes - gpu_bytes,
            "gpu_pack_calls": sum(row[2] for row in gpu_sizes),
            "largest_shapes": [{"bytes": size, "count": count, "device": key[0], "dtype": key[1],
                                "shape": list(key[2]), "stride": list(key[3]),
                                "saved_on": key[5] if len(key) > 5 else "cpu"}
                               for size, key, count in sorted(sizes, reverse=True)[:8]],
            "measurement": "TBPTT-window cumulative saved-tensor copy payload; not live memory or RSS"}


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
        if context.collect_local_stats and (config.local_update or config.persistent_meta
                or getattr(config, "memory_update", "delta") == "proximal"):
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
    history: dict = field(default_factory=history_settings)
    history_noise: torch.Tensor | None = None


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
        def boundary_backward(unused):
            raise RuntimeError("TBPTT-boundary clean history must be detached before a future loss")
        for index in range(first, last):
            start, end = episode.ranges[index]
            context = session.begin_chunk(start, end)
            if episode.history["source"] == "generated":
                _memory_phase(episode.memory_callback, "history_generate_begin", chunk=index)
                clean_history = generate_history_chunk(session, episode.history_noise[:, :, start:end],
                    episode.clean[:, :, :1], episode.y, context, episode.cache, start, end,
                    episode.mask, episode.data_info, episode.history)
                _memory_phase(episode.memory_callback, "history_generate_end", chunk=index)
            else:
                clean_history = episode.clean[:, :, start:end]
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
            # Only clean updates with a later loss inside this window need a graph.
            # Keep grad mode: native SANA selects different forward kernels under no_grad.
            # Discard only this unused graph's saved tensors, then detach its published state.
            future_credit = index + 1 < last
            _memory_phase(episode.memory_callback, "clean_begin", chunk=index, start=start, end=end)
            boundary = (nullcontext() if future_credit else torch.autograd.graph.saved_tensors_hooks(
                lambda tensor: None, boundary_backward))
            with boundary, annotation(execution, "CleanForward"):
                _, episode.cache = session.clean_forward(clean_history, episode.y, context,
                                                          episode.cache, start, end, episode.mask, episode.data_info,
                                                          cached_chunks=episode.history["cached_chunks"])
            if not future_credit: session.runtime.detach()
            episode.records.append(dict(session.runtime.last_stats, chunk=index, start=start, end=end,
                history_source=episode.history["source"], clean_future_credit=future_credit,
                history_generation_steps=episode.history["steps"] if episode.history_noise is not None else 0,
                clean_history_rms=float(clean_history.detach().float().square().mean().sqrt())))
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
               activation_gpu_budget_gib=0.,
               memory_callback=None,
               audit_update=False,
               history_training=None):
    if tbptt not in (1, 2, 4): raise ValueError("reference TBPTT supports K=1,2,4")
    activation_storage(activation_offload, device="cpu", gpu_budget_gib=activation_gpu_budget_gib)
    b, _, frames, _, _ = clean.shape
    history = history_settings(history_training if history_training is not None
                               else getattr(model, "ttn_history_training", None))
    if history["source"] == "generated" and (frames < 4 or frames % 3 != 1 or chunk_size != 3):
        raise ValueError("generated history requires 1+3n frames and chunk size 3")
    if history["source"] == "generated" and data_info and any(
            key in data_info for key in ("image_vae_embeds", "image_embeds")):
        raise ValueError("generated history accepts initial latent conditioning only; image extras need a leakage audit")
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
    offload_audit = Counter() if memory_callback is not None and activation_offload == "cpu" else None
    if offload_audit is not None:
        callback = memory_callback
        def traced(phase, **info):
            callback(phase, offload_saved_tensors=activation_storage_stats(offload_audit), **info)
        memory_callback = traced
    session = TTNSession(model, camera_conditions, width, height, valid, extras, collect_local_stats=True)
    runtime = session.reset(b)
    optimizer.zero_grad(set_to_none=True)
    model.eval()  # disables stochastic conditioning; does NOT disable differentiation
    episode = ClipTrainingContext(session, clean, y, timesteps, noise, loss_valid, total,
                                  chunk_ranges(frames, chunk_size), [[None] * 10 for _ in model.blocks], mask, data_info,
                                  memory_callback=memory_callback)
    episode.history = history
    # Independent of flow-supervision noise. Global RNG is saved by the existing checkpoint path.
    if history["source"] == "generated": episode.history_noise = torch.randn_like(clean)
    runner = parallel.window if parallel else window_model
    if runner is None: runner = TTNTrainingWindow(model, loss_fn)
    if parallel: parallel.validate_schedule(frames, tbptt, len(episode.ranges), b)
    # PyTorch 2.9 caches FP32 -> BF16 weights without grad_fn when their first
    # use is under no_grad. Prefill and training share the caller's AMP scope,
    # so keep prefill out of that weight cache; training still uses caching.
    with torch.no_grad(), torch.autocast(device_type=clean.device.type,
                                        enabled=torch.is_autocast_enabled(clean.device.type),
                                        dtype=torch.get_autocast_dtype(clean.device.type),
                                        cache_enabled=False):
        runner(episode, prefill=True)
    if parallel: parallel.reshard()
    runtime.detach()  # initial observed prefill is a boundary before generated-chunk windows
    for first in range(0, len(episode.ranges), tbptt):
        if offload_audit is not None: offload_audit.clear()
        last = min(first + tbptt, len(episode.ranges))
        sync = parallel.accumulation(last == len(episode.ranges)) if parallel else nullcontext()
        # FSDP swaps full parameter storage between chunk forwards. Cached AMP
        # weight casts share one backward node across those forwards and can
        # defer weight gradients past FSDP's per-forward reduction hooks.
        amp = torch.autocast(device_type=clean.device.type,
                             enabled=torch.is_autocast_enabled(clean.device.type),
                             dtype=torch.get_autocast_dtype(clean.device.type),
                             cache_enabled=False) if parallel is not None and parallel.mode == "fsdp2" else nullcontext()
        # DDP no_sync still encloses both forward and backward. CPU storage
        # preserves the full window's S/optional Persistent-meta graph.
        with sync, amp, activation_storage(activation_offload, device=clean.device, audit=offload_audit,
                                           gpu_budget_gib=activation_gpu_budget_gib):
            loss = runner(episode, first, last, on_prediction=on_prediction)
            _memory_phase(memory_callback, "backward_begin", first=first, last=last)
            from .performance import DEFAULT_EXECUTION, annotation
            execution = getattr(model.ttn_system, "ttn_execution", DEFAULT_EXECUTION)
            with annotation(execution, "Backward"), trace_backward(loss):
                loss.backward()
            _memory_phase(memory_callback, "backward_end", first=first, last=last)
        runtime.detach()
        episode.cache = carry_cache(episode.cache, model.ttn_system.config.camera_attention)
    runtime.verify_sink_reference()
    runtime.verify_replay_reference()
    runtime.verify_memory_prefix()
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
    if updates is not None: result["optimizer_updates"] = updates
    return result
