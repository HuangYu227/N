"""Causal clean-history training, one optimizer step per clip, explicit TBPTT windows."""
from contextlib import nullcontext
from dataclasses import dataclass, field
import torch
from .session import TTNSession, carry_cache
from .cuda_debug import trace_backward


def activation_storage(mode):
    """Copy saved tensors to host before FSDP resharding; never replay TTN forwards.

    Native save_on_cpu copies values (including weight views) during forward,
    so backward does not retain/read freed FSDP all-gather storage. FSDP's own
    parameter hooks still unshard and reduce the actual parameter gradients.
    """
    if mode == "none": return nullcontext()
    if mode == "cpu": return torch.autograd.graph.save_on_cpu(pin_memory=True)
    raise ValueError("activation_offload must be none or cpu")


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

        def model_call(x, timestep, **unused):
            result, _ = session.forward(x, timestep, y, context, cache, start, end, mask, data_info)
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
            _memory_phase(episode.memory_callback, "prefill_end")
            return episode.clean.new_zeros(())
        losses_in_window = []
        for index in range(first, last):
            start, end = episode.ranges[index]
            context = session.begin_chunk(start, end)
            lm = episode.loss_valid[:, None, start:end, None, None].to(torch.float32)
            callback = (lambda output, i=index: on_prediction(i, output)) if on_prediction else None
            _memory_phase(episode.memory_callback, "noisy_begin", chunk=index, start=start, end=end)
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
            _, episode.cache = session.clean_forward(episode.clean[:, :, start:end], episode.y, context,
                                                      episode.cache, start, end, episode.mask, episode.data_info)
            episode.records.append(dict(session.runtime.last_stats))
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
               memory_callback=None):
    if tbptt not in (1, 2, 4): raise ValueError("reference TBPTT supports K=1,2,4")
    if activation_offload not in ("none", "cpu"): raise ValueError("activation_offload must be none or cpu")
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
    session = TTNSession(model, camera_conditions, width, height, valid, extras)
    runtime = session.reset(b)
    optimizer.zero_grad(set_to_none=True)
    model.eval()  # disables stochastic conditioning; does NOT disable differentiation
    episode = ClipTrainingContext(session, clean, y, timesteps, noise, loss_valid, total,
                                  chunk_ranges(frames, chunk_size), [[None] * 10 for _ in model.blocks], mask, data_info,
                                  memory_callback=memory_callback)
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
        # preserves the full window's S graph and detached inner updates.
        with sync, amp, activation_storage(activation_offload):
            loss = runner(episode, first, last, on_prediction=on_prediction)
            _memory_phase(memory_callback, "backward_begin", first=first, last=last)
            with trace_backward(loss):
                loss.backward()
            _memory_phase(memory_callback, "backward_end", first=first, last=last)
        runtime.detach()
        episode.cache = carry_cache(episode.cache)
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = parallel.clip_grad_norm(outer_clip) if parallel else torch.nn.utils.clip_grad_norm_(
        params, outer_clip, error_if_nonfinite=True)
    _memory_phase(memory_callback, "optimizer_begin")
    optimizer.step()  # slow parameters stay fixed throughout every clip's windows
    _memory_phase(memory_callback, "optimizer_end")
    return {"loss": episode.total_loss, "outer_grad_norm": float(grad_norm), "runtime": runtime,
            "chunks": episode.records}
