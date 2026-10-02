"""Causal clean-history training, one optimizer step per clip, explicit TBPTT windows."""
from contextlib import nullcontext
from dataclasses import dataclass, field
import torch
from .session import TTNSession, carry_cache


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
            session.prefill(episode.clean[:, :, :1], episode.y, episode.mask, episode.data_info)
            return episode.clean.new_zeros(())
        losses_in_window = []
        for index in range(first, last):
            start, end = episode.ranges[index]
            context = session.begin_chunk(start, end)
            lm = episode.loss_valid[:, None, start:end, None, None].to(torch.float32)
            callback = (lambda output, i=index: on_prediction(i, output)) if on_prediction else None
            losses = self.loss_fn(session, episode.clean[:, :, start:end], episode.timesteps[:, :, start:end],
                                  episode.noise[:, :, start:end], episode.y, context, episode.cache, start, end,
                                  episode.mask, episode.data_info, lm, callback)
            loss = (losses * (episode.loss_valid[:, start:end].sum(-1) / episode.total)).mean()
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite flow loss")
            losses_in_window.append(loss)
            episode.total_loss += float(loss.detach())
            # The current GT clean chunk becomes history only AFTER its noisy prediction.
            _, episode.cache = session.clean_forward(episode.clean[:, :, start:end], episode.y, context,
                                                      episode.cache, start, end, episode.mask, episode.data_info)
            episode.records.append(dict(session.runtime.last_stats))
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
               parallel=None):
    if tbptt not in (1, 2, 4): raise ValueError("reference TBPTT supports K=1,2,4")
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
                                  chunk_ranges(frames, chunk_size), [[None] * 10 for _ in model.blocks], mask, data_info)
    runner = parallel.window if parallel else window_model
    if runner is None: runner = TTNTrainingWindow(model, loss_fn)
    if parallel: parallel.validate_schedule(frames, tbptt, len(episode.ranges), b)
    with torch.no_grad():
        runner(episode, prefill=True)
    if parallel: parallel.reshard()
    runtime.detach()  # initial observed prefill is a boundary before generated-chunk windows
    for first in range(0, len(episode.ranges), tbptt):
        last = min(first + tbptt, len(episode.ranges))
        sync = parallel.accumulation(last == len(episode.ranges)) if parallel else nullcontext()
        with sync:  # DDP no_sync must enclose BOTH forward and backward.
            loss = runner(episode, first, last, on_prediction=on_prediction)
            loss.backward()
        runtime.detach()
        episode.cache = carry_cache(episode.cache)
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = parallel.clip_grad_norm(outer_clip) if parallel else torch.nn.utils.clip_grad_norm_(
        params, outer_clip, error_if_nonfinite=True)
    optimizer.step()  # slow parameters stay fixed throughout every clip's windows
    return {"loss": episode.total_loss, "outer_grad_norm": float(grad_norm), "runtime": runtime,
            "chunks": episode.records}
