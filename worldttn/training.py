"""Causal clean-history training, one optimizer step per clip, explicit TBPTT windows."""
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
               on_prediction=None):
    if tbptt not in (1, 2, 4): raise ValueError("reference TBPTT supports K=1,2,4")
    b, _, frames, _, _ = clean.shape
    if timesteps.ndim == 1: timesteps = timesteps[:, None, None].expand(b, 1, frames).clone()
    if timesteps.shape != (b, 1, frames) or noise.shape != clean.shape:
        raise ValueError("noise/timestep dimensions mismatch")
    timesteps = timesteps.clone()
    timesteps[:, :, 0] = 0
    valid = torch.ones(b, frames, dtype=torch.bool, device=clean.device) if valid_mask is None else valid_mask.bool()
    loss_valid = valid.clone()
    loss_valid[:, 0] = False
    total = loss_valid.sum(-1).clamp_min(1)
    session = TTNSession(model, camera_conditions, width, height, valid, extras)
    runtime = session.reset(b)
    optimizer.zero_grad(set_to_none=True)
    model.eval()  # disables stochastic conditioning; does NOT disable differentiation
    cache = [[None] * 10 for _ in model.blocks]
    session.prefill(clean[:, :, :1], y, mask, data_info)
    runtime.detach()  # initial observed prefill is a boundary before generated-chunk windows
    windows = []
    records = []
    total_loss = 0.
    ranges = chunk_ranges(frames, chunk_size)
    for index, (start, end) in enumerate(ranges):
        context = session.begin_chunk(start, end)
        lm = loss_valid[:, None, start:end, None, None].to(torch.float32)
        prediction_callback = (lambda output, i=index: on_prediction(i, output)) if on_prediction else None
        losses = loss_fn(session, clean[:, :, start:end], timesteps[:, :, start:end], noise[:, :, start:end], y,
                         context, cache, start, end, mask, data_info, lm, prediction_callback)
        loss = (losses * (loss_valid[:, start:end].sum(-1) / total)).mean()
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite flow loss")
        windows.append(loss)
        total_loss += float(loss.detach())
        # Only AFTER predicting this chunk may its GT clean history become visible.
        _, cache = session.clean_forward(clean[:, :, start:end], y, context, cache, start, end, mask, data_info)
        records.append(dict(runtime.last_stats))
        if (index + 1) % tbptt == 0 or index + 1 == len(ranges):
            torch.stack(windows).sum().backward()
            windows.clear()
            runtime.detach()
            cache = carry_cache(cache)
    params = [p for p in model.parameters() if p.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(params, outer_clip, error_if_nonfinite=True)
    optimizer.step()  # slow parameters stay fixed throughout every clip's windows
    return {"loss": total_loss, "outer_grad_norm": float(grad_norm), "runtime": runtime, "chunks": records}
