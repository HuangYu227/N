"""Original chunk mathematics, topologically scanned per layer, one backward."""
from types import SimpleNamespace
from dataclasses import replace
import gc
import torch
from torch import nn

from .session import TTNSession, repeat_batch, record_noise
from .runtime import TTNChunkContext
from .core import ANCHORS
from .training import chunk_ranges, activation_storage, _memory_phase


def _ffn_record_rows(candidates, chunk_count):
    """Move detached per-layer diagnostics once, outside all replay regions."""
    rows = [[] for _ in range(chunk_count)]
    for index, (_, records) in sorted(candidates.items()):
        if len(records) != chunk_count + 1 or any(record is None for record in records[1:]):
            raise RuntimeError('missing full-sequence FFN diagnostics')
        keys = [key for key, value in records[1].items() if isinstance(value, torch.Tensor)]
        values = (torch.stack([torch.stack([record[key] for key in keys], -1)
                               for record in records[1:]]).detach().cpu().tolist() if keys else None)
        for chunk, record in enumerate(records[1:]):
            converted = {key: value for key, value in record.items() if key not in keys}
            if keys:
                converted.update({key: [sample[column] for sample in values[chunk]]
                                  for column, key in enumerate(keys)})
            rows[chunk].append(dict(block=index, **converted))
    return rows


class LegacySequenceSession(TTNSession):
    def forward(self, x, t, y, context, cache, start, end, mask=None, data_info=None, **kwargs):
        batch, _, frames, _, _ = x.shape
        measured = replace(context, sequence_mode=False, collect_local_stats=True, local_trajectory=[])
        record_noise(measured, t, kwargs.pop('noise_sigma', None), kwargs.pop('sampled_timestep', None))
        context.noise_info = measured.noise_info
        x = torch.cat((self.clean[:, :, :1], x[:, :, 1:]), dim=2)
        t = t.clone(); t[:, :, 0] = 0
        observed = torch.cat((self.clean[:, :, :1], torch.zeros_like(self.clean[:, :, 1:])), dim=2)
        x = torch.cat((observed, x, self.clean), 0)
        t = torch.cat((torch.zeros_like(t), t, torch.zeros_like(t)), 0)
        y = repeat_batch(y, 3*batch)
        if mask is not None: mask = repeat_batch(mask, 3*batch)
        kwargs['save'] = True
        output, caches = super().forward(x, t, y, context, cache, start, end, mask, data_info, **kwargs)
        return output[batch:2*batch], caches


class LegacySequenceTraining(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__(); self.model, self.loss_fn = model, loss_fn

    def forward(self, episode):
        system, clean = self.model.ttn_system, episode.clean
        b, _, frames, _, _ = clean.shape
        camera = episode.session.camera
        coefficients = []
        for index, (start, end) in enumerate(chunk_ranges(frames)):
            prior = camera[:, 0 if index == 0 else start-1, :16].reshape(b, 4, 4)
            mask = torch.ones(b, end-start, dtype=torch.bool, device=clean.device)
            if start == 0: mask[:, 0] = False
            coefficients.append(system.controller(camera[:, start:end, :16].reshape(b, end-start, 4, 4),
                camera[:, start:end, 16:], prior, mask, episode.session.width, episode.session.height))
        episode.context.sequence_coefficients = tuple(coefficients)
        def measured_prediction(output):
            with torch.no_grad():
                velocity_error = (output.float()-(episode.noise-clean).float()).square().mean((1,3,4))
                sigma = episode.context.noise_info.get('noise_sigma')
                sigma = torch.as_tensor(sigma,device=clean.device) if sigma is not None else episode.timesteps.float()/1000
                estimate = clean.detach().float()+sigma[...,None,None]*((episode.noise-clean).detach().float()-output.detach().float())
                estimate[:,:,0] = clean[:,:,0]
                dc = estimate.mean((-1,-2),keepdim=True)
                episode.measurements = {
                    'per_frame_velocity_mse': velocity_error.cpu().tolist(),
                    'estimated_x0_rms': estimate.square().mean((1,3,4)).sqrt().cpu().tolist(),
                    'estimated_x0_dc_rms': dc.square().mean((1,3,4)).sqrt().cpu().tolist(),
                    'estimated_x0_detail_rms': (estimate-dc).square().mean((1,3,4)).sqrt().cpu().tolist(),
                    'estimated_x0_adjacent_rms': (estimate[:,:,1:]-estimate[:,:,:-1]).square().mean((1,3,4)).sqrt().cpu().tolist(),
                    'measurement': 'detached one-step x0 estimate under sampled training noise and GT-clean history; not autoregressive video quality'}
            if episode.on_prediction:
                for index,(start,end) in enumerate(chunk_ranges(frames)):
                    episode.on_prediction(index,output[:,:,start:end])
        return self.loss_fn(episode.session, clean, episode.timesteps, episode.noise, episode.y,
            episode.context, [[None]*10 for _ in self.model.blocks], 0, frames, episode.mask,
            episode.data_info, episode.loss_mask, measured_prediction).mean()


def train_legacy_sequence(model, clean, y, camera_conditions, optimizer, loss_fn, timesteps, noise, *,
        width, height, mask=None, data_info=None, extras=None, valid_mask=None, tbptt=0,
        chunk_size=3, outer_clip=.5, on_prediction=None, parallel=None, activation_offload='none',
        memory_callback=None, audit_update=False, cached_chunks=2, window_model=None):
    if tbptt != 0 or chunk_size != 3 or cached_chunks != 2:
        raise ValueError('legacy full sequence requires tbptt0, chunk3, two camera groups')
    b, _, frames, _, _ = clean.shape
    if frames < 4 or frames%3 != 1 or noise.shape != clean.shape or timesteps.shape != (b, 1, frames):
        raise ValueError('complete 1+3n sequence and matching noise/time required')
    valid = torch.ones(b, frames, dtype=torch.bool, device=clean.device) if valid_mask is None else valid_mask.bool()
    if valid.shape != (b, frames) or not valid.all(): raise ValueError('unpadded complete clips required')
    if set(extras or {})-{'chunk_plucker'} or data_info and any(n in data_info for n in ('image_embeds','image_vae_embeds')):
        raise ValueError('unsupported full-sequence image/extras conditioning')
    session = LegacySequenceSession(model, camera_conditions, width, height, valid, extras, live_cache=True)
    runtime = session.reset(b)
    session.clean = clean
    state = runtime.world_state.repeat(3, 1, 1, 1, 1)
    psi = runtime.transition_fast.repeat(3, 1, 1, 1)
    ids = torch.arange(frames, device=clean.device).expand(3*b, -1)
    context = TTNChunkContext(state, state, psi, psi, model.ttn_system, ids,
        camera_conditions[:, :, :16].reshape(b, frames, 4, 4).repeat(3, 1, 1, 1),
        valid.repeat(3, 1), valid.repeat(3, 1), 0, id(runtime), sequence_mode=True,
        camera_cached_chunks=cached_chunks, checkpoint_cpu_inputs=clean.is_cuda and activation_offload != 'none',
        collect_local_stats=True, sequence_memory_callback=memory_callback,
        ffn_fast=runtime.ffn_fast, ffn_delta_clip=None)
    lm = valid[:, None, :, None, None].float();lm[:, :, 0] = 0
    episode = SimpleNamespace(session=session, context=context, clean=clean, timesteps=timesteps, noise=noise,
        y=y, mask=mask, data_info=data_info, loss_mask=lm, on_prediction=on_prediction)
    runner = parallel.window if parallel else window_model or LegacySequenceTraining(model, loss_fn)
    if parallel: parallel.validate_schedule(frames, 0, len(chunk_ranges(frames)), b)
    from .training_health import local_eta_snapshot, local_eta_update
    eta_before = local_eta_snapshot(model)
    model.eval();optimizer.zero_grad(set_to_none=True)
    amp = torch.autocast(device_type=clean.device.type, enabled=torch.is_autocast_enabled(clean.device.type),
        dtype=torch.get_autocast_dtype(clean.device.type), cache_enabled=False)
    storage = activation_storage(activation_offload)
    with amp, storage:
        _memory_phase(memory_callback, 'sequence_begin', frames=frames, tbptt=0)
        loss = runner(episode)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite legacy full-sequence loss')
        if set(context.candidates) != set(range(5)): raise RuntimeError('missing sequence anchors')
        if not all(torch.isfinite(value).all() for row in context.candidates.values() for value in row[:2]):
            raise FloatingPointError('nonfinite full-sequence S/psi; optimizer not updated')
        inplace = model.ttn_system.config.ttt_backend == 'inplace-ffn'
        if inplace:
            if set(context.ffn_candidates) != set(ANCHORS):
                raise RuntimeError('missing full-sequence FFN candidates')
            if any(value.shape != runtime.ffn_fast[index].shape
                   for index, (value, _) in context.ffn_candidates.items()):
                raise ValueError('full-sequence FFN candidate dimensions mismatch; optimizer not updated')
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                   for value, _ in context.ffn_candidates.values()):
                raise FloatingPointError('nonfinite or non-FP32 full-sequence FFN state; optimizer not updated')
            ffn_rows = _ffn_record_rows(context.ffn_candidates, len(chunk_ranges(frames)))
        _memory_phase(memory_callback, 'backward_begin')
        loss.backward()
        _memory_phase(memory_callback, 'backward_end')
    norm = parallel.clip_grad_norm(outer_clip) if parallel else torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], outer_clip, error_if_nonfinite=True)
    _memory_phase(memory_callback, 'optimizer_begin')
    probe = None
    if audit_update:
        from .training_health import FirstUpdateProbe
        probe = FirstUpdateProbe(model)
    optimizer.step()
    _memory_phase(memory_callback, 'optimizer_end')
    if set(context.candidates) != set(range(5)): raise RuntimeError('missing sequence anchors')
    runtime.world_state = torch.stack([context.candidates[i][0].detach() for i in range(5)], 1)
    runtime.transition_fast = torch.stack([context.candidates[i][1].detach() for i in range(5)], 1)
    if inplace:
        runtime.ffn_fast = {index: value.detach() for index, (value, _) in context.ffn_candidates.items()}
        runtime.ffn_commit_count = len(chunk_ranges(frames))
    runtime.commit_count = runtime.predict_count = len(chunk_ranges(frames))+1
    runtime.previous_committed_pose = camera_conditions[:, -1, :16].reshape(b,4,4)
    runtime.committed_frame_ids = [set(range(frames)) for _ in range(b)]
    runtime.revision, runtime.prefilled = runtime.commit_count, True
    rows = [context.candidates[i][2] for i in range(5)]
    chunks = [dict(chunk=j, start=start, end=end, anchors=[rows[i][j+1] for i in range(5)])
              for j,(start,end) in enumerate(chunk_ranges(frames))]
    if inplace:
        for chunk, ffn in zip(chunks, ffn_rows): chunk['ffn_ttt'] = ffn
    result = dict(loss=float(loss.detach()), outer_grad_norm=float(norm), runtime=runtime, chunks=chunks,
        activation_storage=dict(getattr(storage,'storage_stats',{})),
        prediction_diagnostics=episode.measurements,
        prefill=dict(chunk=-1,start=0,end=1,anchors=[rows[i][0] for i in range(5)],live_gradient=True),
        exposure=dict(clips=b,valid_latent_frames=b*frames,predicted_latent_frames=b*(frames-1),
                      predicted_chunks=b*len(chunks),model_forwards=1),
        sequence=dict(protocol=model.ttn_training_protocol,backward_calls=1,temporal_detaches=0,
                      cached_chunks=2,live_prefill=True))
    if inplace:
        result['sequence'].update(ffn_clean_commits=len(chunks), ffn_prefill_writes=0,
                                  psi_updates=False, ffn_delta_clip=None)
    if probe is not None: result['optimizer_updates'] = probe.report(optimizer)
    if eta_before is not None: result['local_eta_update'] = local_eta_update(model, eta_before, optimizer)
    # Episode credit has finished; release closure cycles, not temporal gradients.
    before = torch.cuda.memory_allocated(clean.device) if clean.is_cuda else 0
    context.candidates.clear()
    context.ffn_candidates.clear()
    context.ffn_fast.clear()
    context.sequence_coefficients = ()
    context.sequence_memory_callback = None
    session.clean = None
    del loss, episode, context, session, rows
    collected = gc.collect()
    after = torch.cuda.memory_allocated(clean.device) if clean.is_cuda else 0
    result['lifecycle'] = dict(collected=collected, gpu_bytes_before=before, gpu_bytes_after=after,
                               reclaimed_gpu_bytes=before-after, boundary='after episode backward and optimizer')
    _memory_phase(memory_callback, 'episode_released', **result['lifecycle'])
    return result
