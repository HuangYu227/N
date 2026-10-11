"""Frame-wise Direct-S regression; spatial tokens form one joint observation."""
from dataclasses import replace
import torch

from . import memory_cache
from . import proximal
from .proximal import proximal_correct


def _prefix_statistics(retained, observation, config, ids, dtype):
    """Reuse only after no later frame can add to the protected prefix."""
    if not config.memory_prefix_frames or retained.temporal_aligned or not (ids >= config.memory_prefix_frames).all():
        return None
    cached = retained.prefix_statistics
    if cached is not None and cached[0] == config.memory_prefix_frames and cached[1][4].dtype == dtype:
        return cached[1]
    with torch.autocast(device_type=observation.key.device.type, enabled=False):
        return proximal._source(*(t.to(dtype) for t in (observation.key, observation.value, observation.weight)))


def _active_statistics(source, active):
    if active.all():
        return source
    masks = (active[:, None, None, None], active[:, None, None, None], active[:, None, None],
             active[:, None], active[:, None, None, None], active[:, None, None, None])
    return tuple(torch.where(mask, tensor, torch.zeros_like(tensor)) for tensor, mask in zip(source, masks))


def frame_correct(prior, q, k, v, beta, write, frame_ids, hw, config, *, retained=None,
                  prefill=False, collect_stats=False):
    frames, height, width = hw
    spatial = height * width
    if frames != frame_ids.shape[-1] or q.shape != k.shape or q.shape[-2] != frames * spatial:
        raise ValueError("frame memory requires a matching T/H/W token layout")
    if frame_ids.shape[0] != q.shape[0] or write.shape != (q.shape[0], frames * spatial):
        raise ValueError("frame IDs/write mask must match the token batch")
    if frames > 1 and not (frame_ids[:, 1:] > frame_ids[:, :-1]).all():
        raise ValueError("frame IDs must increase strictly")
    state, reads, weights, records = prior, [], [], []
    for f in range(frames):
        sl = slice(f * spatial, (f + 1) * spatial)
        ids = frame_ids[:, f:f+1]
        active = ids[:, 0] >= config.memory_start_frame
        histories = memory_cache.sources(retained, prefix_frames=config.memory_prefix_frames,
                                        recent_frames=config.memory_recent_frames) if retained is not None else ()
        reused = (retained is not None and retained.prefix_statistics is not None
                  and retained.prefix_statistics[0] == config.memory_prefix_frames)
        summary = (_prefix_statistics(retained, histories[0], config, ids,
                   torch.float64 if prior.dtype == torch.float64 else torch.float32) if histories else None)
        if summary is not None:
            retained = replace(retained, prefix_statistics=(config.memory_prefix_frames, summary))
        statistics = (_active_statistics(summary, active), None, None) if summary is not None else ()
        # A batch can cross the history threshold at different absolute positions.
        histories = tuple(replace(item, weight=item.weight * active[:, None, None]) for item in histories)
        incoming = state
        candidate, weight, stats = proximal_correct(state, k[:, :, sl], v[:, :, sl], beta[:, :, sl],
            write[:, sl], history=histories, history_weight=config.memory_history_weight,
            kappa=config.memory_kappa if prefill or bool((ids == 0).all()) else config.memory_frame_kappa,
            eps=config.eps, collect_stats=collect_stats, _history_statistics=statistics)
        valid = write[:, sl].any(-1)
        state = torch.where(valid[:, None, None, None], candidate, incoming)
        reads.append(q[:, :, sl] @ state)
        weights.append(weight)
        if config.memory_selection != "none" and valid.any():
            retained = memory_cache.update_cache(retained, k[:, :, sl], v[:, :, sl], weight, q[:, :, sl],
                write[:, sl], ids, (1, height, width), capacity_frames=config.memory_capacity_frames,
                prefix_frames=config.memory_prefix_frames, recent_frames=config.memory_recent_frames,
                selection=config.memory_selection)
        if collect_stats:
            stats.update(frame_ids=ids.detach().cpu().tolist(), history_active=active.detach().cpu().tolist(),
                         written=valid.detach().cpu().tolist(), prefix_statistics_reused=bool(reused and summary is not None))
            records.append(stats)
    return state, torch.cat(reads, dim=-2), torch.cat(weights, dim=-1), retained, records
