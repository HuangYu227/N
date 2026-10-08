"""Bounded clean observations: protected prefix, recent queries, participative Top-C.

DeepForcing's summed query/key selection is applied to TLA observations, not to
visual Softmax. Eviction removes fitting constraints, not their legacy inside S.
"""
from dataclasses import dataclass
import hashlib
import math

import torch

from .geometry import apply_complex_rope
from .replay import Observation


@dataclass(frozen=True)
class MemoryCache:
    observation: Observation
    query: torch.Tensor
    hw: tuple[int, int]
    temporal_aligned: bool = False


def _options(capacity_frames, prefix_frames, recent_frames, selection):
    for name, value, minimum in (("capacity_frames", capacity_frames, 1),
                                 ("prefix_frames", prefix_frames, 0),
                                 ("recent_frames", recent_frames, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if capacity_frames < prefix_frames + recent_frames:
        raise ValueError("capacity must cover protected prefix plus recent frames")
    if selection not in ("participative", "fifo"):
        raise ValueError("selection must be participative or fifo")


def _spatial_shape(hw, n):
    if (hw is None or len(hw) not in (2, 3)
            or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in hw)):
        raise ValueError("cache requires an explicit positive HW or THW patch grid")
    h, w = hw[-2:]
    if n % (h*w) or (len(hw) == 3 and math.prod(hw) != n):
        raise ValueError("patch grid and token count disagree")
    return h, w


def _token_frames(frame_ids, b, n, area, device):
    if (frame_ids.ndim != 2 or frame_ids.shape[0] != b
            or frame_ids.dtype not in (torch.int32, torch.int64) or frame_ids.device != device):
        raise ValueError("frame IDs must be integer [batch, frames or tokens] on the feature device")
    if frame_ids.shape[1] == n:
        return frame_ids.long()
    if frame_ids.shape[1]*area == n:
        return frame_ids.repeat_interleave(area, dim=1).long()
    raise ValueError("frame IDs, token count and patch grid disagree")


def _pack(rows, hw, *, aligned=False):
    """Ragged batch padding is inert; selected feature gathers remain live."""
    count = max(row[3].numel() for row in rows)
    fields = []
    for field in range(6):
        dim, fill = (-2 if field in (0, 1, 5) else -1), (-1 if field in (3, 4) else 0)
        tensors = []
        for row in rows:
            x = row[field]
            shape = list(x.shape)
            shape[dim] = count - x.shape[dim]
            tensors.append(torch.cat((x, x.new_full(shape, fill)), dim=dim))
        fields.append(torch.stack(tensors))
    return MemoryCache(Observation(*fields[:5]), fields[5].detach(), hw, aligned)


def _row(cache, batch, index):
    o = cache.observation
    return (o.key[batch].index_select(1, index), o.value[batch].index_select(1, index),
            o.weight[batch].index_select(1, index), o.token_indices[batch].index_select(0, index),
            o.frame_ids[batch].index_select(0, index), cache.query[batch].index_select(1, index))


def _regions(frame_ids, prefix_frames, recent_frames):
    valid = frame_ids >= 0
    prefix = valid & (frame_ids < prefix_frames)
    unique = frame_ids[valid].unique(sorted=True)
    recent = valid & torch.isin(frame_ids, unique[-recent_frames:]) & ~prefix
    return prefix, valid & ~prefix & ~recent, recent


@torch.no_grad()
def validate_cache(cache):
    if not isinstance(cache, MemoryCache):
        raise TypeError("expected MemoryCache")
    o = cache.observation
    if o.key.ndim != 4 or o.value.shape != o.key.shape or cache.query.shape != o.key.shape:
        raise ValueError("cache K/V/Q must have matching [batch, head, token, dim] shapes")
    b, h, n, d = o.key.shape
    _spatial_shape(cache.hw, math.prod(cache.hw))
    if min(b, h, d) < 1 or o.weight.shape != (b, h, n):
        raise ValueError("cache weights must be [batch, head, token]")
    if (o.token_indices.shape != (b, n) or o.frame_ids.shape != (b, n)
            or any(x.dtype not in (torch.int32, torch.int64) for x in (o.token_indices, o.frame_ids))):
        raise ValueError("cache IDs must be integer [batch, token]")
    tensors = (o.key, o.value, o.weight, o.token_indices, o.frame_ids, cache.query)
    if any(x.device != o.key.device for x in tensors) or any(not x.is_floating_point() for x in tensors[:3]):
        raise ValueError("cache tensors must share a device and K/V/W must be floating point")
    if (any(not torch.isfinite(x).all() for x in (o.key, o.value, o.weight, cache.query))
            or (o.weight < 0).any()):
        raise ValueError("cache contains nonfinite features or negative weights")
    area = math.prod(cache.hw)
    if ((o.frame_ids < -1).any() or (o.token_indices < -1).any()
            or not torch.equal(o.frame_ids == -1, o.token_indices == -1)
            or (o.token_indices >= area).any()):
        raise ValueError("cache IDs contain invalid padding or spatial indices")
    pad = o.frame_ids < 0
    if ((o.weight.masked_select(pad[:, None]) != 0).any()
            or (o.key.masked_select(pad[:, None, :, None]) != 0).any()
            or (o.value.masked_select(pad[:, None, :, None]) != 0).any()
            or (cache.query.masked_select(pad[:, None, :, None]) != 0).any()):
        raise ValueError("cache padding must have zero features and weights")
    for frames, spatial in zip(o.frame_ids, o.token_indices):
        ids = frames[frames >= 0]*area + spatial[frames >= 0]
        if ids.numel() > 1 and not (ids[1:] > ids[:-1]).all():
            raise ValueError("cache observations must have unique, sorted frame/spatial identities")
    if cache.query.requires_grad:
        raise ValueError("selection query history must be detached")


def update_cache(previous, k, v, w, q, write, frame_ids, hw, *, capacity_frames=16,
                 prefix_frames=10, recent_frames=4, selection="participative"):
    """Capture a clean transaction candidate. Caller publishes only after commit.

    Previous prefix values are copied by gather, never overwritten. The hard
    selection uses detached recent-query sums; selected K/V/W keep gradients.
    """
    _options(capacity_frames, prefix_frames, recent_frames, selection)
    if k.ndim != 4 or v.shape != k.shape or q.shape != k.shape or w.shape != k.shape[:-1]:
        raise ValueError("clean K/V/Q/W shapes disagree")
    b, heads, n, d = k.shape
    shape = _spatial_shape(hw, n)
    area = math.prod(shape)
    if (write.shape != (b, n) or write.device != k.device
            or any(x.device != k.device for x in (v, w, q))
            or any(not x.is_floating_point() for x in (k, v, w, q))):
        raise ValueError("clean features and write mask must have matching batch/tokens/device")
    if not ((write == 0) | (write == 1)).all():
        raise ValueError("write mask must be boolean or zero/one")
    frames = _token_frames(frame_ids, b, n, area, k.device)
    if (frames[write.bool()] < 0).any():
        raise ValueError("valid writes require nonnegative absolute frame IDs")
    if previous is not None:
        validate_cache(previous)
        if previous.temporal_aligned:
            raise ValueError("a temporally aligned read view cannot be committed")
        if (previous.hw != shape or previous.observation.key.shape[:2] != (b, heads)
                or previous.observation.key.shape[-1] != d or previous.observation.key.device != k.device):
            raise ValueError("previous cache and new clean features disagree")
    spatial = torch.arange(n, device=k.device) % area
    rows = []
    for batch in range(b):
        idx = write[batch].bool().nonzero().flatten()
        row = (k[batch].index_select(1, idx), v[batch].index_select(1, idx),
               w[batch].index_select(1, idx), spatial[idx], frames[batch, idx],
               q[batch].detach().index_select(1, idx))
        if previous is not None:
            prev = _row(previous, batch, (previous.observation.frame_ids[batch] >= 0).nonzero().flatten())
            if row[4].numel() and prev[4].numel() and row[4].min() < prev[4].max():
                raise ValueError("clean cache writes must be causal; older frames cannot be reintroduced")
            row = tuple(torch.cat((a, c), dim=(1 if i in (0, 1, 2, 5) else 0))
                        for i, (a, c) in enumerate(zip(prev, row)))
        ids = row[4]*area + row[3]
        order = ids.argsort(stable=True)
        ids = ids[order]
        if ids.numel() > 1 and (ids[1:] == ids[:-1]).any():
            raise ValueError("duplicate clean frame/spatial write; committed overlap must be masked")
        row = tuple(x.index_select(1 if i in (0, 1, 2, 5) else 0, order) for i, x in enumerate(row))
        if ids.numel() > capacity_frames*area:
            prefix, middle, recent = _regions(row[4], prefix_frames, recent_frames)
            forced = prefix | recent
            slots = capacity_frames*area - int(forced.sum())
            candidates = middle.nonzero().flatten()
            if slots < 0:
                raise ValueError("protected prefix and recent observations exceed cache capacity")
            with torch.no_grad():
                if selection == "participative":
                    recent_ids = row[4].unique(sorted=True)[-recent_frames:]
                    query_sum = row[5][:, torch.isin(row[4], recent_ids)].float().sum(1)
                    scores = torch.einsum("hd,hnd->n", query_sum, row[0][:, candidates].detach().float())
                    scores = scores/(math.sqrt(d)*heads)
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError("nonfinite participative selection score")
                    chosen = candidates[scores.argsort(descending=True, stable=True)[:slots]]
                else:
                    chosen = candidates[-slots:] if slots else candidates[:0]
                keep = forced.nonzero().flatten()
                keep = torch.cat((keep, chosen)).sort().values
            row = tuple(x.index_select(1 if i in (0, 1, 2, 5) else 0, keep) for i, x in enumerate(row))
        rows.append(row)
    result = _pack(rows, shape)
    validate_cache(result)
    return result


def sources(cache, *, prefix_frames=10, recent_frames=4):
    """Return disjoint (prefix, selected middle, recent) fitting observations."""
    _options(prefix_frames + recent_frames, prefix_frames, recent_frames, "participative")
    rows = [[], [], []]
    for b, frames in enumerate(cache.observation.frame_ids):
        for group, mask in enumerate(_regions(frames, prefix_frames, recent_frames)):
            rows[group].append(_row(cache, b, mask.nonzero().flatten()))
    return tuple(_pack(group, cache.hw, aligned=cache.temporal_aligned).observation for group in rows)


def detach_cache(cache):
    if cache is None:
        return None
    o = cache.observation
    return MemoryCache(Observation(*(x.detach() for x in
        (o.key, o.value, o.weight, o.token_indices, o.frame_ids))), cache.query.detach(), cache.hw,
        cache.temporal_aligned)


def storage_bytes(cache):
    if cache is None:
        return 0
    o = cache.observation
    return sum(x.numel()*x.element_size() for x in
               (o.key, o.value, o.weight, o.token_indices, o.frame_ids, cache.query))


def prefix_digest(cache, prefix_frames=10):
    """Hash original payload, independent of ragged padding or non-prefix entries."""
    if cache.temporal_aligned:
        raise ValueError("audit the original prefix, not its temporary aligned read view")
    digest = hashlib.sha256()
    for b, frames in enumerate(cache.observation.frame_ids):
        idx = ((frames >= 0) & (frames < prefix_frames)).nonzero().flatten()
        for tensor in _row(cache, b, idx):
            digest.update(str((b, tensor.shape, tensor.dtype)).encode("ascii"))
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def temporal_realign(cache, rope, current_frame_ids, *, prefix_frames=10, recent_frames=4):
    """Temporary key-only view: prefix -> packed Top-C -> recent absolute frames.

    This adapts DeepForcing's relocation principle, not its hardcoded first-roll
    offsets. Top-C tokens pack in stable absolute-frame/spatial order. Original
    IDs and V stay unchanged; this view must never be committed or realigned.
    """
    validate_cache(cache)
    if cache.temporal_aligned:
        raise ValueError("temporal cache alignment must start from original absolute keys")
    if (rope is None or not any(cls.__name__ == "CausalWanRotaryPosEmbed" for cls in type(rope).__mro__)
            or not isinstance(getattr(rope, "freqs", None), torch.Tensor)):
        raise ValueError("cache alignment requires actual CausalWanRoPE layout")
    o, table = cache.observation, rope.freqs
    dim, area = o.key.shape[-1], math.prod(cache.hw)
    if (dim != rope.attention_head_dim or dim % 2 or table.ndim != 2
            or table.shape[1] != dim//2 or not table.is_complex()):
        raise ValueError("cache and causal RoPE dimensions disagree")
    if (current_frame_ids.ndim != 2 or current_frame_ids.shape[0] != o.key.shape[0]
            or current_frame_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("current frame IDs must be integer [batch, frames]")
    targets = o.frame_ids.clone()
    for b, frames in enumerate(o.frame_ids):
        prefix, middle, recent = _regions(frames, prefix_frames, recent_frames)
        future = current_frame_ids[b][current_frame_ids[b] >= 0]
        if not future.numel():
            raise ValueError("temporal alignment needs at least one valid current frame per batch")
        tail_start = int(frames[recent].min()) if recent.any() else int(future.min())
        middle_idx, prefix_ids = middle.nonzero().flatten(), frames[prefix].unique(sorted=True)
        middle_frames = (middle_idx.numel() + area - 1)//area
        prefix_start = tail_start - middle_frames - prefix_ids.numel()
        if prefix_start < 0:
            # Initial overlapping prefill has no earlier nonnegative location.
            continue
        if middle_idx.numel():
            targets[b, middle_idx] = tail_start - middle_frames + torch.arange(
                middle_idx.numel(), device=frames.device)//area
        for i, frame in enumerate(prefix_ids):
            targets[b, frames == frame] = prefix_start + i
    valid = o.frame_ids >= 0
    if (valid.any() and ((o.frame_ids[valid] >= table.shape[0]).any()
                       or (targets[valid] >= table.shape[0]).any())):
        raise ValueError("cache time coordinates exceed the causal RoPE table")
    table = table.to(o.key.device)
    time_channels = dim//2 - 2*(dim//6)
    phase = table.new_ones((*o.frame_ids.shape, dim//2))
    phase[..., :time_channels] = (table[targets.clamp_min(0), :time_channels]
                                  / table[o.frame_ids.clamp_min(0), :time_channels])
    if not torch.isfinite(phase).all():
        raise ValueError("nonfinite cache temporal phase")
    key = apply_complex_rope(o.key, phase[:, None])
    return MemoryCache(Observation(key, o.value, o.weight, o.token_indices, o.frame_ids),
                       cache.query, cache.hw, True)
