"""Inference-only event replay in TTN state coordinates; no model parameters.

Events own detached FP32 CPU statistics of post-RoPE keys.  A read changes only
the caller's temporary state; archive publication and chunk selection belong to
the runtime transaction.  Absolute RoPE phases are preserved, never reapplied.
"""
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class EventMemoryOptions:
    mode: str = "off"
    anchors: tuple[int, ...] = (11, 15, 19)
    capacity: int = 8
    topk: int = 2
    temperature: float = .1
    gain: float = .1
    max_relative_delta: float = .05
    selector: str = "content"
    retention: str = "coverage"
    start_chunk: int = 5
    max_read_sigma: float = .5

    def __post_init__(self):
        if self.mode not in ("off", "state-replay"):
            raise ValueError("event memory mode must be off or state-replay")
        anchors = tuple(self.anchors)
        if (not anchors or any(type(i) is not int or i not in (3, 7, 11, 15, 19) for i in anchors)
                or anchors != tuple(sorted(set(anchors)))):
            raise ValueError("event memory anchors must be sorted unique TTN block indices")
        object.__setattr__(self, "anchors", anchors)
        for name in ("capacity", "topk", "start_chunk"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"event memory {name} must be a positive integer")
        if self.topk > self.capacity:
            raise ValueError("event memory topk cannot exceed capacity")
        for name in ("temperature", "gain", "max_relative_delta", "max_read_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"event memory {name} must be finite")
        if self.temperature <= 0 or not 0 <= self.gain <= 1 or not 0 <= self.max_relative_delta <= 1:
            raise ValueError("event memory temperature/gain/state cap outside their valid range")
        if not 0 <= self.max_read_sigma <= 1:
            raise ValueError("event memory max_read_sigma must be in [0, 1]")
        if self.selector not in ("content", "recent", "shuffled", "initial"):
            raise ValueError("unknown event memory selector")
        if self.retention not in ("coverage", "fifo"):
            raise ValueError("unknown event memory retention")

    @property
    def active(self):
        return self.mode == "state-replay" and self.gain > 0


def _inference(options):
    if options.active and torch.is_grad_enabled():
        raise ValueError("event memory is inference-only; disable gradients")


def _finite_float(value, shape, name, *, dtype=torch.float32):
    if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype
            or not torch.isfinite(value).all()):
        raise ValueError(f"event memory {name} has invalid shape/dtype/finite values")


def _masked_features(k, v, write):
    if (not isinstance(k, torch.Tensor) or k.ndim != 4 or min(k.shape) < 1
            or not isinstance(v, torch.Tensor) or v.shape != k.shape
            or not k.is_floating_point() or not v.is_floating_point()
            or v.device != k.device or not isinstance(write, torch.Tensor)
            or write.shape != (k.shape[0], k.shape[2]) or write.dtype != torch.bool
            or write.device != k.device or not write.any(-1).all()):
        raise ValueError("event memory requires matching K/V and nonempty boolean write masks per batch")
    selected = write[:, None, :, None]
    # Mask before arithmetic: zero times a padded NaN still produces NaN.
    k, v = (torch.where(selected, item.detach().float(), 0.) for item in (k, v))
    if not torch.isfinite(k).all() or not torch.isfinite(v).all():
        raise ValueError("event memory active K/V must be finite")
    return k, v


def pooled_descriptor(k0, v, write, grid=None):
    """Normalized K0/V mean/std, [B,4*H*D], or 4 spatial cells if grid.

    K0 must come from before RoPE.  This removes positional phases, not the
    noise/timestep dependence of the model features; the caller gates sigma.
    """
    k0, v = _masked_features(k0, v, write)
    cells = [(k0, v, write)]
    if grid is not None:
        if (not isinstance(grid, (tuple, list)) or len(grid) != 3
                or any(type(size) is not int or size < 1 for size in grid)
                or grid[1] < 2 or grid[2] < 2 or math.prod(grid) != k0.shape[2]):
            raise ValueError("event descriptor grid must match tokens and have at least 2x2 spatial cells")
        t, height, width = grid
        shape = (*k0.shape[:2], t, height, width, k0.shape[-1])
        keys, values, masks = k0.reshape(shape), v.reshape(shape), write.reshape(-1, t, height, width)
        cells = []
        for top, bottom in ((0, height // 2), (height // 2, height)):
            for left, right in ((0, width // 2), (width // 2, width)):
                cells.append((keys[..., top:bottom, left:right, :].flatten(2, 4),
                              values[..., top:bottom, left:right, :].flatten(2, 4),
                              masks[..., top:bottom, left:right].flatten(1)))
    parts = []
    for keys, values, selected in cells:
        count = selected.sum(-1).float().clamp_min(1)[:, None, None]
        for value in (keys, values):
            mean = value.sum(-2) / count
            variance = ((value - mean[..., None, :]).square()
                        * selected[:, None, :, None]).sum(-2) / count
            parts.extend((mean.flatten(1), variance.clamp_min(0).sqrt().flatten(1)))
    descriptor = F.normalize(torch.cat(parts, -1), dim=-1, eps=1e-12)
    if not torch.isfinite(descriptor).all():
        raise ValueError("nonfinite event descriptor")
    return descriptor


def sufficient_statistics(k, v, w, write, eps):
    """Exact Correct statistics for already normalized W and post-RoPE K."""
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("event statistics eps must be finite and positive")
    k, v = _masked_features(k, v, write)
    if (not isinstance(w, torch.Tensor) or w.shape != k.shape[:-1]
            or not w.is_floating_point() or w.device != k.device):
        raise ValueError("event statistics W shape/device/dtype mismatch")
    weights = torch.where(write[:, None], w.detach().float(), 0.)
    if not torch.isfinite(weights).all() or (weights < 0).any() or not (weights.sum(-1) > 0).all():
        raise ValueError("event statistics require finite nonnegative W and positive mass per head")
    gram = k.transpose(-1, -2) @ (weights[..., None] * k)
    rhs = k.transpose(-1, -2) @ (weights[..., None] * v)
    denom = eps + (weights * k.square().sum(-1)).sum(-1)
    if not all(torch.isfinite(item).all() for item in (gram, rhs, denom)):
        raise ValueError("nonfinite event sufficient statistics")
    return gram, rhs, denom


@dataclass(frozen=True)
class EventPayload:
    descriptor: torch.Tensor  # [F], normalized; detached owned CPU storage
    gram: torch.Tensor        # [H,D,D], post-RoPE K^T W K
    rhs: torch.Tensor         # [H,D,D], K^T W V in the current Cayley coordinate
    denom: torch.Tensor       # [H], eps + trace(gram)
    frame_ids: torch.Tensor   # [frames], unique sorted absolute latent IDs


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    chunk_id: int
    source: str
    payloads: Mapping[int, EventPayload]


@dataclass(frozen=True)
class EventSelection:
    batch: int
    anchor: int
    chunk_id: int
    records: tuple[EventRecord, ...]
    selected_event_ids: tuple[int, ...]
    weights: torch.Tensor  # [selected], detached FP32 CPU
    similarities: tuple[float, ...]
    entropy: float
    selector: str


class EventMemoryBank:
    """Bounded per-CFG-branch archive; every mutation returns a new snapshot."""

    def __init__(self, options, batch_size):
        if not isinstance(options, EventMemoryOptions) or type(batch_size) is not int or batch_size < 1:
            raise ValueError("event memory requires validated options and a positive batch size")
        _inference(options)
        self.options, self.batch_size = options, batch_size
        self.records = tuple(() for _ in range(batch_size))
        self.next_event_id = 0

    def _snapshot(self, records, next_event_id=None):
        other = object.__new__(type(self))
        other.options, other.batch_size = self.options, self.batch_size
        other.records = tuple(tuple(items) for items in records)
        other.next_event_id = self.next_event_id if next_event_id is None else next_event_id
        return other

    def __len__(self):
        return max(map(len, self.records), default=0)

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for items in self.records for record in items
                   for payload in record.payloads.values()
                   for t in (payload.descriptor, payload.gram, payload.rhs, payload.denom, payload.frame_ids))

    def _validated_payloads(self, pending):
        if not isinstance(pending, Mapping) or set(pending) != set(self.options.anchors):
            raise ValueError("event commit must provide every configured anchor")
        normalized, expected, expected_ids = {}, None, None
        for anchor in self.options.anchors:
            value = pending[anchor]
            fields = ("descriptor", "gram", "rhs", "denom", "frame_ids")
            if not isinstance(value, Mapping) or set(value) != set(fields):
                raise ValueError("event payload fields mismatch")
            gram = value["gram"]
            if (not isinstance(gram, torch.Tensor) or gram.ndim != 4 or min(gram.shape) < 1
                    or gram.shape[0] != self.batch_size or gram.shape[-1] != gram.shape[-2]):
                raise ValueError("event gram must be [B,H,D,D]")
            b, h, d, _ = gram.shape
            desc, ids = value["descriptor"], value["frame_ids"]
            if not isinstance(desc, torch.Tensor) or desc.ndim != 2 or desc.shape[0] != b or desc.shape[1] < 1:
                raise ValueError("event descriptor must be [B,F]")
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != b or ids.shape[1] < 1
                    or ids.dtype not in (torch.int32, torch.int64) or (ids < 0).any()
                    or (ids[:, 1:] <= ids[:, :-1]).any()):
                raise ValueError("event frame IDs must be sorted nonempty unique integer [B,F]")
            for name, shape in (("descriptor", desc.shape), ("gram", gram.shape), ("rhs", gram.shape), ("denom", (b, h))):
                _finite_float(value[name], shape, name)
            if (value["denom"] <= 0).any():
                raise ValueError("event denominator must be positive")
            if (gram.diagonal(dim1=-2, dim2=-1).sum(-1) <= 0).any():
                raise ValueError("empty event write statistics")
            signature = (h, d, desc.shape[-1])
            if expected is not None and signature != expected:
                raise ValueError("event anchors disagree on H/D/descriptor dimensions")
            if expected_ids is not None and (ids.shape != expected_ids.shape or not torch.equal(ids.cpu(), expected_ids)):
                raise ValueError("event anchors disagree on written frame IDs")
            expected = signature
            expected_ids = ids.detach().cpu()
            if self.records[0]:
                old = self.records[0][0].payloads[anchor]
                if old.gram.shape != (h, d, d) or old.descriptor.shape != (desc.shape[-1],):
                    raise ValueError("event dimensions changed within an episode")
            normalized[anchor] = {name: value[name].detach().to("cpu").contiguous().clone() for name in fields}
            normalized[anchor]["frame_ids"] = normalized[anchor]["frame_ids"].long()
            normalized[anchor]["descriptor"] = F.normalize(normalized[anchor]["descriptor"], dim=-1, eps=1e-12)
        return normalized

    def _retain(self, records):
        if len(records) <= self.options.capacity:
            return tuple(records)
        if self.options.retention == "fifo" or self.options.capacity == 1:
            return tuple(records[-self.options.capacity:])
        records = list(records)
        while len(records) > self.options.capacity:
            # Preserve first/newest; evict the most temporally redundant interior.
            index = min(range(1, len(records) - 1),
                        key=lambda i: (records[i + 1].chunk_id - records[i - 1].chunk_id,
                                       records[i].event_id))
            records.pop(index)
        return tuple(records)

    def with_pending(self, pending, chunk_id, source="generated"):
        """Validate/clone before publication; failure leaves this bank unchanged."""
        _inference(self.options)
        if type(chunk_id) is not int or chunk_id < 0 or source not in ("observed", "generated"):
            raise ValueError("invalid event chunk ID or source")
        if any(items and chunk_id <= items[-1].chunk_id for items in self.records):
            raise ValueError("event chunks must be committed once in increasing order")
        values = self._validated_payloads(pending)
        result = []
        for batch, items in enumerate(self.records):
            payloads = {anchor: EventPayload(*(values[anchor][name][batch].clone() for name in
                        ("descriptor", "gram", "rhs", "denom", "frame_ids"))) for anchor in self.options.anchors}
            record = EventRecord(self.next_event_id, chunk_id, source, MappingProxyType(payloads))
            result.append(self._retain((*items, record)))
        return self._snapshot(result, self.next_event_id + 1)

    def with_transport(self, factors):
        """Cumulatively transport old RHS by R=I+P L P^T; G is unchanged."""
        _inference(self.options)
        if not isinstance(factors, Mapping) or set(factors) != set(self.options.anchors):
            raise ValueError("event transport requires every configured anchor")
        checked = {}
        for anchor in self.options.anchors:
            pair = factors[anchor]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("event transport requires (P,L) per anchor")
            p, l = pair
            if (not isinstance(p, torch.Tensor) or p.ndim != 3 or min(p.shape) < 1
                    or not isinstance(l, torch.Tensor) or l.shape != (self.batch_size, p.shape[0], p.shape[-1], p.shape[-1])):
                raise ValueError("event transport P/L shape mismatch")
            for value, name in ((p, "P"), (l, "L")):
                _finite_float(value, value.shape, name)
            if self.records[0] and self.records[0][0].payloads[anchor].gram.shape[:2] != p.shape[:2]:
                raise ValueError("event transport H/D changed")
            checked[anchor] = (p.detach().cpu(), l.detach().cpu())
        records = []
        for batch, items in enumerate(self.records):
            transformed = []
            for record in items:
                payloads = {}
                for anchor, payload in record.payloads.items():
                    p, l = checked[anchor]
                    rhs = payload.rhs + ((payload.rhs @ p) @ l[batch]) @ p.transpose(-1, -2)
                    if not torch.isfinite(rhs).all():
                        raise ValueError("nonfinite transported event RHS")
                    payloads[anchor] = EventPayload(payload.descriptor, payload.gram, rhs.contiguous(), payload.denom, payload.frame_ids)
                transformed.append(EventRecord(record.event_id, record.chunk_id, record.source, MappingProxyType(payloads)))
            records.append(transformed)
        return self._snapshot(records)

    def select(self, batch, anchor, query, chunk_id):
        _inference(self.options)
        if type(batch) is not int or not 0 <= batch < self.batch_size or anchor not in self.options.anchors:
            raise ValueError("invalid event query batch/anchor")
        if type(chunk_id) is not int or chunk_id < 0:
            raise ValueError("invalid event query chunk")
        if not isinstance(query, torch.Tensor) or query.ndim != 1 or query.numel() < 1:
            raise ValueError("event query must be one descriptor vector")
        _finite_float(query, query.shape, "query")
        items = self.records[batch]
        empty = EventSelection(batch, anchor, chunk_id, (), (), torch.empty(0), (), 0., self.options.selector)
        if not self.options.active or chunk_id < self.options.start_chunk or not items:
            return empty
        if any(item.chunk_id >= chunk_id for item in items):
            raise ValueError("event query cannot read the current/future chunk")
        query = F.normalize(query.detach().cpu(), dim=-1, eps=1e-12)
        if any(item.payloads[anchor].descriptor.shape != query.shape for item in items):
            raise ValueError("event query descriptor dimension mismatch")
        scores = torch.stack([torch.dot(query, item.payloads[anchor].descriptor) for item in items])
        count = min(self.options.topk, len(items))
        if self.options.selector in ("content", "shuffled"):
            chosen = torch.argsort(scores, descending=True, stable=True)[:count].tolist()
            weights = torch.softmax(scores[chosen] / self.options.temperature, dim=0)
        else:
            chosen = [0] if self.options.selector == "initial" else list(range(len(items) - 1, len(items) - count - 1, -1))
            weights = torch.full((len(chosen),), 1. / len(chosen))
        if self.options.selector == "shuffled" and len(items) < 2:
            return empty
        sources = [(index + 1) % len(items) for index in chosen] if self.options.selector == "shuffled" else chosen
        entropy = float(-(weights * weights.clamp_min(1e-12).log()).sum() / math.log(len(weights))) if len(weights) > 1 else 0.
        return EventSelection(batch, anchor, chunk_id, tuple(items[i] for i in sources),
                              tuple(items[i].event_id for i in chosen), weights.clone(),
                              tuple(float(scores[i]) for i in chosen), entropy, self.options.selector)

    def read(self, state, selections, anchor, alpha_s=.5, eps=1e-6):
        """Apply replay to CURRENT post-Correct state; never commit this result."""
        _inference(self.options)
        if (not isinstance(state, torch.Tensor) or state.ndim != 4 or state.shape[0] != self.batch_size
                or min(state.shape) < 1 or state.shape[-1] != state.shape[-2] or anchor not in self.options.anchors):
            raise ValueError("event read state must be [B,H,D,D] at a configured anchor")
        _finite_float(state, state.shape, "read state")
        if (isinstance(alpha_s, bool) or not isinstance(alpha_s, (int, float)) or not math.isfinite(alpha_s)
                or not 0 < alpha_s < 2 or isinstance(eps, bool) or not isinstance(eps, (int, float))
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("invalid event read alpha/eps")
        if not isinstance(selections, (tuple, list)) or len(selections) != self.batch_size:
            raise ValueError("event read requires one selection per batch item")
        result, rows = [], []
        active = False
        for batch, selection in enumerate(selections):
            if (not isinstance(selection, EventSelection) or selection.batch != batch or selection.anchor != anchor
                    or selection.selector != self.options.selector or len(selection.records) != len(selection.weights)
                    or len(selection.records) != len(selection.similarities)
                    or len(selection.records) != len(selection.selected_event_ids)):
                raise ValueError("event read selection does not belong to this batch/anchor")
            _finite_float(selection.weights, (len(selection.records),), "selection weights")
            if ((selection.weights < 0).any() or (selection.records and not torch.isclose(selection.weights.sum(), torch.tensor(1.), atol=1e-6))
                    or not math.isfinite(selection.entropy) or not all(math.isfinite(x) for x in selection.similarities)):
                raise ValueError("invalid event selection weights/similarities")
            current = state[batch]
            delta, events = torch.zeros_like(current), []
            if self.options.active:
                for index, (record, weight) in enumerate(zip(selection.records, selection.weights)):
                    if (record.chunk_id >= selection.chunk_id or anchor not in record.payloads
                            or not any(record is item for item in self.records[batch])):
                        raise ValueError("event selection violates read-before-write")
                    payload = record.payloads[anchor]
                    if payload.gram.shape != current.shape:
                        raise ValueError("event read H/D mismatch")
                    gram, rhs, denom = (value.to(state.device) for value in (payload.gram, payload.rhs, payload.denom))
                    correction = (alpha_s / denom.clamp_min(eps))[..., None, None] * (rhs - gram @ current)
                    delta = delta + float(weight) * correction
                    events.append(dict(selected_event_id=selection.selected_event_ids[index], source_event_id=record.event_id,
                                       chunk_id=record.chunk_id, source=record.source,
                                       frame_ids=payload.frame_ids.tolist(), similarity=selection.similarities[index], weight=float(weight)))
            delta = delta * self.options.gain
            state_norm, delta_norm = current.norm(dim=(-1, -2)), delta.norm(dim=(-1, -2))
            cap = self.options.max_relative_delta * state_norm
            scale = (cap / delta_norm.clamp_min(eps)).clamp_max(1.)
            delta = delta * scale[..., None, None]
            if not torch.isfinite(delta).all():
                raise ValueError("nonfinite event read correction")
            active |= bool(events)
            result.append(current + delta if events else current)
            rows.append(dict(batch=batch, active=bool(events), entropy=selection.entropy, events=events,
                             delta_norm=delta.norm(dim=(-1, -2)).cpu().tolist(),
                             relative_delta=(delta.norm(dim=(-1, -2)) / state_norm.clamp_min(eps)).cpu().tolist(),
                             clip_scale=scale.cpu().tolist()))
        output = torch.stack(result) if active else state
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite event read output")
        return output, dict(active=active, anchor=anchor, selector=self.options.selector, gain=self.options.gain,
                            max_relative_delta=self.options.max_relative_delta, batches=rows,
                            measurement="temporary post-Correct state intervention; no rollout quality claim")


__all__ = ["EventMemoryOptions", "EventMemoryBank", "EventPayload", "EventRecord", "EventSelection",
           "pooled_descriptor", "sufficient_statistics"]
