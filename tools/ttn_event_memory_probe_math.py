"""Same-state, norm-matched event reads for inference diagnostics only."""
from copy import copy
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math

import torch
from torch.nn import functional as F

from worldttn.evaluation import tensor_sha256
from worldttn.event_memory import EventMemoryBank


POLICIES = ("content", "initial", "recent", "shuffled")


def bank_fingerprint(bank):
    """Identify a bank's values, independent of its temporary retrieval policy.

    Tensor hashes include shape and dtype. Record order and CFG branch membership
    are preserved; selector/topk alone may vary in the counterfactual bank views.
    """
    if not isinstance(bank, EventMemoryBank):
        raise ValueError("probe requires an EventMemoryBank")
    options = asdict(bank.options)
    for name in ("selector", "topk"):
        options.pop(name)
    records = []
    for batch, items in enumerate(bank.records):
        rows = []
        for record in items:
            payloads = {
                str(anchor): {name: tensor_sha256(getattr(payload, name)) for name in
                              ("descriptor", "gram", "rhs", "denom", "frame_ids")}
                for anchor, payload in sorted(record.payloads.items())
            }
            rows.append(dict(event_id=record.event_id, chunk_id=record.chunk_id,
                             source=record.source, payloads=payloads))
        records.append(dict(batch=batch, events=rows))
    identity = dict(options=options, batch_size=bank.batch_size,
                    next_event_id=bank.next_event_id, records=records)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def counterfactual_reads(bank, state, queries, anchor, chunk_id, alpha_s=.5, eps=1e-6,
                         read_fn=EventMemoryBank.read, select_fn=EventMemoryBank.select):
    """Read all top-1 policies from identical bank/state, then match per-head norm.

    The common norm is the minimum observed intervention norm. Zero/tiny heads
    are zeroed for every policy. Scaling is downwards only, so this diagnostic
    cannot amplify an intervention beyond the original state-relative cap.
    Captured read/select methods bypass a caller's temporary diagnostic hooks.
    """
    if torch.is_grad_enabled():
        raise ValueError("counterfactual event reads are inference-only; disable gradients")
    if not isinstance(bank, EventMemoryBank):
        raise ValueError("probe requires an EventMemoryBank")
    if (not isinstance(state, torch.Tensor) or state.ndim != 4 or min(state.shape) < 1
            or state.shape[0] != bank.batch_size or state.shape[-1] != state.shape[-2]
            or state.dtype != torch.float32 or not torch.isfinite(state).all()):
        raise ValueError("probe state must be finite FP32 [B,H,D,D]")
    if (not isinstance(queries, torch.Tensor) or queries.ndim != 2
            or queries.shape[0] != bank.batch_size or queries.shape[1] < 1
            or queries.dtype != torch.float32 or not torch.isfinite(queries).all()):
        raise ValueError("probe queries must be finite FP32 [B,F]")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("probe eps must be finite and positive")
    if anchor not in bank.options.anchors or type(chunk_id) is not int or chunk_id < 0:
        raise ValueError("invalid probe anchor/chunk")
    if any(record.chunk_id >= chunk_id for items in bank.records for record in items):
        raise ValueError("event query cannot read the current/future chunk")

    raw, reports = {}, {}
    for policy in POLICIES:
        view = copy(bank)
        view.options = replace(bank.options, selector=policy, topk=1)
        selections = tuple(select_fn(view, batch, anchor, queries[batch], chunk_id)
                           for batch in range(bank.batch_size))
        output, reports[policy] = read_fn(view, state, selections, anchor, alpha_s, eps)
        raw[policy] = output - state

    norms = {policy: delta.norm(dim=(-1, -2)) for policy, delta in raw.items()}
    if not all(torch.isfinite(value).all() for value in norms.values()):
        raise ValueError("nonfinite event intervention norm")
    common = torch.stack(tuple(norms.values())).amin(0)
    common = torch.where(common > eps, common, 0.)
    matched, scales = {}, {}
    for policy, delta in raw.items():
        scale = (common / norms[policy].clamp_min(eps)).clamp_max(1.)
        matched[policy] = delta * scale[..., None, None]
        scales[policy] = scale
        torch.testing.assert_close(matched[policy].norm(dim=(-1, -2)), common,
                                   rtol=2e-5, atol=eps * .01)
        if not torch.isfinite(matched[policy]).all() or (scale > 1).any():
            raise ValueError("invalid norm-matched event intervention")

    scores = []
    normalized = F.normalize(queries.detach().cpu(), dim=-1, eps=1e-12)
    for batch, items in enumerate(bank.records):
        scores.append([dict(event_id=record.event_id, chunk_id=record.chunk_id,
                            source=record.source,
                            similarity=float(torch.dot(normalized[batch], record.payloads[anchor].descriptor)))
                       for record in items])
    cosines = {}
    for left, right in itertools.combinations(POLICIES, 2):
        directions = [raw[policy] / norms[policy].clamp_min(eps)[..., None, None]
                      for policy in (left, right)]
        values = (directions[0] * directions[1]).sum(dim=(-1, -2))
        valid = (norms[left] > eps) & (norms[right] > eps)
        cosines[f"{left}:{right}"] = [
            [float(value) if bool(keep) else None for value, keep in zip(row, mask)]
            for row, mask in zip(values.clamp(-1., 1.).cpu(), valid.cpu())
        ]
    diagnostics = dict(
        norms={policy: dict(raw=norms[policy].cpu().tolist(),
                            matched=matched[policy].norm(dim=(-1, -2)).cpu().tolist(),
                            scale=scales[policy].cpu().tolist()) for policy in POLICIES},
        direction_cosines=cosines,
        zero_common_head_fraction=float((common == 0).float().mean()),
        common_norm=common.cpu().tolist(),
        selected_event_ids={policy: [
            [event["selected_event_id"] for event in row["events"]]
            for row in reports[policy]["batches"]] for policy in POLICIES},
        source_event_ids={policy: [
            [event["source_event_id"] for event in row["events"]]
            for row in reports[policy]["batches"]] for policy in POLICIES},
        scale_down_only=True,
        measurement="same-state top-1 read; per-head norm matching; no rollout quality claim",
    )
    return dict(raw_deltas=raw, matched_deltas=matched, common_norm=common,
                reports=reports, candidate_scores=scores, diagnostics=diagnostics)
