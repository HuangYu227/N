"""Same-state event controls: policy, strength, branch and archive invariants."""
from copy import copy
from dataclasses import replace
import json
from types import MappingProxyType

import pytest
import torch

from tools.ttn_event_memory_probe_math import POLICIES, bank_fingerprint, counterfactual_reads
from worldttn.event_memory import EventMemoryBank, EventMemoryOptions


def make_bank(*, zero_initial_head=False):
    options = EventMemoryOptions(mode="state-replay", anchors=(11,), start_chunk=1,
                                 gain=1., max_relative_delta=1., topk=2)
    bank = EventMemoryBank(options, 2)
    descriptors = [torch.tensor([[1., 0.], [0., 1.]]),
                   torch.tensor([[0., 1.], [1., 0.]]),
                   torch.tensor([[-1., 0.], [-1., 0.]])]
    gram = torch.eye(2).expand(2, 2, 2, 2).clone()
    for index, descriptor in enumerate(descriptors):
        rhs = torch.ones_like(gram) * (index + 2)
        rhs[1] *= 1.5
        if zero_initial_head and index == 0:
            rhs[:, 0] = 1.
        bank = bank.with_pending({11: dict(descriptor=descriptor, gram=gram, rhs=rhs,
                                denom=torch.full((2, 2), 2.),
                                frame_ids=torch.full((2, 1), index, dtype=torch.long))},
                                index, source="observed" if index == 0 else "generated")
    return bank


@torch.no_grad()
def test_policies_are_top1_and_cfg_branches_choose_independently():
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.tensor([[0., 1.], [0., 1.]])
    fingerprint, before_state, before_query = bank_fingerprint(bank), state.clone(), queries.clone()
    result = counterfactual_reads(bank, state, queries, 11, 3)
    ids = result["diagnostics"]["source_event_ids"]
    assert ids == {"content": [[1], [0]], "initial": [[0], [0]],
                   "recent": [[2], [2]], "shuffled": [[2], [1]]}
    assert result["diagnostics"]["selected_event_ids"]["shuffled"] == [[1], [0]]
    assert [row["similarity"] for row in result["candidate_scores"][0]] == [0., 1., 0.]
    assert [row["similarity"] for row in result["candidate_scores"][1]] == [1., 0., 0.]
    assert bank.options.topk == 2 and bank.options.selector == "content"
    assert bank_fingerprint(bank) == fingerprint
    torch.testing.assert_close(state, before_state, rtol=0, atol=0)
    torch.testing.assert_close(queries, before_query, rtol=0, atol=0)
    json.dumps(result["diagnostics"], allow_nan=False)
    json.dumps(result["reports"], allow_nan=False)
    json.dumps(result["candidate_scores"], allow_nan=False)


@torch.no_grad()
def test_norm_matching_only_scales_down_and_zero_common_heads_are_zero():
    state, queries = torch.ones(2, 2, 2, 2), torch.tensor([[0., 1.], [0., 1.]])
    for zero in (False, True):
        result = counterfactual_reads(make_bank(zero_initial_head=zero), state, queries, 11, 3)
        raw_norms = torch.stack([result["raw_deltas"][p].norm(dim=(-1, -2)) for p in POLICIES])
        expected = raw_norms.amin(0)
        expected = torch.where(expected > 1e-6, expected, 0.)
        torch.testing.assert_close(result["common_norm"], expected, rtol=0, atol=0)
        for policy in POLICIES:
            matched = result["matched_deltas"][policy].norm(dim=(-1, -2))
            torch.testing.assert_close(matched, expected)
            assert (matched <= result["raw_deltas"][policy].norm(dim=(-1, -2)) + 1e-7).all()
        if zero:
            assert (result["common_norm"][:, 0] == 0).all()
            assert result["diagnostics"]["zero_common_head_fraction"] == .5
            assert result["diagnostics"]["direction_cosines"]["content:initial"][0][0] is None


@torch.no_grad()
def test_fingerprint_identifies_payloads_order_cfg_and_options_but_not_policy():
    bank = make_bank()
    original = bank_fingerprint(bank)
    view = copy(bank)
    view.options = replace(bank.options, selector="shuffled", topk=1)
    assert bank_fingerprint(view) == original
    view.options = replace(bank.options, gain=.5)
    assert bank_fingerprint(view) != original
    for field in ("descriptor", "gram", "rhs", "denom", "frame_ids"):
        record = bank.records[0][0]
        payload = record.payloads[11]
        value = getattr(payload, field).clone()
        value.flatten()[0] += 1
        modified = replace(payload, **{field: value})
        changed = replace(record, payloads=MappingProxyType({11: modified}))
        altered = bank._snapshot(((changed, *bank.records[0][1:]), bank.records[1]))
        assert bank_fingerprint(altered) != original
        assert bank_fingerprint(bank) == original
    assert bank_fingerprint(bank._snapshot((bank.records[1], bank.records[0]))) != original
    assert bank_fingerprint(bank._snapshot(bank.records, bank.next_event_id + 1)) != original


@torch.no_grad()
def test_current_or_future_archive_reads_and_foreign_selections_rejected():
    bank, state = make_bank(), torch.ones(2, 2, 2, 2)
    queries = torch.tensor([[0., 1.], [0., 1.]])
    with pytest.raises(ValueError, match="future"):
        counterfactual_reads(bank, state, queries, 11, 2)
    foreign = make_bank()
    original_read = EventMemoryBank.read

    def foreign_read(view, current, selections, anchor, alpha_s, eps):
        selections = tuple(foreign.select(b, anchor, queries[b], 3) for b in range(2))
        return original_read(view, current, selections, anchor, alpha_s, eps)

    with pytest.raises(ValueError, match="read-before-write"):
        counterfactual_reads(bank, state, queries, 11, 3, read_fn=foreign_read)


@torch.no_grad()
def test_original_read_and_select_defaults_bypass_scope_monkeypatch(monkeypatch):
    bank = make_bank()

    def forbidden(*args, **kwargs):
        raise AssertionError("recursed into the runner hook")

    monkeypatch.setattr(EventMemoryBank, "read", forbidden)
    monkeypatch.setattr(EventMemoryBank, "select", forbidden)
    result = counterfactual_reads(bank, torch.ones(2, 2, 2, 2), torch.eye(2), 11, 3)
    assert set(result["reports"]) == set(POLICIES)


@torch.no_grad()
def test_perturbing_state_or_queries_changes_controls_without_mutating_bank():
    bank = make_bank()
    before = bank_fingerprint(bank)
    state, query = torch.ones(2, 2, 2, 2), torch.eye(2)
    original = counterfactual_reads(bank, state, query, 11, 3)
    changed_state = counterfactual_reads(bank, state * .5, query, 11, 3)
    changed_query = counterfactual_reads(bank, state, query.flip(1), 11, 3)
    assert not torch.equal(original["raw_deltas"]["content"], changed_state["raw_deltas"]["content"])
    assert (original["diagnostics"]["source_event_ids"]["content"] !=
            changed_query["diagnostics"]["source_event_ids"]["content"])
    assert bank_fingerprint(bank) == before


def test_requires_no_grad_and_strict_fp32_inputs():
    with torch.no_grad():
        bank = make_bank()
    state, query = torch.ones(2, 2, 2, 2), torch.eye(2)
    with pytest.raises(ValueError, match="disable gradients"):
        counterfactual_reads(bank, state, query, 11, 3)
    with torch.no_grad():
        with pytest.raises(ValueError, match="state"):
            counterfactual_reads(bank, state.double(), query, 11, 3)
        with pytest.raises(ValueError, match="queries"):
            counterfactual_reads(bank, state, query.double(), 11, 3)
