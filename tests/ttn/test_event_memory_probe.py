"""CPU checks for scoped, baseline-frozen inference diagnostics."""
from copy import deepcopy

import pytest
import torch

from tools.ttn_event_memory_probe import FrozenReadProbe, ProbeComplete
from tools.ttn_event_memory_probe_math import POLICIES, bank_fingerprint, counterfactual_reads
from worldttn.evaluation import tensor_sha256
from worldttn.event_memory import EventMemoryBank, EventMemoryOptions


def make_bank(*, final_chunk=2):
    options = EventMemoryOptions(mode="state-replay", anchors=(11,), start_chunk=1,
                                 gain=.1, max_relative_delta=.05, topk=1)
    bank = EventMemoryBank(options, 2)
    descriptors = (torch.tensor([[1., 0.], [0., 1.]]),
                   torch.tensor([[0., 1.], [1., 0.]]),
                   torch.tensor([[-1., 0.], [-1., 0.]]))
    gram = torch.eye(2).expand(2, 2, 2, 2).clone()
    for index, descriptor in enumerate(descriptors):
        rhs = gram.clone() * (index + 2)
        rhs[..., 0, 1] = index + .25
        rhs[1, ..., 1, 0] = index + .75
        bank = bank.with_pending({11: dict(descriptor=descriptor, gram=gram, rhs=rhs,
            denom=torch.full((2, 2), 2.), frame_ids=torch.full((2, 1), index, dtype=torch.long))},
            final_chunk if index == 2 else index,
            source="observed" if index == 0 else "generated")
    return bank


class TinySampler:
    def sample_chunks(self, latents):
        yield 19, latents[:, :, 58:61], 58, 61


def select_queries(bank, queries, *, chunk=20):
    return tuple(bank.select(batch, 11, queries[batch], chunk) for batch in range(2))


def capture(bank, state, queries, *, calls=1):
    probe = FrozenReadProbe((19,))
    with probe.install(TinySampler):
        selections = select_queries(bank, queries)
        for _ in range(calls):
            output, _ = bank.read(state, selections, 11)
            assert output is state
    return {"snapshots": probe.snapshots, "bank_hashes": probe.bank_hashes,
            "outputs": probe.outputs}


@torch.no_grad()
def test_shadow_capture_keeps_state_identity_and_matches_all_policy_budgets():
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.tensor([[0., 1.], [0., 1.]])
    state_before, query_before = state.clone(), queries.clone()
    bank_before = bank_fingerprint(bank)
    expected = counterfactual_reads(bank, state, queries, 11, 20)
    probe = FrozenReadProbe((19,))
    with probe.install(TinySampler):
        selections = select_queries(bank, queries)
        # These distinct source IDs prove both CFG queries were kept independent.
        assert selections[0].records[0].event_id == 1
        assert selections[1].records[0].event_id == 0
        for _ in range(2):
            output, report = bank.read(state, selections, 11)
            assert output is state and not report["active"]
    assert set(probe.snapshots) == {"20:11:0", "20:11:1"}
    assert probe.bank_hashes == {20: bank_before}
    for saved in probe.snapshots.values():
        assert saved["baseline_state_sha256"] == tensor_sha256(state)
        assert saved["query_sha256"] == tensor_sha256(queries)
        torch.testing.assert_close(saved["common_norm"], expected["common_norm"], rtol=0, atol=0)
        raw_min = torch.stack([delta.norm(dim=(-1, -2))
                               for delta in expected["raw_deltas"].values()]).amin(0)
        torch.testing.assert_close(saved["common_norm"], raw_min, rtol=0, atol=0)
        for policy in POLICIES:
            torch.testing.assert_close(saved["matched_deltas"][policy],
                                       expected["matched_deltas"][policy], rtol=0, atol=0)
            torch.testing.assert_close(saved["matched_deltas"][policy].norm(dim=(-1, -2)),
                                       saved["common_norm"])
    assert not torch.equal(expected["matched_deltas"]["content"], expected["matched_deltas"]["recent"])
    assert bank_fingerprint(bank) == bank_before
    assert torch.equal(state, state_before) and torch.equal(queries, query_before)


@pytest.mark.parametrize("policy", POLICIES)
@torch.no_grad()
def test_injection_uses_fixed_saved_direction_and_leaves_archive_untouched(policy):
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.tensor([[0., 1.], [0., 1.]])
    frozen = capture(bank, state, queries, calls=2)
    before = bank_fingerprint(bank)
    probe = FrozenReadProbe((19,), frozen=frozen, policy=policy)
    with probe.install(TinySampler):
        selections = select_queries(bank, queries)
        first, report = bank.read(state, selections, 11)
        torch.testing.assert_close(first, state + frozen["snapshots"]["20:11:0"]["matched_deltas"][policy],
                                   rtol=0, atol=0)
        assert report["active"] and report["policy"] == policy
        # Later solver states can diverge; the saved direction must not be recomputed.
        later = state * 1.1
        second, _ = bank.read(later, selections, 11)
        torch.testing.assert_close(second, later + frozen["snapshots"]["20:11:1"]["matched_deltas"][policy],
                                   rtol=0, atol=0)
    assert probe.injected_keys == ["20:11:0", "20:11:1"]
    assert bank_fingerprint(bank) == before and torch.equal(state, torch.ones_like(state))
    for row in probe.audit:
        assert max(value for branch in row["relative_head_norms"] for value in branch) <= .050001


@pytest.mark.parametrize("changed", ("state", "query", "archive"))
@torch.no_grad()
def test_first_intervention_rejects_unpaired_state_query_or_archive(changed):
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.eye(2)
    frozen = capture(bank, state, queries)
    if changed == "state":
        state = state * 1.01
    elif changed == "query":
        queries = queries.flip(1)
    else:
        bank = make_bank(final_chunk=3)
    probe = FrozenReadProbe((19,), frozen=frozen, policy="content")
    message = "archive" if changed == "archive" else "identical state/query"
    with pytest.raises(AssertionError, match=message), probe.install(TinySampler):
        bank.read(state, select_queries(bank, queries), 11)
    assert not probe.injected_keys and not probe.audit


@pytest.mark.parametrize("invalid", ("shape", "dtype", "nan", "budget", "cap"))
@torch.no_grad()
def test_frozen_payload_rejected_before_any_injection(invalid):
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.eye(2)
    frozen = deepcopy(capture(bank, state, queries))
    saved = frozen["snapshots"]["20:11:0"]
    delta = saved["matched_deltas"]["content"]
    if invalid == "shape":
        saved["matched_deltas"]["content"] = delta[..., :1]
    elif invalid == "dtype":
        saved["matched_deltas"]["content"] = delta.double()
    elif invalid == "nan":
        delta.flatten()[0] = float("nan")
    elif invalid == "budget":
        saved["common_norm"] *= .5
    else:
        delta *= 100
        saved["common_norm"] = delta.norm(dim=(-1, -2))
    probe = FrozenReadProbe((19,), frozen=frozen, policy="content")
    with pytest.raises(AssertionError, match="shape/dtype/finite|budgets changed|exceeds 5%"), probe.install(TinySampler):
        bank.read(state, select_queries(bank, queries), 11)
    assert not probe.injected_keys and not probe.audit


@torch.no_grad()
def test_off_target_active_read_is_exact_noop_without_capture_or_injection():
    bank = make_bank()
    state, queries = torch.ones(2, 2, 2, 2), torch.eye(2)
    real_output, _ = bank.read(state, select_queries(bank, queries, chunk=19), 11)
    assert not torch.equal(real_output, state)
    probe = FrozenReadProbe((19,))
    with probe.install(TinySampler):
        output, report = bank.read(state, select_queries(bank, queries, chunk=19), 11)
        assert output is state and not report["active"]
    assert not probe.queries and not probe.snapshots and not probe.injected_keys


@torch.no_grad()
def test_repeated_query_or_future_archive_fails_and_restores_methods():
    original = (EventMemoryBank.select, EventMemoryBank.read, TinySampler.sample_chunks)
    probe, bank = FrozenReadProbe((19,)), make_bank()
    with pytest.raises(AssertionError, match="selection unexpectedly repeated"), probe.install(TinySampler):
        bank.select(0, 11, torch.tensor([1., 0.]), 20)
        bank.select(0, 11, torch.tensor([1., 0.]), 20)
    assert (EventMemoryBank.select, EventMemoryBank.read, TinySampler.sample_chunks) == original
    with pytest.raises(ValueError, match="future"), FrozenReadProbe((19,)).install(TinySampler):
        make_bank(final_chunk=20).select(0, 11, torch.tensor([1., 0.]), 20)
    assert (EventMemoryBank.select, EventMemoryBank.read, TinySampler.sample_chunks) == original


def test_sampler_captures_cloned_target_and_exact_prefix_then_closes_on_probe_complete():
    class Sampler:
        closed = False

        def sample_chunks(self, latents):
            try:
                yield 18, latents[:, :, 55:58], 55, 58
                yield 19, latents[:, :, 58:61].clone(), 58, 61
                raise AssertionError("must stop before asking the original sampler for another chunk")
            finally:
                self.closed = True

    original = (EventMemoryBank.select, EventMemoryBank.read, Sampler.sample_chunks)
    latents = torch.arange(61.).reshape(1, 1, 61, 1, 1)
    expected = latents[:, :, 58:61].clone()
    probe, sampler = FrozenReadProbe((19,), stop_index=19), Sampler()
    with pytest.raises(ProbeComplete), probe.install(Sampler):
        stream = sampler.sample_chunks(latents)
        assert next(stream)[0] == 18
        assert next(stream)[0] == 19
        next(stream)
    assert sampler.closed
    assert (EventMemoryBank.select, EventMemoryBank.read, Sampler.sample_chunks) == original
    saved = probe.outputs[19]
    assert saved["prefix_sha256"] == tensor_sha256(latents[:, :, :58])
    assert saved["start"] == 58 and saved["end"] == 61
    latents.add_(100)
    assert torch.equal(saved["latents"], expected)


@pytest.mark.parametrize("failure", ("boundary", "upstream", "consumer"))
def test_sampler_restores_patches_and_closes_generator_after_failures(failure):
    class Sampler:
        closed = False

        def sample_chunks(self, latents):
            try:
                if failure == "boundary":
                    yield 19, latents[:, :, 57:60], 57, 60
                else:
                    yield 19, latents[:, :, 58:61], 58, 61
                    raise RuntimeError("upstream sampler failed")
            finally:
                self.closed = True

    original = (EventMemoryBank.select, EventMemoryBank.read, Sampler.sample_chunks)
    sampler, probe = Sampler(), FrozenReadProbe((19,))
    latents = torch.ones(1, 1, 61, 1, 1)
    with pytest.raises((AssertionError, RuntimeError)), probe.install(Sampler):
        stream = sampler.sample_chunks(latents)
        try:
            next(stream)
            if failure == "consumer":
                raise RuntimeError("caller failed after receiving a chunk")
            next(stream)
        finally:
            stream.close()
    assert sampler.closed
    assert (EventMemoryBank.select, EventMemoryBank.read, Sampler.sample_chunks) == original


@pytest.mark.parametrize("kwargs", ({"targets": ()}, {"targets": (18,)}, {"targets": (19.,)},
    {"targets": (19,), "stop_index": 29}, {"targets": (19,), "policy": "content"},
    {"targets": (19,), "frozen": {}}, {"targets": (19,), "frozen": {}, "policy": "unknown"}))
def test_invalid_probe_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        FrozenReadProbe(**kwargs)
