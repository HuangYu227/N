"""Event replay math and immutable archive contracts, without SANA imports."""
from dataclasses import replace
import json

import pytest
import torch

from worldttn.core import CayleyFactors, correct, write_weights
from worldttn.event_memory import (EventMemoryBank, EventMemoryOptions, pooled_descriptor,
                                   sufficient_statistics)


def options(**changes):
    return replace(EventMemoryOptions(mode="state-replay", anchors=(11,), start_chunk=1), **changes)


def payload(seed=1, batch=2, heads=2, dim=4, tokens=7, frames=(0,), descriptor=None):
    generator = torch.Generator().manual_seed(seed)
    k0, v = [torch.randn(batch, heads, tokens, dim, generator=generator) for _ in range(2)]
    k = k0.clone()
    write = torch.ones(batch, tokens, dtype=torch.bool)
    beta = torch.rand(batch, heads, tokens, generator=generator)
    weight = write_weights(beta, write, 1e-6)
    gram, rhs, denom = sufficient_statistics(k, v, weight, write, 1e-6)
    return dict(descriptor=pooled_descriptor(k0, v, write) if descriptor is None else descriptor,
                gram=gram, rhs=rhs, denom=denom,
                frame_ids=torch.tensor(frames).expand(batch, -1).clone())


def selections(bank, anchor=11, chunk=10):
    return tuple(bank.select(b, anchor, bank.records[b][0].payloads[anchor].descriptor, chunk)
                 for b in range(bank.batch_size))


@torch.no_grad()
def test_statistics_equal_correct_with_normalized_weights_and_masks():
    torch.manual_seed(17)
    state = torch.randn(2, 3, 6, 6)
    k, v = [torch.randn(2, 3, 9, 6) for _ in range(2)]
    beta = torch.rand(2, 3, 9)
    mask = torch.tensor([[True] * 7 + [False] * 2, [False] * 2 + [True] * 7])
    candidate, weight = correct(state, k, v, beta, mask)
    gram, rhs, denom = sufficient_statistics(k, v, weight, mask, 1e-6)
    replay = state + (.5 / denom)[..., None, None] * (rhs - gram @ state)
    torch.testing.assert_close(candidate, replay, rtol=2e-6, atol=3e-7)
    assert all(item.dtype == torch.float32 and not item.requires_grad for item in (gram, rhs, denom))


@torch.no_grad()
def test_descriptor_has_per_head_mean_std_and_ignores_masked_nan():
    k = torch.tensor([[[[1., 2.], [3., 6.], [float("nan"), 0.]],
                       [[5., 4.], [9., 8.], [0., float("nan")]]]])
    v = k * 2
    descriptor = pooled_descriptor(k, v, torch.tensor([[True, True, False]]))
    expected = torch.tensor([[2., 4., 7., 6., 1., 2., 2., 2.,
                              4., 8., 14., 12., 2., 4., 4., 4.]])
    expected = torch.nn.functional.normalize(expected, dim=-1)
    torch.testing.assert_close(descriptor, expected)
    assert descriptor.shape == (1, 16) and torch.isfinite(descriptor).all()


@torch.no_grad()
def test_real_descriptor_grid_has_exact_batch_isolation_and_spatial_layout():
    generator = torch.Generator().manual_seed(32)
    k, v = [torch.randn(2, 20, 3 * 22 * 40, 112, generator=generator) for _ in range(2)]
    write = torch.ones(2, 3 * 22 * 40, dtype=torch.bool)
    write[0, :22 * 40] = False
    combined = pooled_descriptor(k, v, write, grid=(3, 22, 40))
    isolated = torch.cat([pooled_descriptor(k[b:b + 1], v[b:b + 1], write[b:b + 1], grid=(3, 22, 40))
                          for b in range(2)])
    assert combined.shape == (2, 16 * 20 * 112)
    torch.testing.assert_close(combined, isolated, rtol=0, atol=0)
    cell_keys = torch.arange(1, 5).float().reshape(1, 1, 4, 1).expand(1, 1, 4, 2)
    cell_values = cell_keys * 2
    small = torch.ones(1, 4, dtype=torch.bool)
    permutation = torch.tensor([3, 2, 1, 0])
    torch.testing.assert_close(pooled_descriptor(cell_keys, cell_values, small),
                               pooled_descriptor(cell_keys[:, :, permutation], cell_values[:, :, permutation], small))
    assert not torch.allclose(pooled_descriptor(cell_keys, cell_values, small, (1, 2, 2)),
                              pooled_descriptor(cell_keys[:, :, permutation], cell_values[:, :, permutation], small, (1, 2, 2)))
    with pytest.raises(ValueError): pooled_descriptor(k, v, write, (3, 1, 880))


@pytest.mark.parametrize("bad", ["empty_mask", "empty_head", "negative", "shape", "nan"])
@torch.no_grad()
def test_statistics_reject_invalid_or_empty_writes(bad):
    k, v, weight = torch.ones(2, 2, 3, 4), torch.ones(2, 2, 3, 4), torch.ones(2, 2, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    if bad == "empty_mask": mask[0] = False
    if bad == "empty_head": weight[0, 1] = 0
    if bad == "negative": weight[0, 1, 0] = -1
    if bad == "shape": weight = weight[:, :, :1]
    if bad == "nan": k[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        sufficient_statistics(k, v, weight, mask, 1e-6)


@torch.no_grad()
def test_bank_payload_owns_detached_cpu_storage_and_commit_is_atomic():
    bank = EventMemoryBank(options(anchors=(11, 15)), 2)
    pending = {11: payload(), 15: payload(2)}
    updated = bank.with_pending(pending, 0, source="observed")
    saved = updated.records[0][0].payloads[11]
    assert len(bank) == 0 and len(updated) == 1
    assert saved.gram.device.type == "cpu" and not saved.gram.requires_grad
    original = saved.gram.clone()
    pending[11]["gram"].zero_()
    torch.testing.assert_close(saved.gram, original, rtol=0, atol=0)
    assert saved.gram.data_ptr() != pending[11]["gram"].data_ptr()
    bad = {11: payload(3), 15: payload(4)}
    bad[15]["rhs"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError): updated.with_pending(bad, 1)
    assert len(updated) == 1 and updated.next_event_id == 1
    torch.testing.assert_close(saved.gram, original, rtol=0, atol=0)
    with pytest.raises(ValueError): updated.with_pending({11: payload()}, 1)
    with pytest.raises(ValueError): updated.with_pending({}, 1)
    with pytest.raises(ValueError): updated.with_pending({11: payload(), 15: payload(2)}, 0)


@pytest.mark.parametrize("mutation", ["dtype", "batch", "heads", "denom", "frames", "anchor_frames", "empty"])
@torch.no_grad()
def test_bank_rejects_shape_dtype_and_empty_payloads(mutation):
    bank = EventMemoryBank(options(anchors=(11, 15)), 2)
    left, right = payload(), payload(2)
    if mutation == "dtype": right["rhs"] = right["rhs"].double()
    if mutation == "batch": right["gram"] = right["gram"][:1]
    if mutation == "heads": right = payload(2, heads=1)
    if mutation == "denom": right["denom"][0, 0] = 0
    if mutation == "frames": right["frame_ids"] = torch.tensor([[1, 1], [1, 2]])
    if mutation == "anchor_frames": right["frame_ids"] = right["frame_ids"] + 1
    if mutation == "empty": right["gram"].zero_()
    with pytest.raises(ValueError): bank.with_pending({11: left, 15: right}, 0)
    assert len(bank) == 0


@torch.no_grad()
def test_multi_transport_covariance_and_old_snapshot_unchanged():
    bank = EventMemoryBank(options(gain=.1, max_relative_delta=1.), 2).with_pending({11: payload()}, 0)
    state = torch.randn(2, 2, 4, 4) * 4
    before, _ = bank.read(state, selections(bank), 11)
    original_rhs = bank.records[0][0].payloads[11].rhs.clone()
    product = torch.eye(4).expand(2, 2, 4, 4).clone()
    current = bank
    generator = torch.Generator().manual_seed(88)
    for _ in range(3):
        u, v = [torch.randn(2, 2, 4, generator=generator) * .2 for _ in range(2)]
        coefficient = torch.randn(2, 2, 2, generator=generator) * .1
        factors = CayleyFactors(u, v, coefficient)
        rotation = factors.right(torch.eye(4).expand(2, 2, 4, 4))
        product = product @ rotation
        current = current.with_transport({11: (factors.p, factors.l)})
    after, stats = current.read(state @ product, selections(current), 11)
    torch.testing.assert_close(after, before @ product, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(bank.records[0][0].payloads[11].rhs, original_rhs, rtol=0, atol=0)
    expected_rhs = torch.stack([bank.records[b][0].payloads[11].rhs for b in range(2)]) @ product
    torch.testing.assert_close(torch.stack([current.records[b][0].payloads[11].rhs for b in range(2)]), expected_rhs,
                               rtol=2e-5, atol=2e-6)
    assert stats["active"] and all(item["clip_scale"] == [1., 1.] for item in stats["batches"])
    with pytest.raises(ValueError): current.with_transport({11: (factors.p, factors.l[:, :, :1])})


@torch.no_grad()
def test_batch_isolation_and_content_selection():
    bank = EventMemoryBank(options(), 2)
    d0 = torch.tensor([[1., 0.], [0., 1.]])
    d1 = torch.tensor([[0., 1.], [1., 0.]])
    bank = bank.with_pending({11: payload(descriptor=d0)}, 0)
    bank = bank.with_pending({11: payload(2, descriptor=d1)}, 1)
    query = torch.tensor([1., 0.])
    left, right = [bank.select(b, 11, query, 2) for b in range(2)]
    assert left.selected_event_ids[0] == 0 and right.selected_event_ids[0] == 1
    state = torch.randn(2, 2, 4, 4)
    output, stats = bank.read(state, (left, right), 11)
    json.dumps(stats, allow_nan=False)
    with pytest.raises(ValueError): bank.read(state, (right, left), 11)
    forged = replace(right, batch=0)
    with pytest.raises(ValueError): bank.read(state, (forged, right), 11)
    isolated = EventMemoryBank(options(), 1)
    one = {key: value[:1].clone() for key, value in payload(descriptor=d0).items()}
    isolated = isolated.with_pending({11: one}, 0)
    one = {key: value[:1].clone() for key, value in payload(2, descriptor=d1).items()}
    isolated = isolated.with_pending({11: one}, 1)
    selected = isolated.select(0, 11, query, 2)
    expected, _ = isolated.read(state[:1], (selected,), 11)
    torch.testing.assert_close(output[:1], expected, rtol=0, atol=0)


@torch.no_grad()
def test_empty_zero_gain_and_activation_boundary_return_exact_input():
    state = torch.randn(2, 2, 4, 4)
    for config in (options(), options(mode="off"), options(gain=0.)):
        bank = EventMemoryBank(config, 2)
        selected = tuple(bank.select(b, 11, torch.ones(3), 10) for b in range(2))
        output, stats = bank.read(state, selected, 11)
        assert output is state and not stats["active"]
    bank = EventMemoryBank(options(start_chunk=5), 2).with_pending({11: payload()}, 0)
    selected = selections(bank, chunk=4)
    assert not selected[0].records
    assert bank.read(state, selected, 11)[0] is state


@torch.no_grad()
def test_replay_recomputes_residual_against_current_state_and_enforces_cap():
    config = options(gain=1., max_relative_delta=.01)
    bank = EventMemoryBank(config, 2).with_pending({11: payload()}, 0)
    state = torch.randn(2, 2, 4, 4) * .01
    before = state.clone()
    output, stats = bank.read(state, selections(bank), 11)
    change = (output - state).norm(dim=(-1, -2))
    assert (change <= .01001 * state.norm(dim=(-1, -2))).all()
    assert any(min(row["clip_scale"]) < 1 for row in stats["batches"])
    zero, _ = bank.read(torch.zeros_like(state), selections(bank), 11)
    assert zero.count_nonzero() == 0 and torch.isfinite(zero).all()
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    unlimited = EventMemoryBank(options(gain=.1, max_relative_delta=1.), 2).with_pending({11: payload()}, 0)
    first, _ = unlimited.read(state, selections(unlimited), 11)
    second, _ = unlimited.read(state + 2, selections(unlimited), 11)
    assert not torch.allclose(first - state, second - (state + 2))


@torch.no_grad()
def test_coverage_preserves_long_horizon_while_fifo_is_explicit():
    coverage = EventMemoryBank(options(), 1)
    fifo = EventMemoryBank(options(retention="fifo"), 1)
    for chunk in range(41):
        pending = {11: payload(chunk + 1, batch=1, frames=(chunk,))}
        coverage = coverage.with_pending(pending, chunk, source="observed" if chunk == 0 else "generated")
        fifo = fifo.with_pending(pending, chunk, source="observed" if chunk == 0 else "generated")
    chunks = [row.chunk_id for row in coverage.records[0]]
    assert len(chunks) == 8 and chunks[0] == 0 and chunks[-1] == 40
    assert any(10 <= i <= 30 for i in chunks)
    assert [row.chunk_id for row in fifo.records[0]] == list(range(33, 41))
    assert coverage.storage_bytes > 0


@torch.no_grad()
def test_shuffled_pairs_whole_event_statistics_and_is_stable():
    bank = EventMemoryBank(options(selector="shuffled"), 1)
    bank = bank.with_pending({11: payload(batch=1)}, 0, source="observed")
    assert not selections(bank)[0].records
    bank = bank.with_pending({11: payload(2, batch=1)}, 1)
    bank = bank.with_pending({11: payload(3, batch=1)}, 2)
    first, second = selections(bank)[0], selections(bank)[0]
    assert first.selected_event_ids == second.selected_event_ids
    for selected_id, source in zip(first.selected_event_ids, first.records):
        assert selected_id != source.event_id
        assert source is bank.records[0][source.event_id]
    output, stats = bank.read(torch.randn(1, 2, 4, 4), (first,), 11)
    assert torch.isfinite(output).all()
    assert all(item["selected_event_id"] != item["source_event_id"] for item in stats["batches"][0]["events"])


@torch.no_grad()
def test_recent_initial_and_equal_content_ties_are_deterministic():
    banks = [EventMemoryBank(options(selector=mode), 1) for mode in ("content", "recent", "initial")]
    descriptor = torch.ones(1, 3)
    for chunk in range(3):
        banks = [bank.with_pending({11: payload(chunk + 1, batch=1, descriptor=descriptor)}, chunk) for bank in banks]
    selected = [bank.select(0, 11, descriptor[0], 3) for bank in banks]
    assert selected[0].selected_event_ids == (0, 1)
    assert selected[1].selected_event_ids == (2, 1)
    assert selected[2].selected_event_ids == (0,)
    with pytest.raises(ValueError): banks[0].select(0, 11, descriptor[0], 2)


@pytest.mark.parametrize("changes", [dict(anchors=(12,)), dict(capacity=0), dict(topk=9), dict(gain=float("nan")),
                                     dict(temperature=0), dict(max_read_sigma=2), dict(retention="bad"), dict(selector="bad")])
def test_invalid_options_and_training_guard(changes):
    with pytest.raises(ValueError): options(**changes)
    with torch.enable_grad(), pytest.raises(ValueError, match="inference-only"):
        EventMemoryBank(options(), 1)
