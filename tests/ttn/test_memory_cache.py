"""Clean transaction cache, participative selection and temporal-view contracts."""
from dataclasses import replace

import pytest
import torch

from worldttn.geometry import apply_complex_rope
from worldttn.memory_cache import (MemoryCache, detach_cache, prefix_digest, sources,
    storage_bytes, temporal_realign, update_cache, validate_cache)
from worldttn.replay import Observation


def features(frames, *, batch=1, heads=2, hw=(1, 2), dim=8, grad=False):
    count = len(frames)*hw[0]*hw[1]
    generator = torch.Generator().manual_seed(71 + sum(frames))
    k, v, q = (torch.randn(batch, heads, count, dim, dtype=torch.float64,
                           generator=generator).requires_grad_(grad) for _ in range(3))
    w = torch.ones(batch, heads, count, dtype=torch.float64, requires_grad=grad)
    write = torch.ones(batch, count, dtype=torch.bool)
    ids = torch.tensor(frames)[None].expand(batch, -1)
    return k, v, w, q, write, ids, (len(frames), *hw)


def identities(cache, batch=0):
    o = cache.observation
    return list(zip(o.frame_ids[batch][o.frame_ids[batch] >= 0].tolist(),
                    o.token_indices[batch][o.frame_ids[batch] >= 0].tolist()))


def update(previous, frames, **kwargs):
    return update_cache(previous, *features(frames), capacity_frames=4,
                        prefix_frames=1, recent_frames=1, **kwargs)


def test_participative_matches_token_pair_oracle_and_preserves_prefix():
    torch.manual_seed(5)
    previous = update(None, [0, 1, 2, 3])
    before = prefix_digest(previous, 1)
    data = features([4, 5])
    actual = update_cache(previous, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    # The last frame's two queries, all heads, score every candidate old token.
    k, v, w, q, *_ = data
    all_keys = torch.cat((previous.observation.key, k), dim=2)
    queries = q[:, :, -2:]
    pairs = queries @ all_keys.transpose(-1, -2)
    score = pairs.sum((1, 2))[0]/(8**.5*2)
    eligible = torch.arange(2, 10)
    selected = eligible[score[eligible].argsort(descending=True, stable=True)[:4]]
    expected = sorted([0, 1, 10, 11, *selected.tolist()])
    assert identities(actual) == [(i//2, i%2) for i in expected]
    assert prefix_digest(actual, 1) == before
    torch.testing.assert_close(actual.observation.key[:, :, :2], previous.observation.key[:, :, :2],
                               rtol=0, atol=0)
    assert actual.observation.key.shape[-2] == 8


def test_recent_query_history_is_used_and_query_gradient_is_not_retained():
    previous = update_cache(None, *features([0, 1, 2, 3], grad=True),
                            capacity_frames=4, prefix_frames=1, recent_frames=2)
    new = list(features([4], grad=True))
    new[3] = torch.zeros_like(new[3], requires_grad=True)
    actual = update_cache(previous, *new, capacity_frames=4, prefix_frames=1, recent_frames=2)
    # Query from preceding frame 3 still participates despite zero current Q.
    scores = torch.einsum("hd,hnd->n", previous.query[0, :, -2:].sum(1),
                          previous.observation.key[0, :, 2:6])
    chosen = (scores.argsort(descending=True, stable=True)[:2] + 2).tolist()
    assert identities(actual) == sorted([(0, 0), (0, 1), (3, 0), (3, 1), (4, 0), (4, 1)]
                                              + [(i//2, i%2) for i in chosen])
    assert not actual.query.requires_grad


def test_stable_ties_and_fifo_are_distinct_controls():
    data = list(features(list(range(7))))
    data[3].zero_()
    pc = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    fifo = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1, selection="fifo")
    assert identities(pc) == [(f, i) for f in (0, 1, 2, 6) for i in range(2)]
    assert identities(fifo) == [(f, i) for f in (0, 4, 5, 6) for i in range(2)]


def test_cfg_batches_do_not_share_selection_and_padding_is_inert():
    data = list(features(list(range(6)), batch=2))
    data[0].zero_(); data[3].zero_()
    # Opposite query signs must select opposite middle frame ranks.
    data[0][:, :, :, 0] = torch.arange(12, dtype=torch.float64)
    data[3][0, :, :, 0] = 1
    data[3][1, :, :, 0] = -1
    data[4][1, 3:] = False
    cache = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    assert identities(cache, 0) == [(f, i) for f in (0, 3, 4, 5) for i in range(2)]
    assert identities(cache, 1) == [(0, 0), (0, 1), (1, 0)]
    assert (cache.observation.frame_ids[1, 3:] == -1).all()
    assert (cache.observation.weight[1, :, 3:] == 0).all()
    assert (cache.observation.key[1, :, 3:] == 0).all()
    validate_cache(cache)
    # Same candidate set but opposite CFG Q gives genuinely different Top-C.
    data[4][:] = True
    both = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    assert identities(both, 1) == [(f, i) for f in (0, 1, 2, 5) for i in range(2)]


def test_selected_kvw_keep_gradients_but_discarded_observations_and_q_do_not():
    data = list(features(list(range(6)), grad=True))
    data[3] = torch.zeros_like(data[3], requires_grad=True)
    cache = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    loss = cache.observation.key.square().sum() + cache.observation.value.square().sum()
    loss = loss + cache.observation.weight.square().sum()
    loss.backward()
    for tensor in data[:3]:
        assert tensor.grad[:, :, 6:10].count_nonzero() == 0
        assert tensor.grad[:, :, :6].count_nonzero() > 0
    assert data[3].grad is None
    detached = detach_cache(cache)
    assert all(not x.requires_grad for x in (detached.observation.key, detached.observation.value,
                                             detached.observation.weight, detached.query))
    assert detach_cache(None) is None and storage_bytes(None) == 0
    expected = sum(x.numel()*x.element_size() for x in (cache.observation.key,
        cache.observation.value, cache.observation.weight, cache.observation.token_indices,
        cache.observation.frame_ids, cache.query))
    assert storage_bytes(cache) == expected


def test_source_partition_is_disjoint_with_prefix_precedence_and_no_history():
    cache = update(None, [0, 1, 2, 3, 4])
    groups = sources(cache, prefix_frames=1, recent_frames=1)
    assert len(groups) == 3
    combined = [(int(f), int(i)) for o in groups for f, i in zip(o.frame_ids[0], o.token_indices[0]) if f >= 0]
    assert len(set(combined)) == len(combined)
    assert sorted(combined) == identities(cache)
    prefix = update(None, [0])
    one, two, three = sources(prefix, prefix_frames=1, recent_frames=1)
    assert one.key.shape[-2] == 2 and two.key.shape[-2] == three.key.shape[-2] == 0
    data = list(features([0], batch=2))
    data[4][:] = False
    empty = update_cache(None, *data)
    assert empty.observation.key.shape == (2, 2, 0, 8)
    validate_cache(empty)
    assert all(x.key.shape[-2] == 0 for x in sources(empty))


def test_valid_write_excludes_padding_overlap_and_rejects_duplicate_or_past():
    cache = update(None, [0])
    data = list(features([0, 1]))
    data[4][:, :2] = False
    next_cache = update_cache(cache, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    assert identities(next_cache) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        update(cache, [0, 1])
    with pytest.raises(ValueError, match="causal"):
        update(next_cache, [0])
    data = list(features([2, 3]))
    data[5] = torch.tensor([[2, -1]])
    data[4][:, 2:] = False
    padded = update_cache(next_cache, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    assert identities(padded)[-2:] == [(2, 0), (2, 1)]


class CausalWanRotaryPosEmbed:
    def __init__(self):
        self.attention_head_dim = 8
        angles = torch.arange(64, dtype=torch.float64)[:, None] * torch.tensor([.5, .2, .07, .01])[None]
        self.freqs = torch.polar(torch.ones_like(angles), angles)


def test_temporal_realignment_is_key_only_absolute_delta_and_payload_immutable():
    data = list(features(list(range(10)), grad=True))
    data[3] = torch.zeros_like(data[3])
    cache = update_cache(None, *data, capacity_frames=4, prefix_frames=1, recent_frames=1)
    # Retained absolute frames 0 / 1,2 / 9 relocate to 6 / 7,8 / 9.
    assert identities(cache) == [(f, i) for f in (0, 1, 2, 9) for i in range(2)]
    digest = prefix_digest(cache, 1)
    rope = CausalWanRotaryPosEmbed()
    shifted = temporal_realign(cache, rope, torch.tensor([[10]]), prefix_frames=1, recent_frames=1)
    target = torch.tensor([[6, 6, 7, 7, 8, 8, 9, 9]])
    phase = rope.freqs.new_ones((1, 8, 4))
    phase[:, :, :2] = rope.freqs[target, :2]/rope.freqs[cache.observation.frame_ids, :2]
    expected = apply_complex_rope(cache.observation.key, phase[:, None])
    torch.testing.assert_close(shifted.observation.key, expected, atol=0, rtol=0)
    assert shifted.observation.value is cache.observation.value
    assert shifted.observation.weight is cache.observation.weight
    assert torch.equal(shifted.observation.frame_ids, cache.observation.frame_ids)
    assert prefix_digest(cache, 1) == digest
    assert shifted.observation.key.requires_grad
    with pytest.raises(ValueError, match="read view"):
        update(shifted, [10])
    with pytest.raises(ValueError, match="absolute keys"):
        temporal_realign(shifted, rope, torch.tensor([[10]]))
    with pytest.raises(ValueError, match="temporary"):
        prefix_digest(shifted)


def test_temporal_initial_overlap_does_not_invent_negative_time():
    cache = update(None, [0])
    shifted = temporal_realign(cache, CausalWanRotaryPosEmbed(), torch.tensor([[0, 1, 2, 3]]),
                              prefix_frames=1, recent_frames=1)
    torch.testing.assert_close(shifted.observation.key, cache.observation.key, atol=0, rtol=0)


@pytest.mark.parametrize("options", [dict(capacity_frames=2, prefix_frames=2, recent_frames=1),
    dict(capacity_frames=True), dict(prefix_frames=-1), dict(recent_frames=0), dict(selection="random")])
def test_invalid_options_fail(options):
    with pytest.raises(ValueError):
        update_cache(None, *features([0]), **options)


def test_invalid_shapes_coordinates_and_payload_fail():
    data = list(features([0, 1]))
    data[-1] = (2, 2, 2)
    with pytest.raises(ValueError, match="grid"):
        update_cache(None, *data)
    data = list(features([0]))
    data[2][0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative weights"):
        update_cache(None, *data)
    cache = update(None, [0])
    broken = replace(cache, observation=replace(cache.observation, frame_ids=torch.tensor([[-2, 0]])))
    with pytest.raises(ValueError, match="invalid padding"):
        validate_cache(broken)
    with pytest.raises(ValueError, match="actual Causal"):
        temporal_realign(cache, None, torch.tensor([[1]]))
    with pytest.raises(ValueError, match="current frame"):
        temporal_realign(cache, CausalWanRotaryPosEmbed(), torch.tensor([[-1]]))
