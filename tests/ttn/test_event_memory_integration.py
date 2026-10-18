"""IRMA integration guards on real TTN anchors, including CFG's two branches."""
import copy
from dataclasses import replace

import pytest
import torch

from test_anchor import ProjectionContract
from worldttn.anchor import TTNAnchor
from worldttn.core import ANCHORS, TTNConfig, correct
from worldttn.event_memory import EventMemoryOptions, pooled_descriptor
from worldttn.performance import ExecutionOptions
from worldttn.runtime import TTNRuntimeState, TTNSystem


def _episode(options=None, *, batch=2, heads=20, dim=8, state_ttt=True):
    torch.manual_seed(61)
    config = TTNConfig(heads=heads, head_dim=dim, generators=3, stage="C",
                       state_ttt_rank=4 if state_ttt else 0, state_ttt_gain=.5)
    system = TTNSystem(config)
    anchors = [TTNAnchor(ProjectionContract(heads, dim), index, config) for index in range(5)]
    runtime = TTNRuntimeState.create(config, batch, "cpu", memory_options=options)
    return config, system, anchors, runtime


def _begin(system, runtime, ids, *, prefill=False):
    batch = runtime.world_state.shape[0]
    frames = torch.tensor(ids).expand(batch, -1)
    poses = torch.eye(4).expand(batch, len(ids), 4, 4)
    intrinsics = torch.tensor([100., 100., 50., 50.]).expand(batch, len(ids), 4)
    return runtime.begin_chunk(system, poses, intrinsics, frames,
                               torch.ones_like(frames, dtype=torch.bool), 100, 100, prefill=prefill)


def _forward(anchors, context, features):
    return [anchor(features, HW=(context.frame_ids.shape[-1], 2, 2),
                   ttn_chunk_context=context) for anchor in anchors]


def _prefill(system, anchors, runtime):
    context = _begin(system, runtime, [0], prefill=True).for_clean()
    width = runtime.config.heads * runtime.config.head_dim
    features = torch.randn(runtime.world_state.shape[0], 4, width)
    # Different CFG branches must create different descriptors and payloads.
    features[1:] = features[1:] * .7 + 2.
    _forward(anchors, context, features)
    runtime.prefill(context)
    return context


@pytest.mark.parametrize("dim", [8, 112])
def test_real_cfg_descriptor_shape_and_batch_isolation(dim):
    generator = torch.Generator().manual_seed(63)
    keys = torch.randn(2, 20, 7, dim, generator=generator)
    values = torch.randn(2, 20, 7, dim, generator=generator)
    values[1] += 3.
    write = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    together = pooled_descriptor(keys, values, write)
    separate = torch.cat([pooled_descriptor(keys[index:index + 1], values[index:index + 1],
                                           write[index:index + 1]) for index in range(2)])
    assert together.ndim == 2 and together.shape[0] == 2
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
    assert not torch.equal(together[0], together[1])
    assert torch.isfinite(together).all()


@pytest.mark.parametrize("options", [EventMemoryOptions(),
                                    EventMemoryOptions(mode="state-replay", gain=0.)])
@torch.no_grad()
def test_off_and_zero_gain_preserve_original_outputs_states_and_rng(options):
    _, system, anchors, plain = _episode()
    rng = torch.get_rng_state().clone()
    changed = TTNRuntimeState.create(plain.config, 2, "cpu", memory_options=options)
    assert torch.equal(torch.get_rng_state(), rng)
    assert changed.event_memory is None
    features = torch.randn(2, 8, 160)
    baseline = _begin(system, plain, [0, 1])
    intervention = _begin(system, changed, [0, 1])
    for left, right in zip(_forward(anchors, baseline, features), _forward(anchors, intervention, features)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not intervention.candidates and not intervention.memory_pending
    for runtime, context in ((plain, baseline), (changed, intervention)):
        clean = context.for_clean()
        _forward(anchors, clean, features)
        runtime.commit_chunk(clean)
    torch.testing.assert_close(plain.world_state, changed.world_state, rtol=0, atol=0)
    torch.testing.assert_close(plain.transition_fast, changed.transition_fast, rtol=0, atol=0)


@torch.no_grad()
def test_noisy_reads_use_physical_anchors_and_never_commit_local_state():
    options = EventMemoryOptions(mode="state-replay", start_chunk=1)
    _, system, anchors, runtime = _episode(options)
    prefill = _prefill(system, anchors, runtime)
    assert set(prefill.memory_pending) == {11, 15, 19}
    assert len(runtime.event_memory) == 1
    before_s = runtime.world_state.clone()
    before_psi = runtime.transition_fast.clone()
    before_bank = runtime.event_memory
    context = _begin(system, runtime, [1, 2])
    context.memory_sigma = .25
    context.memory_call = 1
    baseline = copy.copy(context)
    baseline.memory_bank = None
    baseline.memory_selections = {}
    baseline.memory_pending = {}
    baseline.memory_stats = []
    features = torch.randn(2, 8, 160)
    original = _forward(anchors, baseline, features)
    actual = _forward(anchors, context, features)
    assert set(context.memory_selections) == {11, 15, 19}
    for index, physical in enumerate(ANCHORS):
        if physical not in options.anchors:
            torch.testing.assert_close(original[index], actual[index], rtol=0, atol=0)
    assert any(not torch.equal(original[ANCHORS.index(block)], actual[ANCHORS.index(block)])
               for block in options.anchors)
    assert context.memory_stats and not context.memory_pending and not context.candidates
    selections = dict(context.memory_selections)
    _forward(anchors, context, features + .5)
    for block in options.anchors:
        assert context.memory_selections[block] is selections[block]
    assert runtime.event_memory is before_bank and len(before_bank) == 1
    torch.testing.assert_close(runtime.world_state, before_s, rtol=0, atol=0)
    torch.testing.assert_close(runtime.transition_fast, before_psi, rtol=0, atol=0)
    assert runtime.commit_count == 1


@torch.no_grad()
def test_clean_memory_payload_is_invisible_until_commit_and_candidates_are_unchanged():
    options = EventMemoryOptions(mode="state-replay", start_chunk=1)
    _, system, anchors, runtime = _episode(options)
    _prefill(system, anchors, runtime)
    context = _begin(system, runtime, [1, 2])
    clean = context.for_clean()
    baseline = copy.copy(clean)
    baseline.memory_bank = None
    baseline.memory_selections = {}
    baseline.memory_pending = {}
    baseline.memory_stats = []
    baseline.candidates = {}
    features = torch.randn(2, 8, 160)
    _forward(anchors, clean, features)
    _forward(anchors, baseline, features)
    assert len(runtime.event_memory) == 1
    assert context.memory_pending == {}
    assert set(clean.memory_pending) == {11, 15, 19}
    for index in range(5):
        torch.testing.assert_close(clean.candidates[index][0], baseline.candidates[index][0], rtol=0, atol=0)
        torch.testing.assert_close(clean.candidates[index][1], baseline.candidates[index][1], rtol=0, atol=0)
    runtime.commit_chunk(clean)
    assert len(runtime.event_memory) == 2 and runtime.commit_count == 2
    future = _begin(system, runtime, [3, 4])
    assert len(future.memory_bank) == 2
    with pytest.raises(RuntimeError):
        runtime.commit_chunk(clean)


@pytest.mark.parametrize("failure", ["missing", "nan", "stale"])
@torch.no_grad()
def test_failed_clean_commit_preserves_archive_and_recurrent_state(failure):
    _, system, anchors, runtime = _episode(EventMemoryOptions(mode="state-replay"))
    _prefill(system, anchors, runtime)
    clean = _begin(system, runtime, [1, 2]).for_clean()
    _forward(anchors, clean, torch.randn(2, 8, 160))
    before = (runtime.event_memory, runtime.world_state.clone(), runtime.transition_fast.clone(),
              runtime.revision, runtime.commit_count, copy.deepcopy(runtime.committed_frame_ids))
    if failure == "missing":
        clean.candidates.pop(4)
    elif failure == "nan":
        state, gradient, stats = clean.candidates[4]
        clean.candidates[4] = (state * float("nan"), gradient, stats)
    else:
        clean.revision -= 1
    with pytest.raises((ValueError, RuntimeError)):
        runtime.commit_chunk(clean)
    assert runtime.event_memory is before[0] and len(runtime.event_memory) == 1
    torch.testing.assert_close(runtime.world_state, before[1], rtol=0, atol=0)
    torch.testing.assert_close(runtime.transition_fast, before[2], rtol=0, atol=0)
    assert (runtime.revision, runtime.commit_count, runtime.committed_frame_ids) == before[3:]


def test_active_memory_refuses_training_before_state_creation():
    config = TTNConfig(heads=20, head_dim=8, stage="C", state_ttt_rank=4, state_ttt_gain=.5)
    with pytest.raises(ValueError, match="inference|grad"):
        TTNRuntimeState.create(config, 2, "cpu", memory_options=EventMemoryOptions(mode="state-replay"))


def test_rollout_rejects_ground_truth_history_before_model_access():
    from worldttn.cli import rollout
    with pytest.raises(ValueError, match="ground-truth"):
        rollout(None, None, {}, history_reference=object(),
                memory_options=EventMemoryOptions(mode="state-replay"))


@torch.no_grad()
def test_session_output_failure_discards_pending_event_and_retry_commits_once():
    from test_training import TinyWorldModel, inputs
    from worldttn.session import TTNSession
    torch.manual_seed(67)
    model = TinyWorldModel(state_ttt_rank=4, state_ttt_gain=.5)
    clean, _, _, camera = inputs()
    session = TTNSession(model, camera, 100, 100,
                         memory_options=EventMemoryOptions(mode="state-replay", start_chunk=1))
    session.reset(1)
    condition = torch.zeros(1, 1, 2, 8)
    session.prefill(clean[:, :, :1], condition)
    runtime = session.runtime
    before = (runtime.event_memory, runtime.world_state.clone(), runtime.transition_fast.clone(),
              runtime.commit_count, runtime.revision)
    context = session.begin_chunk(0, 4)
    cache = [[None] * 10 for _ in model.blocks]
    forward = model.forward
    def invalid_output(*args, **kwargs):
        result, updated = forward(*args, **kwargs)
        assert set(kwargs["ttn_chunk_context"].memory_pending) == {11, 15, 19}
        return result * float("nan"), updated
    model.forward = invalid_output
    with pytest.raises(FloatingPointError, match="nonfinite clean"):
        session.clean_forward(clean[:, :, :4], condition, context, cache, 0, 4)
    assert runtime.event_memory is before[0] and len(runtime.event_memory) == 1
    torch.testing.assert_close(runtime.world_state, before[1], rtol=0, atol=0)
    torch.testing.assert_close(runtime.transition_fast, before[2], rtol=0, atol=0)
    assert (runtime.commit_count, runtime.revision) == before[3:]
    assert not context.memory_pending and not context.candidates
    model.forward = forward
    session.clean_forward(clean[:, :, :4], condition, context, cache, 0, 4)
    assert runtime.commit_count == before[3] + 1 and len(runtime.event_memory) == 2


@torch.no_grad()
def test_session_commit_rejection_keeps_camera_lengths_and_retry_appends_once(monkeypatch):
    from test_training import TinyWorldModel, inputs
    from worldttn.session import TTNSession
    torch.manual_seed(71)
    model = TinyWorldModel(state_ttt_rank=4, state_ttt_gain=.5)
    config = replace(model.ttn_system.config, camera_attention="sana")
    model.ttn_system.config = config
    for block in ANCHORS:
        model.blocks[block].attn.config = config
    # TinyWorldModel does not run camera attention; add distinct current K/V
    # payloads so the real session cache preparation and truncation run.
    forward = model.forward
    def camera_payloads(*args, **kwargs):
        output, caches = forward(*args, **kwargs)
        tokens = args[0].shape[2]
        for block in ANCHORS:
            caches[block][2] = torch.full((1, 2, tokens, 8), float(block))
            caches[block][3] = torch.full((1, 2, tokens, 8), float(block + 1))
        return output, caches
    model.forward = camera_payloads
    clean, _, _, camera = inputs()
    session = TTNSession(model, camera, 100, 100,
                         memory_options=EventMemoryOptions(mode="state-replay"))
    session.reset(1)
    condition = torch.zeros(1, 1, 2, 8)
    session.prefill(clean[:, :, :1], condition)
    session.camera_cache_lengths = {block: [2] for block in ANCHORS}
    runtime = session.runtime
    context = session.begin_chunk(0, 4)
    cache = [[None] * 10 for _ in model.blocks]
    for block in ANCHORS:
        cache[block][2] = torch.ones(1, 2, 2, 8)
        cache[block][3] = torch.ones(1, 2, 2, 8) * 2
    before_cache = copy.deepcopy(cache)
    before_lengths = copy.deepcopy(session.camera_cache_lengths)
    before = (runtime.event_memory, runtime.world_state.clone(), runtime.transition_fast.clone(),
              runtime.commit_count, runtime.revision)
    commit, seen_lengths = runtime.commit_chunk, []
    def reject_once(candidate):
        seen_lengths.append(copy.deepcopy(session.camera_cache_lengths))
        assert set(candidate.memory_pending) == {11, 15, 19}
        if len(seen_lengths) == 1:
            raise ValueError("clean commit rejected after cache preparation")
        return commit(candidate)
    monkeypatch.setattr(runtime, "commit_chunk", reject_once)
    with pytest.raises(ValueError, match="commit rejected"):
        session.clean_forward(clean[:, :, :4], condition, context, cache, 0, 4, cached_chunks=2)
    assert session.camera_cache_lengths == before_lengths
    assert runtime.event_memory is before[0]
    torch.testing.assert_close(runtime.world_state, before[1], rtol=0, atol=0)
    torch.testing.assert_close(runtime.transition_fast, before[2], rtol=0, atol=0)
    assert (runtime.commit_count, runtime.revision) == before[3:]
    _, updated = session.clean_forward(clean[:, :, :4], condition, context, cache, 0, 4, cached_chunks=2)
    assert seen_lengths == [before_lengths, before_lengths]
    assert session.camera_cache_lengths == {block: [2, 4] for block in ANCHORS}
    assert runtime.commit_count == before[3] + 1 and len(runtime.event_memory) == 2
    for block in ANCHORS:
        assert updated[block][2].shape[2] == updated[block][3].shape[2] == 6
        for slot in (2, 3):
            torch.testing.assert_close(cache[block][slot], before_cache[block][slot], rtol=0, atol=0)


@torch.no_grad()
def test_off_memory_preserves_compiled_read_only_branch(monkeypatch):
    from worldttn import compiled
    _, system, anchors, runtime = _episode(state_ttt=False)
    anchor = anchors[0]
    anchor.ttn_execution = ExecutionOptions("compiled", "reference")
    context = _begin(system, runtime, [0, 1])
    calls = []
    def read_only(q, k, v, beta, predicted, write, alpha, eps):
        calls.append(1)
        return q @ correct(predicted, k, v, beta, write, alpha, eps)[0]
    monkeypatch.setattr(compiled, "read_only", read_only)
    x = torch.randn(2, 8, 160)
    actual = anchor(x, HW=(2, 2, 2), ttn_chunk_context=context)
    anchor.ttn_execution = ExecutionOptions()
    expected = anchor(x, HW=(2, 2, 2), ttn_chunk_context=context)
    assert calls == [1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert runtime.event_memory is None and not context.candidates
