"""Proximal memory uses the existing transaction, camera and training lifecycle."""
from dataclasses import replace
import pytest
import torch
from test_full_training import model, update, EulerOracle
from test_alignment_detail import cached_sana
from worldttn import training
from worldttn.anchor import configure_train_scope
from worldttn.core import TTNConfig
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint


def proximal_model(cached_sana):
    m = model(cached_sana, psi=False)
    cfg = replace(m.ttn_system.config, sink_gain=0., replay_strength=0.,
                  memory_update="proximal", memory_capacity_frames=6,
                  memory_prefix_frames=1, memory_recent_frames=2)
    m.ttn_system.config = cfg
    for index in (3, 7, 11, 15, 19): m.blocks[index].attn.config = cfg
    return configure_train_scope(m, "dit")


def test_proximal_config_rejects_competing_updates():
    with pytest.raises(ValueError, match="proximal"):
        TTNConfig(stage="C", camera_attention="sana", memory_update="proximal", local_update=True)
    with pytest.raises(ValueError, match="proximal"):
        TTNConfig(stage="C", camera_attention="sana", memory_update="proximal", sink_gain=.1)


def test_proximal_training_bounded_cache_and_future_credit(monkeypatch, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    m = proximal_model(cached_sana)
    original = m.forward
    def forward(*args, **kwargs):
        output = original(*args, **kwargs)
        for state, _, _ in kwargs["ttn_chunk_context"].candidates.values():
            if state.requires_grad: state.retain_grad()
        return output
    m.forward = forward
    result = update(m)
    runtime = result["runtime"]
    assert runtime.commit_count == 9 and runtime.committed_frame_ids == [set(range(25))]
    assert len(runtime.memory_caches) == 5
    for cache in runtime.memory_caches:
        assert cache.observation.key.shape[2] <= 6
        assert not cache.observation.key.requires_grad
    assert m.saved_features[5].grad is not None and m.saved_features[5].grad.norm() > 0
    assert m.saved_features[4].grad is None
    assert m.blocks[3].attn.beta_proj.weight.grad.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in m.ttn_system.parameters())
    assert runtime.transition_fast.count_nonzero() == 0
    for chunk, row in enumerate(result["chunks"], 1):
        for anchor in row["anchors"]:
            assert anchor["proximal"]["implementation"] == "live_proximal_s"
            assert ("state_spectrum" in anchor) == (chunk in (1, 4, 5, 8))
            if "state_spectrum" in anchor:
                assert anchor["inner_update_applied"] is True
                assert anchor["state_spectrum"]["committed"]["sigma_max"][0][0] >= 0
    assert runtime.verify_memory_prefix()


def test_proximal_checkpoint_identity(monkeypatch, tmp_path, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    a = proximal_model(cached_sana)
    a.base_load_report = {"sha256": "test"}
    update(a)
    path = tmp_path / "last.pt"
    save_checkpoint(path, a, None, 1)
    b = proximal_model(cached_sana); b.base_load_report = a.base_load_report
    load_checkpoint(path, b)
    for x, y in zip(a.parameters(), b.parameters()): torch.testing.assert_close(x, y, rtol=0, atol=0)
    old = model(cached_sana, psi=False); old.base_load_report = a.base_load_report
    with pytest.raises(ValueError, match="identity mismatch"): load_checkpoint(path, old)


@pytest.mark.parametrize("detach_history", [False, True])
def test_future_credit_isolated_to_selected_cache_and_cut_at_boundary(monkeypatch, cached_sana, detach_history):
    from worldttn.session import TTNSession
    from worldttn.memory_cache import detach_cache
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    m = proximal_model(cached_sana)
    retained = []
    def capture(module, args, output):
        ctx = args[2]
        if ctx.clean_mode and not ctx.prefill_mode:
            o = ctx.memory_candidates[4].observation
            tensors = o.key, o.value, o.weight
            for t in tensors: t.retain_grad()
            retained.append(tensors)
    m.blocks[19].register_forward_hook(capture)
    original = TTNSession.clean_forward
    def isolate(session, *args, **kwargs):
        result = original(session, *args, **kwargs)
        session.runtime.world_state = session.runtime.world_state.detach()
        if detach_history:
            session.runtime.memory_caches = tuple(detach_cache(c) for c in session.runtime.memory_caches)
        return result
    monkeypatch.setattr(TTNSession, "clean_forward", isolate)
    update(m)
    for t in retained[4]:
        assert t.grad is None if detach_history else t.grad is not None and t.grad.norm() > 0
    for i in (3, 7): assert all(t.grad is None for t in retained[i])


@pytest.mark.parametrize("selection", ["participative", "fifo"])
def test_noisy_calls_and_failed_clean_commit_preserve_runtime(cached_sana, selection):
    from test_sink import begin
    from worldttn.runtime import TTNRuntimeState
    from tools.ttn_compare_resume import identical
    from worldttn.history import _clone_tree
    m = proximal_model(cached_sana)
    cfg = replace(m.ttn_system.config, memory_selection=selection)
    m.ttn_system.config = cfg
    for i in (3, 7, 11, 15, 19): m.blocks[i].attn.config = cfg
    runtime = TTNRuntimeState.create(cfg, 1, "cpu")
    def run(ctx):
        for i in (3, 7, 11, 15, 19):
            m.blocks[i].attn(torch.randn(1, ctx.frame_ids.shape[1], 16),
                HW=(ctx.frame_ids.shape[1], 1, 1), ttn_chunk_context=ctx, kv_cache=[None]*10)
    with torch.no_grad():
        pre = begin(runtime, m.ttn_system, [0], prefill=True).for_clean()
        run(pre); runtime.prefill(pre)
        old_state = runtime.world_state.clone()
        old_cache = _clone_tree(runtime.memory_caches)
        ctx = begin(runtime, m.ttn_system, [0, 1, 2, 3])
        for _ in range(3): run(ctx)
        assert ctx.memory_candidates == {} and ctx.candidates == {}
        clean = ctx.for_clean(); run(clean)
        clean.memory_stats.pop(4)
        with pytest.raises(RuntimeError, match="diagnostics"): runtime.commit_chunk(clean)
        assert runtime.commit_count == 1 and runtime.committed_frame_ids == [{0}]
        torch.testing.assert_close(runtime.world_state, old_state, rtol=0, atol=0)
        for a, b in zip(runtime.memory_caches, old_cache):
            identical(vars(a.observation), vars(b.observation), "failed transaction cache")
        assert runtime.verify_memory_prefix()


def test_quantized_output_effect_is_measured(cached_sana):
    from test_anchor import context
    m = proximal_model(cached_sana)
    _, ctx = context(m.ttn_system.config, f=4)
    ctx.collect_memory_stats = True
    reports = {}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        m.blocks[3].attn(torch.randn(1, 4, 16).bfloat16(), HW=(4, 1, 1), ttn_chunk_context=ctx,
            ttn_diagnostic=lambda name, value: reports.update({name: value}))
    effect = reports["proximal_update"]["output_effect"]
    assert effect["dtype"] == "torch.bfloat16"
    assert effect["delta_norm"] > 0 and effect["relative_delta"] > 0
    assert 0 < effect["changed_fraction"] <= 1


def test_proximal_noise_telemetry_keeps_exact_sigma_and_sampled_index(cached_sana):
    from test_anchor import context
    from worldttn.session import record_noise
    m = proximal_model(cached_sana)
    _, ctx = context(m.ttn_system.config, f=4)
    timestep = torch.full((1, 1, 4), 731.)
    sampled = torch.full((1, 1, 4), 500)
    sigma = torch.full((1, 1, 4), .73137)
    record_noise(ctx, timestep, sigma, sampled)
    row = ctx.memory_trajectory[-1]
    assert row["noise_sigma_source"] == "scheduler"
    assert row["noise_sigma"] == sigma.tolist() and row["sampled_timestep"] == sampled.tolist()
    assert row["noise_timestep"] == timestep.tolist()


@pytest.mark.parametrize("source,ttn_value,native_value", [("ttn-gt", 2., 1.), ("native-gt", 1., 2.)])
@torch.no_grad()
def test_hybrid_history_selects_retained_observations_with_ttn_state(cached_sana, source, ttn_value, native_value):
    from test_sink import begin
    from worldttn.runtime import TTNRuntimeState
    from worldttn.history import clean_history_transaction
    from worldttn.memory_cache import update_cache
    m = proximal_model(cached_sana)
    runtime = TTNRuntimeState.create(m.ttn_system.config, 1, "cpu")
    ctx = begin(runtime, m.ttn_system, [0, 1, 2, 3])
    generated = torch.ones(1, 16, 4, 1, 1)
    incoming = [[torch.zeros(1)] + [None]*9]
    def forward(x, clean, native):
        value = x.mean()
        for i in range(5):
            k = torch.ones(1, 2, 4, 8)*value
            cache = update_cache(None, k, k, torch.ones(1, 2, 4), k,
                torch.ones(1, 4, dtype=torch.bool), ctx.frame_ids, (4, 1, 1),
                capacity_frames=6, prefix_frames=1, recent_frames=2)
            clean.memory_candidates[i] = cache
            clean.memory_stats[i] = {}
            clean.stage(i, clean.predicted[:, i]+value, torch.zeros_like(clean.psi[:, i]), {})
        native[0][0].fill_(value)
        return x, native
    _, cache = clean_history_transaction(forward, generated, incoming, source=source,
        reference=2*generated, context=ctx, runtime=runtime)
    assert runtime.commit_count == 1 and cache[0][0].item() == native_value
    assert incoming[0][0].item() == 0 and not ctx.memory_candidates
    assert all((c.observation.value == ttn_value).all() for c in runtime.memory_caches)
    assert (runtime.world_state == ttn_value).all()
