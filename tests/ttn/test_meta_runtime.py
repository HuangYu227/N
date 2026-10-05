import copy
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F
from test_anchor import ProjectionContract
from worldttn import core
from worldttn.anchor import TTNAnchor, is_ttn_new_parameter
from worldttn.runtime import TTNSystem, TTNRuntimeState
from worldttn.distributed import _expose_clean_state, _restore_block_output


def episode(local=True, persistent=True, batch=1):
    torch.manual_seed(23)
    cfg = core.TTNConfig(heads=2, head_dim=8, generators=3, stage="C",
                         local_update=local, persistent_meta=persistent)
    system = TTNSystem(cfg)
    runtime = TTNRuntimeState.create(cfg, batch, "cpu")
    runtime.world_state = torch.randn_like(runtime.world_state) * .2
    runtime.transition_fast = torch.randn_like(runtime.transition_fast) * .1
    anchors = [TTNAnchor(ProjectionContract(), i, cfg) for i in range(5)]
    return system, runtime, anchors


def begin(system, runtime, ids=(10, 11), valid=None):
    b = runtime.world_state.shape[0]
    frames = torch.tensor(ids).expand(b, -1)
    poses = torch.eye(4).expand(b, len(ids), 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(b, len(ids), 4)
    if valid is None: valid = torch.ones_like(frames, dtype=torch.bool)
    return runtime.begin_chunk(system, poses, intr, frames, valid, 100, 100)


def test_support_query_uses_absolute_grid_and_excludes_committed_or_invalid_frames():
    system, runtime, _ = episode(batch=2)
    runtime.committed_frame_ids = [{10}, {10}]
    ctx = begin(system, runtime, valid=torch.tensor([[True, True], [True, False]]))
    support, query = ctx.support_query_masks(8, (2, 2, 2))
    assert support.tolist() == [[False]*4 + [False, True, True, False], [False]*8]
    assert query.tolist() == [[False]*4 + [True, False, False, True], [False]*8]
    with pytest.raises(ValueError, match="grid"):
        ctx.support_query_masks(8, None)
    with pytest.raises(ValueError, match="grid"):
        ctx.support_query_masks(8, (1, 2, 4))


def test_local_is_used_immediately_from_original_state_and_never_committed():
    system, runtime, anchors = episode(batch=2)
    ctx = begin(system, runtime)
    x = torch.randn(2, 8, 16)
    a = anchors[0]
    before = (runtime.world_state.clone(), runtime.transition_fast.clone(),
              copy.deepcopy(runtime.committed_frame_ids), runtime.commit_count)
    captured = {}
    out, cache = a(x, HW=(2, 2, 2), ttn_chunk_context=ctx, kv_cache=[None]*10,
                   ttn_diagnostic=lambda name, value: captured.update({name: value}))
    q, k, v = a.visual_features(x, None)
    beta = a.beta_proj(x).sigmoid().transpose(1, 2)
    write = ctx.token_masks(8)[1]
    support = torch.tensor([[True, False, False, True, False, True, True, False]]).expand(2, -1)
    w = core.write_weights(beta, write, 1e-6)
    zeta = torch.zeros_like(ctx.psi[:, 0], requires_grad=True)
    u, vg = system.generators.u[0], system.generators.v[0]
    coeff0 = ctx.cbase[:, 0] + ctx.psi[:, 0].tanh()
    baseline = ctx.previous[:, 0] @ core.cayley_dense(u, vg, coeff0 + zeta.tanh())
    sw = torch.where(support[:, None], w, 0.)
    loss = (sw * (k @ baseline - v).square().sum(-1)).sum(-1) / (
        1e-6 + (sw * v.square().sum(-1)).sum(-1))
    g = torch.autograd.grad(loss.sum(), zeta)[0]
    clipped = g * (1 / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp_max(1)
    local = -F.softplus(system.local_eta_logits[0]) * clipped
    effective = ctx.previous[:, 0] @ core.cayley_dense(u, vg, coeff0 + local.tanh())
    state, _ = core.correct(effective, k, v, beta, write)
    expected = (q @ state).transpose(1, 2).reshape_as(x)
    torch.testing.assert_close(captured["visual_raw"], expected, atol=1e-6, rtol=0)
    old_state, _ = core.correct(ctx.predicted[:, 0], k, v, beta, write)
    assert not torch.allclose(expected, (q @ old_state).transpose(1, 2).reshape_as(x), atol=1e-8, rtol=0)
    assert not ctx.candidates and cache[:9] == [None]*9
    torch.testing.assert_close(runtime.world_state, before[0], atol=0, rtol=0)
    torch.testing.assert_close(runtime.transition_fast, before[1], atol=0, rtol=0)
    assert runtime.committed_frame_ids == before[2] and runtime.commit_count == before[3]
    grad = torch.autograd.grad(out.square().sum(), system.local_eta_logits)[0]
    assert grad[0].abs() > 0 and grad[1:].count_nonzero() == 0
    # A new call consumes new noisy features but does not accumulate a local state.
    again, _ = a(x, HW=(2, 2, 2), ttn_chunk_context=ctx, kv_cache=[None]*10)
    torch.testing.assert_close(out, again, atol=0, rtol=0)


def test_clean_disables_local_and_preserves_live_persistent_gradient():
    system, runtime, anchors = episode()
    ctx = begin(system, runtime).for_clean()
    x = torch.randn(1, 8, 16)
    captured = {}
    anchors[0](x, HW=(2, 2, 2), ttn_chunk_context=ctx,
               ttn_diagnostic=lambda name, value: captured.update({name: value}))
    q, k, v = anchors[0].visual_features(x, None)
    beta = anchors[0].beta_proj(x).sigmoid().transpose(1, 2)
    expected, _ = core.correct(ctx.predicted[:, 0], k, v, beta, ctx.token_masks(8)[1])
    torch.testing.assert_close(ctx.candidates[0][0], expected, atol=0, rtol=0)
    assert ctx.candidates[0][1].requires_grad
    output = (torch.randn(1, 8, 16), [None]*10)
    exposed = _expose_clean_state(SimpleNamespace(attn=anchors[0]), (), {"ttn_chunk_context": ctx}, output)
    assert any(item is ctx.candidates[0][1] for item in exposed), "FSDP must see future psi's live gP"
    assert _restore_block_output(None, (), exposed) is output


@pytest.mark.parametrize("persistent,detach", [(True, False), (False, False), (True, True)])
def test_future_credit_flows_only_through_live_persistent_update_inside_window(persistent, detach):
    system, runtime, anchors = episode(local=False, persistent=persistent)
    ctx = begin(system, runtime).for_clean()
    for anchor in anchors:
        anchor(torch.randn(1, 8, 16), HW=(2, 2, 2), ttn_chunk_context=ctx)
        s, g, stats = ctx.candidates[anchor.index]
        # Remove ordinary S credit to isolate the Persistent meta-update path.
        ctx.candidates[anchor.index] = (s.detach(), g, stats)
    runtime.commit_chunk(ctx)
    assert runtime.transition_fast.requires_grad == persistent
    if detach: runtime.detach()
    future = begin(system, runtime, (12, 13))
    probe = torch.randn_like(future.predicted[:, 0])
    gradient = torch.autograd.grad((future.predicted[:, 0]*probe).sum(),
                                  anchors[0].qkv.weight, allow_unused=True)[0]
    if persistent and not detach: assert gradient is not None and gradient.norm() > 0
    else: assert gradient is None


def test_eta_initialization_keeps_rng_and_parameter_origin():
    torch.manual_seed(12)
    old = TTNSystem(core.TTNConfig(heads=2, head_dim=8, generators=3, stage="C"))
    before = torch.get_rng_state()
    torch.manual_seed(12)
    new = TTNSystem(core.TTNConfig(heads=2, head_dim=8, generators=3, stage="C", local_update=True))
    assert torch.equal(before, torch.get_rng_state())
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
    torch.testing.assert_close(F.softplus(new.local_eta_logits), torch.full((5,), .01))
    assert is_ttn_new_parameter("ttn_system.local_eta_logits")


@pytest.mark.parametrize("mode,local,persistent", [("no-local", False, True), ("no-persistent", True, False),
                                                    ("no-ttt", False, False)])
def test_contribution_controls_disable_only_selected_psi(mode, local, persistent):
    system, runtime, anchors = episode()
    runtime.ablation = mode
    ctx = begin(system, runtime)
    coeff = ctx.cbase + (ctx.psi.tanh() if persistent else 0)
    torch.testing.assert_close(ctx.predicted, core.CayleyFactors(system.generators.u, system.generators.v, coeff).right(ctx.previous))
    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        for anchor in anchors: anchor(x, HW=(2, 2, 2), ttn_chunk_context=ctx)
        clean = ctx.for_clean()
        for anchor in anchors: anchor(x, HW=(2, 2, 2), ttn_chunk_context=clean)
        runtime.commit_chunk(clean)
    assert all(a["local"]["enabled"] == local for a in runtime.last_stats["anchors"])
    assert (runtime.transition_fast.norm() > 0).item() == persistent
    assert all(a["persistent"]["update_enabled"] == persistent for a in runtime.last_stats["anchors"])


def test_legacy_runtime_rejects_new_contribution_modes():
    cfg = core.TTNConfig(heads=2, head_dim=8, generators=3, stage="C")
    with pytest.raises(ValueError, match="Meta-TTT"):
        TTNRuntimeState.create(cfg, 1, "cpu", ablation="no-local")


@pytest.mark.parametrize("frame", [10, 11])
def test_empty_support_or_query_and_zero_state_are_explicit_finite_results(frame):
    system, runtime, anchors = episode()
    runtime.world_state.zero_()
    ctx = begin(system, runtime, (frame,))
    before = runtime.transition_fast.clone()
    out = anchors[0](torch.randn(1, 1, 16), HW=(1, 1, 1), ttn_chunk_context=ctx)
    stats = ctx.local_stats[0]
    empty = "query" if frame % 2 == 0 else "support"
    assert stats[empty]["loss_before"] == [[None, None]]
    assert stats["per_head"]["psi_norm"] == [[0., 0.]]
    assert torch.isfinite(out).all() and torch.equal(runtime.transition_fast, before)
