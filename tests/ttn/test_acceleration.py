"""Direct operator/gradient equivalence; no SANA or CUDA dependency."""
import copy
import pytest
import torch
from worldttn import core


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("d,m", [(8, 3), (112, 16)])
def test_reused_and_projected_gradients_match_dense_autograd(dtype, empty, d, m):
    assert hasattr(core, "correct_with_aux"), "Correct auxiliary reuse is not implemented"
    torch.manual_seed(3407)
    b, h, n = 2, 2, 11
    u, v = [torch.nn.functional.normalize(torch.randn(h, m, d, dtype=dtype), dim=-1) for _ in range(2)]
    psi = (torch.randn(b, h, m, dtype=dtype) * .2).requires_grad_()
    cbase = torch.randn_like(psi) * .15
    previous = torch.randn(b, h, d, d, dtype=dtype)
    # Non-contiguous, storage-offset inputs exercise the production layout contract.
    k, value = [torch.randn(b, h, n + 2, d * 2, dtype=dtype)[:, :, 1:-1, ::2] for _ in range(2)]
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[1, 7:] = False
    if empty: mask.zero_()
    beta = torch.randn(b, h, n, dtype=dtype).sigmoid()
    factors = core.CayleyFactors(u, v, cbase + .7 * psi.tanh())
    predicted = factors.right(previous)
    aux = core.correct_with_aux(predicted, k, value, beta, mask)
    reference_state, reference_w = core.correct(predicted, k, value, beta, mask)
    torch.testing.assert_close(aux.candidate, reference_state, rtol=0, atol=0)
    torch.testing.assert_close(aux.w, reference_w, rtol=0, atol=0)
    snapshot = core.DetachedCayleySnapshot.from_live(factors, psi, cbase, owner=(11, 2, 3))
    count = mask.sum(-1).clamp_min(1)[:, None]
    expected = torch.autograd.grad(core.innovation_loss(previous @ core.cayley_dense(u, v, cbase + .7 * psi.tanh()),
                                 k, value, aux.w.detach(), mask).sum(), psi)[0]
    old = core.analytic_psi_gradient(previous, k, value, aux.w, mask, u, v, cbase, psi, .7)
    dense = core.analytic_psi_gradient_dense_from_aux(previous, aux.kt_weighted_residual, snapshot, count, .7)
    projected = core.analytic_psi_gradient_projected(previous, aux.kt_weighted_residual, snapshot, count, .7)
    for actual in (old, dense, projected):
        torch.testing.assert_close(actual, expected, atol=1e-10 if dtype == torch.float64 else 1e-6,
                                   rtol=0 if dtype == torch.float64 else 1e-4)
        assert not actual.requires_grad
    if empty:
        assert torch.count_nonzero(projected) == 0
        assert torch.equal(aux.candidate, predicted)


def test_correct_aux_preserves_outer_gradients():
    assert hasattr(core, "correct_with_aux"), "Correct auxiliary reuse is not implemented"
    torch.manual_seed(7)
    original = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 9, 8), torch.randn(2, 3, 9, 8),
                torch.rand(2, 3, 9), torch.randn(2, 3, 9, 8)]
    mask = torch.ones(2, 9, dtype=torch.bool)
    results = []
    for reused in (False, True):
        p, k, v, beta, q = [x.clone().requires_grad_() for x in original]
        state = core.correct_with_aux(p, k, v, beta, mask).candidate if reused else core.correct(p, k, v, beta, mask)[0]
        loss = (q @ state).square().mean()
        results.append(torch.autograd.grad(loss, (p, k, v, beta, q)))
    for actual, expected in zip(*results): torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("psi_scale", [0., .3, 4.])
def test_projected_psi_matches_fp64_finite_difference(psi_scale):
    torch.manual_seed(3407)
    u, v = [torch.nn.functional.normalize(torch.randn(1, 3, 8, dtype=torch.float64), dim=-1) for _ in range(2)]
    previous = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    k, value = [torch.randn(1, 1, 5, 8, dtype=torch.float64) for _ in range(2)]
    beta = torch.rand(1, 1, 5, dtype=torch.float64)
    mask = torch.tensor([[True, True, True, False, False]])
    psi = torch.tensor([[[1., -1., .5]]], dtype=torch.float64) * psi_scale
    cbase = torch.randn_like(psi) * .2
    factors = core.CayleyFactors(u, v, cbase + .7 * psi.tanh())
    aux = core.correct_with_aux(factors.right(previous), k, value, beta, mask)
    snapshot = core.DetachedCayleySnapshot.from_live(factors, psi, cbase, (0, 0, 0))
    actual = core.analytic_psi_gradient_projected(previous, aux.kt_weighted_residual, snapshot, mask.sum(-1), .7)
    def loss(p):
        predicted = previous @ core.cayley_dense(u, v, cbase + .7 * p.tanh())
        return core.innovation_loss(predicted, k, value, aux.w, mask).sum()
    fd = torch.empty_like(psi)
    for index in range(3):
        direction = torch.zeros_like(psi); direction[..., index] = 1e-5
        fd[..., index] = (loss(psi + direction) - loss(psi - direction)) / 2e-5
    torch.testing.assert_close(actual, fd, rtol=0, atol=1e-8)


def test_optimized_snapshot_reuse_and_clean_failure_are_atomic(monkeypatch):
    from test_training import TinyWorldModel, inputs
    from worldttn.performance import ExecutionOptions, configure_execution
    from worldttn.session import TTNSession
    import worldttn.runtime as runtime_module
    model = TinyWorldModel()
    configure_execution(model, ExecutionOptions("reuse", "projected"))
    clean, _, _, camera = inputs()
    session = TTNSession(model, camera, 100, 100)
    session.reset(1)
    y = torch.zeros(1, 1, 2, 8)
    cache = [[None] * 10 for _ in model.blocks]
    with torch.no_grad(): session.prefill(clean[:, :, :1], y)
    before_s, before_psi = [t.clone() for t in (session.runtime.world_state, session.runtime.transition_fast)]
    calls = []
    original = runtime_module.CayleyFactors
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(runtime_module, "CayleyFactors", count)
    context = session.begin_chunk(0, 4)
    assert context.for_clean().psi_snapshot is context.psi_snapshot
    for _ in range(3):
        session.forward(clean[:, :, :4], torch.ones(1) * 500, y, context, cache, 0, 4)
    assert len(calls) == 1
    torch.testing.assert_close(session.runtime.world_state, before_s, rtol=0, atol=0)
    torch.testing.assert_close(session.runtime.transition_fast, before_psi, rtol=0, atol=0)
    original_forward = model.forward
    def fail(*args, **kwargs):
        out, caches = original_forward(*args, **kwargs)
        return out * float("nan"), caches
    model.forward = fail
    with pytest.raises(FloatingPointError): session.clean_forward(clean[:, :, :4], y, context, cache, 0, 4)
    assert session.runtime.commit_count == 1
    torch.testing.assert_close(session.runtime.world_state, before_s, rtol=0, atol=0)
    torch.testing.assert_close(session.runtime.transition_fast, before_psi, rtol=0, atol=0)
    model.forward = original_forward
    session.clean_forward(clean[:, :, :4], y, context, cache, 0, 4)
    assert len(calls) == 1 and context.psi_snapshot is None and session.runtime.commit_count == 2


def test_execution_options_do_not_change_weight_or_config_schema():
    from worldttn.performance import ExecutionOptions, configure_execution, execution_report
    from test_training import TinyWorldModel
    model = TinyWorldModel()
    keys, cfg = set(model.state_dict()), model.ttn_system.config.to_dict()
    configure_execution(model, ExecutionOptions("reuse", "projected"))
    assert set(model.state_dict()) == keys and model.ttn_system.config.to_dict() == cfg
    assert execution_report(model)["psi_implementation"] == "projected"
    with pytest.raises(ValueError): ExecutionOptions("reference", "projected")
    with pytest.raises(ValueError, match="profil|gate|implemented"): ExecutionOptions("compiled", "triton")


@pytest.mark.parametrize("backend", [("reuse", "reference"), ("reuse", "projected")])
@pytest.mark.parametrize("tbptt", [1, 2, 4])
def test_optimized_real_training_preserves_slow_gradients_and_states(backend, tbptt):
    from worldttn.performance import ExecutionOptions, configure_execution
    from test_training import TinyWorldModel, inputs
    from worldttn.checkpoint import make_optimizer
    from worldttn.training import train_clip, linear_flow_loss
    torch.manual_seed(19)
    baseline = TinyWorldModel()
    # Test non-zero controller coefficients, not just zero-initialized identity.
    with torch.no_grad():
        for head in baseline.ttn_system.controller.heads:
            head.weight.normal_(0, .002)
    optimized = copy.deepcopy(baseline)
    configure_execution(optimized, ExecutionOptions(*backend))
    clean, noise, t, cam = inputs()
    results = []
    for model in (baseline, optimized):
        results.append(train_clip(model, clean, torch.zeros(1, 1, 2, 8), cam, make_optimizer(model),
                       linear_flow_loss, t, noise, width=100, height=100, tbptt=tbptt, activation_offload="cpu"))
    for key in ("world_state", "transition_fast"):
        torch.testing.assert_close(getattr(results[0]["runtime"], key), getattr(results[1]["runtime"], key), atol=1e-6, rtol=1e-4)
    for (name, a), (_, b) in zip(baseline.named_parameters(), optimized.named_parameters()):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
        if a.grad is None: assert b.grad is None, name
        else: torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    assert optimized.ttn_system.generators.u.grad.norm() > 0
    assert optimized.ttn_system.controller.heads[0].weight.grad.norm() > 0


def test_snapshot_mismatch_rejects_clean_anchor_without_commit():
    from worldttn.performance import ExecutionOptions, configure_execution
    from test_training import TinyWorldModel
    from test_anchor import context
    model = TinyWorldModel()
    configure_execution(model, ExecutionOptions("reuse", "projected"))
    from worldttn.runtime import TTNRuntimeState
    runtime = TTNRuntimeState.create(model.ttn_system.config, 1, "cpu")
    pose = torch.eye(4).expand(1, 2, 4, 4)
    intr = torch.tensor([100., 100., 50., 50.]).expand(1, 2, 4)
    c = runtime.begin_chunk(model.ttn_system, pose, intr, torch.arange(2)[None], torch.ones(1, 2, dtype=torch.bool), 100, 100).for_clean()
    c.begin_id += 1
    with pytest.raises(RuntimeError, match="snapshot"):
        model.blocks[3].attn(torch.randn(1, 2, 16), ttn_chunk_context=c)
    assert runtime.commit_count == 0 and runtime.world_state.count_nonzero() == 0
