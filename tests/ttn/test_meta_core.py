"""Independent autograd oracles catch signs, scaling and detached meta paths."""
import pytest
import torch
from worldttn import core

torch.set_num_threads(1)


def tensors(device="cpu"):
    torch.manual_seed(1105)
    shapes = [(1, 2, 8, 8), (1, 2, 5, 8), (1, 2, 5, 8),
              (1, 2, 5), (2, 2, 8), (2, 2, 8), (1, 2, 2), (1, 2, 2)]
    return tuple((.2 * torch.randn(s, dtype=torch.float64, device=device)).requires_grad_() for s in shapes)


def analytic(args, local):
    s, k, v, beta_logits, u, vgen, cbase, psi = args
    mask = torch.tensor([[True, False, True, True, False]], device=s.device)
    w = core.write_weights(beta_logits.sigmoid(), torch.ones_like(mask), 1e-6)
    fac = core.CayleyFactors(u, vgen, cbase + psi.tanh())
    loss, h = core.innovation_objective(fac.right(s), k, v, w, mask, normalized=local)
    gc = core.coefficient_gradient(s, h, fac.p, fac.l)
    return loss, gc if local else (1 - psi.tanh().square()) * gc


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_projected_gradient_and_meta_derivatives_match_autograd(local, device):
    args = tensors(device)
    s, k, v, beta_logits, u, vgen, cbase, psi = args
    zeta = torch.zeros_like(psi, requires_grad=True)
    mask = torch.tensor([[True, False, True, True, False]], device=s.device)
    w = core.write_weights(beta_logits.sigmoid(), torch.ones_like(mask), 1e-6)
    coeff = cbase + psi.tanh() + (zeta.tanh() if local else 0)
    pred = s @ core.cayley_dense(u, vgen, coeff)
    selected = torch.where(mask[:, None], w, 0.)
    numerator = (selected * (k @ pred - v).square().sum(-1)).sum(-1)
    denominator = (1e-6 + (selected * v.square().sum(-1)).sum(-1) if local
                   else 2 * 8 * 3)
    expected_loss = numerator / denominator
    expected = torch.autograd.grad(expected_loss.sum(), zeta if local else psi, create_graph=True)[0]
    actual_loss, actual = analytic(args, local)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-10, rtol=0)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)
    a = torch.autograd.grad(actual.square().sum(), args, retain_graph=True)
    e = torch.autograd.grad(expected.square().sum(), args)
    for left, right in zip(a, e):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=0)
    # In particular, V/beta derivatives include the live normalization denominator.
    assert a[2].norm() > 0 and a[3].norm() > 0


def test_local_gradient_gradcheck_and_gradgradcheck():
    torch.manual_seed(3)
    shapes = [(1, 1, 4, 4), (1, 1, 3, 4), (1, 1, 3, 4),
              (1, 1, 3), (1, 1, 4), (1, 1, 4), (1, 1, 1), (1, 1, 1)]
    args = tuple((.2 * torch.randn(s, dtype=torch.float64)).requires_grad_() for s in shapes)
    def fn(s, k, v, beta_logits, u, vgen, cbase, psi):
        mask = torch.ones(1, 3, dtype=torch.bool)
        w = core.write_weights(beta_logits.sigmoid(), mask, 1e-6)
        fac = core.CayleyFactors(u, vgen, cbase + psi.tanh())
        _, h = core.innovation_objective(fac.right(s), k, v, w, mask, normalized=True)
        return core.coefficient_gradient(s, h, fac.p, fac.l)
    assert torch.autograd.gradcheck(fn, args, fast_mode=True)
    assert torch.autograd.gradgradcheck(fn, args, fast_mode=True)


@pytest.mark.parametrize("normalized", [True, False])
def test_empty_or_masked_nonfinite_observations_do_not_contaminate_inner_loss(normalized):
    s, k, v, b, *_ = tensors()
    mask = torch.tensor([[True, False, True, True, False]])
    w = b.sigmoid()
    expected = core.innovation_objective(s, k, v, w, mask, normalized=normalized)
    k, v, w = k.clone(), v.clone(), w.clone()
    k[:, :, [1, 4]] = float("nan")
    v[:, :, [1, 4]] = float("nan")
    w[:, :, [1, 4]] = float("nan")
    actual = core.innovation_objective(s, k, v, w, mask, normalized=normalized)
    for a, e in zip(actual, expected): torch.testing.assert_close(a, e, atol=0, rtol=0)
    loss, h = core.innovation_objective(s, k, v, w, torch.zeros_like(mask), normalized=normalized)
    assert loss.count_nonzero() == h.count_nonzero() == 0
    assert torch.isfinite(loss).all() and torch.isfinite(h).all()


def test_clip_is_per_head_and_retains_live_scale_derivative():
    g = torch.tensor([[[.3, .4], [3., 4.]]], dtype=torch.float64, requires_grad=True)
    clipped, scale = core.clip_inner_gradient(g, 1., 1e-6)
    torch.testing.assert_close(clipped, torch.tensor([[[.3, .4], [.6, .8]]], dtype=g.dtype))
    torch.testing.assert_close(scale, torch.tensor([[[1.], [.2]]], dtype=g.dtype))
    actual = torch.autograd.grad(clipped[..., 0].sum(), g)[0]
    torch.testing.assert_close(actual, torch.tensor([[[1., 0.], [.128, -.096]]], dtype=g.dtype))
    zero, _ = core.clip_inner_gradient(torch.zeros_like(g), 1., 1e-6)
    assert torch.isfinite(zero).all() and zero.count_nonzero() == 0


def test_meta_modes_require_stage_c_and_default_to_legacy():
    assert not core.TTNConfig().local_update and not core.TTNConfig().persistent_meta
    for field in ("local_update", "persistent_meta"):
        with pytest.raises(ValueError, match="Stage C"):
            core.TTNConfig(**{field: True})
        assert getattr(core.TTNConfig(stage="C", **{field: True}), field)


def test_rotation_is_blind_to_radial_error_but_direct_s_can_correct_it():
    # Even the COMPLETE skew basis cannot change singular values: more psi
    # coefficients do not fix this limitation. This is not a broken gradient.
    d = 4
    eye = torch.eye(d, dtype=torch.float64)
    pairs = torch.combinations(torch.arange(d))
    u, vgen = eye[pairs[:, 0]], eye[pairs[:, 1]]
    c = torch.zeros(len(pairs), dtype=torch.float64, requires_grad=True)
    state = eye.clone()
    k, v = eye, 2 * eye
    w = torch.ones(d, dtype=eye.dtype)
    mask = torch.ones(d, dtype=torch.bool)
    factors = core.CayleyFactors(u, vgen, c)
    loss, h = core.innovation_objective(factors.right(state), k, v, w, mask, normalized=True)
    analytic = core.coefficient_gradient(state, h, factors.p, factors.l)
    oracle = torch.autograd.grad(loss.sum(), c)[0]
    assert loss > .24 and h.norm() > 0
    torch.testing.assert_close(analytic, oracle, atol=1e-12, rtol=0)
    assert analytic.count_nonzero() == 0
    corrected, _ = core.correct(state, k, v, .5 * w, mask, alpha_s=.5)
    after, _ = core.innovation_objective(corrected, k, v, w, mask, normalized=True)
    assert after < loss and not torch.equal(corrected, state)
