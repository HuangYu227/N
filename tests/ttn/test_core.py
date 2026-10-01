import math
import pytest
import torch
from worldttn.core import cayley_dense, CayleyFactors, correct, innovation_loss, analytic_psi_gradient
from worldttn.geometry import invert_se3, prepare_ray_apply_fns

torch.set_num_threads(1)


def fixture():
    torch.manual_seed(3407)
    u = torch.nn.functional.normalize(torch.randn(2, 3, 8, dtype=torch.float64), dim=-1)
    v = torch.nn.functional.normalize(torch.randn(2, 3, 8, dtype=torch.float64), dim=-1)
    c = torch.randn(2, 3, dtype=torch.float64) * .2
    return u, v, c


def test_lowrank_cayley_matches_dense_and_is_orthogonal():
    u, v, c = fixture()
    r = cayley_dense(u, v, c)
    eye = torch.eye(8, dtype=r.dtype).expand(2, 8, 8)
    assert torch.allclose(CayleyFactors(u, v, c).right(eye), r, atol=1e-10, rtol=0)
    assert torch.allclose(r.transpose(-1, -2) @ r, eye, atol=1e-10, rtol=0)
    assert torch.allclose(cayley_dense(u, v, c * 0), eye, atol=1e-10, rtol=0)
    g = u.unsqueeze(-1) * v.unsqueeze(-2) - v.unsqueeze(-1) * u.unsqueeze(-2)
    assert torch.linalg.matrix_norm(g[:, 0] @ g[:, 1] - g[:, 1] @ g[:, 0]).min() > .01


def test_zero_coefficients_have_nonzero_transition_gradient():
    u, v, c = fixture()
    c = torch.zeros_like(c, requires_grad=True)
    probe = torch.randn(2, 8, 8, dtype=c.dtype)
    loss = (CayleyFactors(u, v, c).right(torch.eye(8, dtype=c.dtype).expand(2, 8, 8)) * probe).sum()
    grad = torch.autograd.grad(loss, c)[0]
    assert torch.isfinite(grad).all() and grad.norm() > .01


def test_correction_reduces_fixed_feature_loss_and_masks_padding():
    u, v, c = fixture()
    k = torch.randn(2, 2, 9, 8, dtype=c.dtype)
    val = torch.randn_like(k)
    state = torch.randn(2, 2, 8, 8, dtype=c.dtype)
    beta = torch.full((2, 2, 9), .5, dtype=c.dtype)
    mask = torch.tensor([[True] * 7 + [False] * 2, [False] * 9])
    new, weights = correct(state, k, val, beta, mask, .5, 1e-6)
    assert (innovation_loss(new, k, val, weights, mask)[0] < innovation_loss(state, k, val, weights, mask)[0]).all()
    assert torch.equal(new[1], state[1])
    val[:, :, 7:] = 1e9
    again, _ = correct(state, k, val, beta, mask, .5, 1e-6)
    assert torch.equal(new, again)
    assert torch.isfinite(again).all()


def test_analytic_psi_gradient_matches_autograd():
    u, v, c = fixture()
    psi = torch.randn_like(c, requires_grad=True) * .1
    prev = torch.randn(2, 8, 8, dtype=c.dtype)
    k = torch.randn(2, 7, 8, dtype=c.dtype)
    val = torch.randn_like(k)
    mask = torch.tensor([[True] * 7, [True] * 4 + [False] * 3])
    beta = torch.sigmoid(torch.randn(2, 7, dtype=c.dtype))
    pred = CayleyFactors(u, v, c + psi.tanh()).right(prev)
    _, w = correct(pred, k, val, beta, mask, .5, 1e-6)
    loss = innovation_loss(pred, k, val, w, mask).sum()
    expected = torch.autograd.grad(loss, psi)[0]
    actual = analytic_psi_gradient(prev, k, val, w, mask, u, v, c, psi, 1.)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=0)


def test_camera_duality_roundtrip_and_associative_mixing():
    torch.manual_seed(7)
    b, h, n, d = 2, 3, 5, 16
    p = torch.eye(4, dtype=torch.float64).expand(b, n, 4, 4).clone()
    theta = torch.randn(b, n, dtype=p.dtype)
    p[..., 0, 0] = theta.cos()
    p[..., 0, 1] = -theta.sin()
    p[..., 1, 0] = theta.sin()
    p[..., 1, 1] = theta.cos()
    p[..., :3, 3] = torch.randn(b, n, 3, dtype=p.dtype)
    freq = torch.polar(torch.ones(b, 1, n, d // 4, dtype=p.dtype), torch.randn(b, 1, n, d // 4, dtype=p.dtype))
    tq, tkv, to = prepare_ray_apply_fns(d, p, p.transpose(-1, -2), invert_se3(p), freq)
    q, k, v = [torch.nn.functional.silu(torch.randn(b, h, n, d, dtype=p.dtype)) for _ in range(3)]
    qt, kt, vt = tq(q), tkv(k), tkv(v)
    assert torch.allclose(to(vt), v, atol=1e-10, rtol=0)
    assert torch.allclose((qt * kt).sum(-1), (q * k).sum(-1), atol=1e-10, rtol=0)
    a = to(qt @ (kt.transpose(-1, -2) @ vt) / (n * math.sqrt(d)))
    explicit = to((qt @ kt.transpose(-1, -2)) @ vt / (n * math.sqrt(d)))
    assert torch.allclose(a, explicit, atol=1e-10, rtol=0)


def test_slow_generator_and_coefficient_gradients_match_dense_oracle():
    u, v, c = fixture()
    u = u.detach().requires_grad_()
    v = v.detach().requires_grad_()
    c = c.detach().requires_grad_()
    x = torch.randn(2, 8, 8, dtype=c.dtype)
    probe = torch.randn_like(x)
    low_loss = (CayleyFactors(u, v, c).right(x) * probe).sum()
    dense_loss = ((x @ cayley_dense(u, v, c)) * probe).sum()
    low = torch.autograd.grad(low_loss, (u, v, c))
    dense = torch.autograd.grad(dense_loss, (u, v, c))
    for actual, expected in zip(low, dense):
        assert torch.allclose(actual, expected, atol=1e-10, rtol=0)
        assert actual.norm() > 0
