"""Direct-S solves and their live inner-update derivatives, independent of DiT."""
import json

import pytest
import torch

from worldttn.proximal import proximal_correct
from worldttn.replay import Observation


def observation(k, v, w):
    ids = torch.zeros(k.shape[0], k.shape[2], dtype=torch.long, device=k.device)
    return Observation(k, v, w, ids, ids)


def inputs(batch=2, heads=2, tokens=5, dim=3):
    generator = torch.Generator().manual_seed(271)
    def rand(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    prior = rand(batch, heads, dim, dim)
    k, v = rand(batch, heads, tokens, dim), rand(batch, heads, tokens, dim)
    beta = rand(batch, heads, tokens).sigmoid()
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    return prior, k, v, beta, mask


def dense_oracle(prior, k, v, beta, mask, histories, strength, kappa, eps):
    result = torch.empty_like(prior)
    for b in range(prior.shape[0]):
        for h in range(prior.shape[1]):
            valid = mask[b]
            weight = beta[b, h, valid]
            # Normalizing write weights by their sum cancels the shared beta mean.
            sources = [(k[b, h, valid], v[b, h, valid], weight, 1.)]
            active = [o for o in histories if o.weight[b, h].sum() > 0]
            sources += [(o.key[b, h], o.value[b, h], o.weight[b, h], strength/len(active))
                        for o in active]
            gram, rhs = torch.zeros_like(prior[b, h]), torch.zeros_like(prior[b, h])
            for key, value, weights, source_weight in sources:
                if weights.sum() == 0:
                    continue
                gram += source_weight * key.T @ (weights[:, None]*key) / weights.sum()
                rhs += source_weight * key.T @ (weights[:, None]*value) / weights.sum()
            lam = kappa*gram.trace()/gram.shape[-1] + eps
            result[b, h] = torch.linalg.solve(gram + lam*torch.eye(gram.shape[-1], dtype=gram.dtype),
                                              rhs + lam*prior[b, h])
    return result


def test_dense_oracle_source_normalization_and_objective_descent():
    p, k, v, beta, mask = inputs()
    mask[0, -1] = False
    first = observation(k[:, :, :2]*.7, v[:, :, :2]*1.3, beta[:, :, :2])
    weights = beta[:, :, 2:4].clone()
    weights[0, 0] = 0
    second = observation(k[:, :, 2:4], v[:, :, 2:4], weights)
    histories = (first, second)
    actual, w, stats = proximal_correct(p, k, v, beta, mask, history=histories,
                                        history_weight=.6, kappa=4., collect_stats=True)
    expected = dense_oracle(p, k, v, beta, mask, histories, .6, 4., 1e-6)
    torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-15)
    assert w[0, :, -1].count_nonzero() == 0
    assert stats['per_head']['active_history_sources'] == [[1, 2], [2, 2]]
    before = torch.tensor(stats['per_head']['inner_objective_before'])
    after = torch.tensor(stats['per_head']['inner_objective_after'])
    assert (after <= before + 1e-7).all()
    json.dumps(stats, allow_nan=False)


def test_gradcheck_gradgradcheck_including_history_and_beta_denominators():
    p, k, v, beta, mask = inputs(batch=1, heads=1, tokens=2, dim=2)
    tensors = tuple(t.requires_grad_() for t in (p, k, v, beta,
                    k.clone(), v.clone(), beta.clone()))
    def function(prior, key, value, weights, hk, hv, hw):
        return proximal_correct(prior, key, value, weights, mask,
             history=(observation(hk, hv, hw),), history_weight=.7, kappa=2.)[0]
    assert torch.autograd.gradcheck(function, tensors, atol=1e-5, rtol=1e-4)
    assert torch.autograd.gradgradcheck(function, tensors, atol=1e-5, rtol=1e-4)


def test_same_forward_inner_derivative_isolated_from_direct_prior_path():
    p, k, v, beta, mask = inputs(batch=1, heads=1)
    p, k, v, beta = (x.requires_grad_() for x in (p, k, v, beta))
    candidate, _, _ = proximal_correct(p, k, v, beta, mask, kappa=4.)
    detached_update = p + (candidate-p).detach()
    torch.testing.assert_close(candidate, detached_update, atol=0, rtol=0)
    live = torch.autograd.grad(candidate.square().sum(), (p, k, v, beta), retain_graph=True)
    detached = torch.autograd.grad(detached_update.square().sum(), (p, k, v, beta), allow_unused=True)
    assert all(g is not None and g.norm() > 1e-8 for g in live)
    assert detached[1:] == (None, None, None)
    assert (live[0]-detached[0]).norm() > 1e-8


def test_isotropic_gain_nullspace_retention_and_zero_prior():
    dtype = torch.float64
    k = torch.eye(3, dtype=dtype).reshape(1, 1, 3, 3)
    p, v = torch.zeros_like(k), k*2
    mask, beta = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 1, 3, dtype=dtype)
    actual, _, _ = proximal_correct(p, k, v, beta, mask, kappa=4., eps=1e-12)
    torch.testing.assert_close(actual, v/(5+3e-12), atol=1e-15, rtol=1e-15)
    p = torch.randn_like(p)
    k[..., 2] = 0
    actual, _, _ = proximal_correct(p, k, v, beta, mask)
    torch.testing.assert_close(actual[..., 2, :], p[..., 2, :], atol=0, rtol=0)


def test_padding_nan_empty_heads_and_no_observations_preserve_prior():
    p, k, v, beta, mask = inputs()
    mask[0] = False
    k[0] = v[0] = beta[0] = float('nan')
    # Head-specific zero beta removes writes without removing the other head.
    beta[1, 0] = 0
    actual, w, stats = proximal_correct(p, k, v, beta, mask, collect_stats=True)
    assert torch.isfinite(actual).all() and torch.isfinite(w).all()
    torch.testing.assert_close(actual[0], p[0], atol=0, rtol=0)
    torch.testing.assert_close(actual[1, 0], p[1, 0], atol=0, rtol=0)
    assert stats['per_head']['current_residual_mse_before'][0] == [None, None]
    empty, _, stats = proximal_correct(p, k[:, :, :0], v[:, :, :0], beta[:, :, :0],
                                        mask[:, :0], collect_stats=True)
    torch.testing.assert_close(empty, p, atol=0, rtol=0)
    json.dumps(stats, allow_nan=False)


def test_history_padding_nan_and_noisy_padding_receive_no_gradient():
    p, k, v, beta, mask = inputs()
    mask[:, -1] = False
    k[:, :, -1] = v[:, :, -1] = float('nan')
    k, v = k.requires_grad_(), v.requires_grad_()
    hw = beta.clone()
    hw[:, :, -1] = 0
    hist = observation(k, v, hw)
    result, _, _ = proximal_correct(p, k, v, beta, mask, history=(hist,))
    grads = torch.autograd.grad(result.square().sum(), (k, v))
    assert all(torch.isfinite(g).all() and g[:, :, -1].count_nonzero() == 0 for g in grads)


def test_token_duplication_weight_scale_and_history_only_are_normalized():
    p, k, v, beta, mask = inputs()
    actual, _, _ = proximal_correct(p, k, v, beta, mask, kappa=4.)
    repeat, _, _ = proximal_correct(p, k.repeat_interleave(2, 2), v.repeat_interleave(2, 2),
             beta.repeat_interleave(2, 2)*.17, mask.repeat_interleave(2, 1), kappa=4.)
    torch.testing.assert_close(actual, repeat, atol=2e-15, rtol=2e-15)
    only, _, _ = proximal_correct(p, k, v, beta, torch.zeros_like(mask),
                history=(observation(k, v, beta),), history_weight=1., kappa=4.)
    torch.testing.assert_close(actual, only, atol=2e-15, rtol=2e-15)
    scaled, _, _ = proximal_correct(p, k*3, v*3, beta, mask, kappa=4., eps=9e-6)
    torch.testing.assert_close(actual, scaled, atol=2e-15, rtol=2e-15)


@pytest.mark.parametrize('parameter,value', [('history_weight', -1.), ('history_weight', float('nan')),
        ('kappa', 0.), ('kappa', float('inf')), ('eps', 0.), ('kappa', True)])
def test_invalid_scalar_rejected(parameter, value):
    with pytest.raises(ValueError):
        proximal_correct(*inputs(), **{parameter: value})


def test_invalid_shapes_live_values_and_history_weights_rejected():
    p, k, v, beta, mask = inputs()
    with pytest.raises(ValueError, match='shape'):
        proximal_correct(p, k[..., :2], v, beta, mask)
    with pytest.raises(ValueError, match='finite'):
        proximal_correct(p, k*float('nan'), v, beta, mask)
    with pytest.raises(ValueError, match='nonnegative'):
        proximal_correct(p, k, v, -beta, mask)
    with pytest.raises(ValueError, match='nonnegative'):
        proximal_correct(p, k, v, beta, mask, history=(observation(k, v, -beta),))


def test_fp32_under_autocast_and_zero_residual_no_change():
    p, k, _, beta, mask = inputs()
    p, k, beta = p.float(), k.float(), beta.float()
    v = k@p
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result, w, stats = proximal_correct(p, k, v, beta, mask, collect_stats=True)
    assert result.dtype == w.dtype == torch.float32
    torch.testing.assert_close(result, p, atol=3e-7, rtol=1e-6)
    json.dumps(stats, allow_nan=False)
