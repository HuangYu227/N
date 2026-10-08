"""Differentiable Direct-S regression with an incoming-state quadratic prior."""
import math

import torch

from .core import write_weights


def _source(key, value, weight):
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("proximal weights must be finite and nonnegative")
    active_tokens = weight > 0
    # Padding may contain NaNs; multiplication by zero does not remove them.
    key = torch.where(active_tokens[..., None], key, 0.)
    value = torch.where(active_tokens[..., None], value, 0.)
    if not torch.isfinite(key).all() or not torch.isfinite(value).all():
        raise ValueError("proximal active K/V must be finite")
    mass = weight.sum(-1)
    active = mass > 0
    normalized = weight / torch.where(active, mass, 1.)[..., None]
    gram = key.transpose(-1, -2) @ (normalized[..., None]*key)
    rhs = key.transpose(-1, -2) @ (normalized[..., None]*value)
    return key, value, normalized, active, gram, rhs


def proximal_correct(prior, k, v, beta, mask, *, history=(), history_weight=.25,
                     kappa=16., eps=1e-6, collect_stats=False):
    """Solve one joint regression; do not append Correct or commit noisy results.

    Current observations have source weight one. Each head divides the total
    history weight equally over its nonempty history sources. All normalization
    and solve operations remain live for outer first- and higher-order gradients.
    """
    for name, number, positive in (("history_weight", history_weight, False),
                                    ("kappa", kappa, True), ("eps", eps, True)):
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or number < 0 or (positive and number == 0)):
            raise ValueError(f"proximal {name} must be finite and {'positive' if positive else 'nonnegative'}")
    if (prior.ndim != 4 or prior.shape[-1] != prior.shape[-2] or k.ndim != 4
            or k.shape[:2] != prior.shape[:2] or k.shape[-1] != prior.shape[-1]
            or v.shape != k.shape or beta.shape != k.shape[:-1]
            or mask.shape not in (k.shape[:1] + k.shape[2:3], beta.shape)
            or min(prior.shape) < 1):
        raise ValueError("proximal prior/K/V/beta/mask shape mismatch")
    if any(t.device != prior.device for t in (k, v, beta, mask)):
        raise ValueError("proximal inputs must share a device")
    if any(not t.is_floating_point() for t in (prior, k, v, beta)):
        raise ValueError("proximal prior/K/V/beta must be floating tensors")
    if not torch.isfinite(prior).all():
        raise ValueError("proximal prior must be finite")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("proximal mask must contain only zero or one")
    dtype = torch.float64 if prior.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=prior.device.type, enabled=False):
        prior, k, v, beta = (t.to(dtype) for t in (prior, k, v, beta))
        valid = mask.bool() if mask.ndim == 3 else mask[:, None].bool()
        beta = torch.where(valid, beta, 0.)
        if not torch.isfinite(beta).all() or (beta < 0).any():
            raise ValueError("proximal active beta must be finite and nonnegative")
        w = write_weights(beta, mask, eps)
        current = _source(k, v, w)
        histories = []
        for item in history:
            if (item.key.ndim != 4 or item.key.shape[:2] != prior.shape[:2]
                    or item.key.shape[-1] != prior.shape[-1] or item.value.shape != item.key.shape
                    or item.weight.shape != item.key.shape[:-1]):
                raise ValueError("proximal history K/V/weight shape mismatch")
            tensors = item.key, item.value, item.weight
            if any(t.device != prior.device or not t.is_floating_point() for t in tensors):
                raise ValueError("proximal history must be floating tensors on the state device")
            histories.append(_source(*(t.to(dtype) for t in tensors)))
        count = sum((item[3].to(dtype) for item in histories), torch.zeros_like(current[3], dtype=dtype))
        history_scale = history_weight / count.clamp_min(1)
        gram, rhs = current[-2:]
        for source in histories:
            gram = gram + history_scale[..., None, None]*source[-2]
            rhs = rhs + history_scale[..., None, None]*source[-1]
        lam = kappa*gram.diagonal(dim1=-2, dim2=-1).sum(-1)/prior.shape[-1] + eps
        eye = torch.eye(prior.shape[-1], dtype=dtype, device=prior.device)
        delta = torch.linalg.solve(gram + lam[..., None, None]*eye, rhs-gram@prior)
        candidate = prior + delta
        if not torch.isfinite(candidate).all():
            raise FloatingPointError("nonfinite proximal Correct candidate")
        stats = _statistics(prior, candidate, current, histories, history_scale, count,
                            gram, lam, history_weight, kappa, eps) if collect_stats else None
    return candidate, w, stats


@torch.no_grad()
def _statistics(prior, candidate, current, histories, history_scale, count, gram, lam,
                history_weight, kappa, eps):
    def loss(source, state):
        key, value, normalized = source[:3]
        return .5*(normalized[..., None]*(key@state-value).square()).sum((-2, -1))
    def values(tensor):
        return tensor.detach().cpu().tolist()
    def nullable(tensor, valid):
        rows, valid = values(tensor), values(valid)
        return [[value if ok else None for value, ok in zip(row, valid_row)]
                for row, valid_row in zip(rows, valid)]
    before, after = loss(current, prior), loss(current, candidate)
    history_before = sum((loss(source, prior) for source in histories), torch.zeros_like(before))
    history_after = sum((loss(source, candidate) for source in histories), torch.zeros_like(after))
    delta = candidate-prior
    current_gradient = current[-2]@prior-current[-1]
    history_gradient = sum((source[-2]@prior-source[-1] for source in histories),
                           torch.zeros_like(prior))*history_scale[..., None, None]
    cur_norm = current_gradient.flatten(-2).norm(dim=-1)
    hist_norm = history_gradient.flatten(-2).norm(dim=-1)
    cosine = (current_gradient*history_gradient).sum((-2, -1))/(cur_norm*hist_norm).clamp_min(eps**2)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    dim = prior.shape[-1]
    return {"mode": "proximal_direct_s", "measurement": "fixed-feature inner objective; no quality claim",
            "kappa": kappa, "history_weight": history_weight,
            "per_head": {
                "active_current": values(current[3]), "active_history_sources": values(count.to(torch.long)),
                "inner_objective_before": values(before + history_scale*history_before),
                "inner_objective_after": values(after + history_scale*history_after
                                                  + .5*lam*delta.square().sum((-2, -1))),
                "current_residual_mse_before": nullable(2*before/dim, current[3]),
                "current_residual_mse_after": nullable(2*after/dim, current[3]),
                "history_residual_mse_before": nullable(2*history_before/(dim*count.clamp_min(1)), count > 0),
                "history_residual_mse_after": nullable(2*history_after/(dim*count.clamp_min(1)), count > 0),
                "lambda": values(lam),
                "current_gradient_norm": values(cur_norm),
                "weighted_history_gradient_norm": values(hist_norm),
                "history_current_gradient_ratio": nullable(hist_norm/cur_norm.clamp_min(eps), cur_norm > eps),
                "history_current_gradient_cosine": nullable(cosine, (cur_norm > eps) & (hist_norm > eps)),
                "relative_state_delta": values(delta.flatten(-2).norm(dim=-1)
                                                 / prior.flatten(-2).norm(dim=-1).clamp_min(eps)),
                "solve_condition_upper_bound": values(1+trace/lam),
                "prior_retention_lower_bound": values(lam/(lam+trace)),
                "prior_retention_upper_bound": values(torch.ones_like(lam))}}
