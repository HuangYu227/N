"""Frozen-probe memory diagnostics; these proxies are not video-quality scores."""
import torch


MODES = ('full', 'no-local', 'no-persistent', 'no-ttt')


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    return value


def sigma_indices(sigmas, requested):
    values = torch.as_tensor(sigmas, dtype=torch.float64).flatten().cpu()
    if not torch.isfinite(values).all() or not values.numel():
        raise ValueError('Nonfinite or empty production sigma grid')
    if any(not 0 < s < 1 for s in requested):
        raise ValueError('Diagnostic sigmas must lie strictly between zero and one')
    indices = [int((values - s).abs().argmin()) for s in requested]
    if len(set(indices)) != len(indices):
        raise ValueError('Requested sigmas collapse to the same production index')
    return indices, [float(values[i]) for i in indices]


def make_reference(q, k, v, state, count=64):
    if q.shape != k.shape or k.shape != v.shape or k.shape[:-2] != state.shape[:-2]:
        raise ValueError('Reference Q/K/V/state dimensions disagree')
    if state.shape[-2:] != (k.shape[-1], v.shape[-1]) or count < 1:
        raise ValueError('Invalid reference state or probe count')
    indices = torch.linspace(0, k.shape[-2] - 1, min(count, k.shape[-2]), device=k.device).long()
    q0, k0, v0 = [x.index_select(-2, indices).float().detach().clone() for x in (q, k, v)]
    return dict(q=q0, k=k0, v=v0, response=(q0 @ state.float()).detach().clone(),
                token_indices=indices.cpu().tolist())


def retention_metrics(state, reference, transport):
    """Compensate only the known value-channel rotation, not physical geometry."""
    target = reference['v'] @ transport
    response = reference['response'] @ transport
    key_error = (reference['k'] @ state - target).square().mean(-1)
    query_error = (reference['q'] @ state - response).square().mean(-1)
    key_energy = target.square().mean((-2, -1), keepdim=True).clamp_min(1e-12).squeeze(-1)
    query_energy = response.square().mean((-2, -1), keepdim=True).clamp_min(1e-12).squeeze(-1)
    key_relative, query_relative = key_error / key_energy, query_error / query_energy
    if not torch.isfinite(key_relative).all() or not torch.isfinite(query_relative).all():
        raise FloatingPointError('Nonfinite frozen-probe metric')
    return dict(old_key_binding_nmse=float(key_relative.mean()),
                old_query_response_drift=float(query_relative.mean()),
                per_token_key_nmse=key_relative.mean((0, 1)).cpu().tolist(),
                per_token_query_drift=query_relative.mean((0, 1)).cpu().tolist())


def check_clean_pair(left, right):
    """Noisy-only ablations must not alter teacher-forced clean commits."""
    if left['clean_state_hashes'] != right['clean_state_hashes']:
        raise AssertionError('Noisy temporary updates leaked into clean-history state')
    if left['prefix_hashes'] != right['prefix_hashes']:
        raise AssertionError('Paired modes did not start from the same observed prefix')
