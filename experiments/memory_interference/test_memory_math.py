import math

import pytest
import torch

from memory_math import (check_clean_pair, clone_tree, make_reference,
                         retention_metrics, sigma_indices)


def inputs():
    torch.manual_seed(771)
    q, k, v = [torch.randn(1, 2, 17, 8, dtype=torch.float64) for _ in range(3)]
    s = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    # Production state math is FP32, so these oracle checks use that dtype too.
    return [x.float() for x in (q, k, v, s)]


def test_known_value_transport_does_not_look_like_forgetting():
    q, k, v, s = inputs()
    ref = make_reference(q, k, v, s, count=7)
    r = torch.linalg.qr(torch.randn(2, 8, 8)).Q
    identity = torch.eye(8).expand(2, -1, -1)
    before = retention_metrics(s, ref, identity)
    after = retention_metrics(s @ r, ref, r)
    assert after['old_query_response_drift'] < 1e-12
    assert math.isclose(before['old_key_binding_nmse'], after['old_key_binding_nmse'], rel_tol=2e-6)
    assert len(after['per_token_key_nmse']) == 7


def test_reference_is_a_snapshot_and_a_changed_state_is_detected():
    q, k, v, s = inputs()
    ref = make_reference(q, k, v, s)
    saved = ref['k'].clone()
    k.zero_()
    assert torch.equal(saved, ref['k'])
    out = retention_metrics(s + .5, ref, torch.eye(8))
    assert out['old_query_response_drift'] > 0


def test_diagonal_addresses_prevent_cross_update_but_dense_projection_does_not():
    old, new = torch.eye(2)
    dense_basis = torch.ones(2, 1) / math.sqrt(2)
    assert float(old @ new) == 0
    assert float(old @ dense_basis @ dense_basis.T @ new) > .49


def test_sigma_indices_are_selected_from_actual_grid():
    indices, actual = sigma_indices([.95, .75, .5, .25, .05], [.2, .5, .8])
    assert indices == [3, 2, 1] and actual == [.25, .5, .75]
    with pytest.raises(ValueError):
        sigma_indices([.5], [.2, .8])


def test_private_cache_copy_cannot_mutate_committed_cache():
    cache = [[torch.ones(3), None], {'cache': (torch.ones(2),)}]
    private = clone_tree(cache)
    private[0][0].zero_()
    private[1]['cache'][0].add_(9)
    assert cache[0][0].sum() == 3 and cache[1]['cache'][0].sum() == 2


def test_clean_pair_check_rejects_a_noisy_commit_leak():
    sample = dict(clean_state_hashes=['a', 'b'], prefix_hashes=['p'])
    check_clean_pair(sample, dict(sample))
    with pytest.raises(AssertionError):
        check_clean_pair(sample, dict(sample, clean_state_hashes=['a', 'changed']))


def test_zero_reference_is_finite_and_bad_layout_is_rejected():
    q, k, v, s = [torch.zeros_like(x) for x in inputs()]
    result = retention_metrics(s, make_reference(q, k, v, s), torch.eye(8))
    assert result['old_key_binding_nmse'] == result['old_query_response_drift'] == 0
    with pytest.raises(ValueError):
        make_reference(q, k[..., :2, :], v, s)
