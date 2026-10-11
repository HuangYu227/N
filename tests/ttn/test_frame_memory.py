from dataclasses import replace

import pytest
import torch

from worldttn.core import TTNConfig
from worldttn.frame_memory import frame_correct
from worldttn.proximal import proximal_correct


def config(**kw):
    return TTNConfig(heads=2, head_dim=8, stage="C", camera_attention="sana", persistent_update=False,
        memory_update="proximal", memory_granularity="frame", memory_frame_kappa=4.,
        memory_capacity_frames=4, memory_prefix_frames=1, memory_recent_frames=1, **kw)


def data(frames=5, spatial=2, dtype=torch.float64):
    torch.manual_seed(81)
    q, k, v = [torch.randn(1, 2, frames*spatial, 8, dtype=dtype) for _ in range(3)]
    return torch.randn(1, 2, 8, 8, dtype=dtype), q, k, v, torch.rand(1, 2, frames*spatial, dtype=dtype)+.1


def test_frame_scan_matches_manual_solve_and_partitioned_calls():
    prior, q, k, v, beta = data()
    ids = torch.arange(1, 6)[None]
    write = torch.ones(1, 10, dtype=torch.bool)
    cfg = config(memory_start_frame=3)
    whole = frame_correct(prior, q, k, v, beta, write, ids, (5, 1, 2), cfg, collect_stats=True)
    state, cache, pieces = prior, None, []
    for f in range(5):
        sl = slice(f*2, f*2+2)
        result = frame_correct(state, q[:, :, sl], k[:, :, sl], v[:, :, sl], beta[:, :, sl],
            write[:, sl], ids[:, f:f+1], (1, 1, 2), cfg, retained=cache)
        state, read, _, cache, _ = result
        pieces.append(read)
    torch.testing.assert_close(whole[0], state, rtol=0, atol=0)
    torch.testing.assert_close(whole[1], torch.cat(pieces, -2), rtol=0, atol=0)
    assert whole[4][0]["history_active"] == [False]
    assert whole[4][2]["history_active"] == [True]
    assert cache.observation.key.shape[-2] <= 8
    one, _, _ = proximal_correct(prior, k[:, :, :2], v[:, :, :2], beta[:, :, :2], write[:, :2], kappa=4.)
    torch.testing.assert_close(whole[1][:, :, :2], q[:, :, :2] @ one)
    assert not torch.allclose(whole[1][:, :, :2], q[:, :, :2] @ whole[0])


def test_frame_permutation_padding_and_no_future_leakage():
    prior, q, k, v, beta = data(frames=3, spatial=4)
    cfg = config(memory_selection="none")
    ids, write = torch.arange(1, 4)[None], torch.ones(1, 12, dtype=torch.bool)
    baseline = frame_correct(prior, q, k, v, beta, write, ids, (3, 1, 4), cfg)
    permutation = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5, 10, 8, 11, 9])
    changed = frame_correct(prior, q[:, :, permutation], k[:, :, permutation], v[:, :, permutation],
        beta[:, :, permutation], write, ids, (3, 1, 4), cfg)
    torch.testing.assert_close(baseline[0], changed[0])
    torch.testing.assert_close(baseline[1][:, :, permutation], changed[1])
    v2 = v.clone(); v2[:, :, -4:] += 100
    future = frame_correct(prior, q, k, v2, beta, write, ids, (3, 1, 4), cfg)
    torch.testing.assert_close(baseline[1][:, :, :8], future[1][:, :, :8], rtol=0, atol=0)
    write[:] = False
    padded = frame_correct(prior, q, k.fill_(float("nan")), v, beta, write, ids, (3, 1, 4), cfg)
    torch.testing.assert_close(padded[0], prior, rtol=0, atol=0)


def test_full_frame_gradient_including_lambda_and_history():
    prior, q, k, v, beta = data(frames=2, spatial=2)
    cfg = config(memory_start_frame=1)
    tensors = tuple(t.requires_grad_() for t in (prior, k, v, beta))
    def fn(p, key, value, weight):
        return frame_correct(p, q, key, value, weight, torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[1, 2]]), (2, 1, 2), cfg)[1][:, :, -2:]
    assert torch.autograd.gradcheck(fn, tensors, fast_mode=True)
    grads = torch.autograd.grad(fn(*tensors).square().sum(), tensors)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    assert grads[1][:, :, :2].norm() > 0
    assert grads[2][:, :, :2].norm() > 0


def test_legacy_identity_and_invalid_frame_recipe():
    assert "memory_granularity" not in TTNConfig().to_dict()
    assert TTNConfig(**config().to_dict()) == config()
    with pytest.raises(ValueError, match="frame memory"):
        replace(config(), memory_transport="cayley")
