"""Protected statistics reuse keeps the full episode gradient live."""
from dataclasses import replace

import pytest
import torch

from test_frame_memory import config, data
from worldttn import frame_memory, memory_cache, proximal


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sealed_prefix_is_reused_and_matches_uncached_gradients(monkeypatch, dtype):
    cfg = config(memory_start_frame=3)
    frames = 19
    tensors = data(frames=frames, dtype=dtype)
    inputs = [t.detach().requires_grad_() for t in tensors]
    calls = []
    original = proximal._source
    def counted(key, value, weight):
        calls.append(key.shape[-2])
        return original(key, value, weight)
    monkeypatch.setattr(proximal, "_source", counted)
    def scan(values):
        prior, q, k, v, beta = values
        return frame_memory.frame_correct(prior, q, k, v, beta,
            torch.ones(1, frames*2, dtype=torch.bool), torch.arange(frames)[None],
            (frames, 1, 2), cfg, collect_stats=True)
    reused = scan(inputs)
    assert reused[3].prefix_statistics is not None
    assert reused[3].prefix_statistics[1][-2].grad_fn is not None
    reuse_calls = len(calls)
    grads = torch.autograd.grad(reused[1][:, :, -2:].square().sum(), inputs)
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    assert grads[2][:, :, :2].norm() > 0  # late loss reaches protected first-frame K
    assert grads[3][:, :, :2].norm() > 0

    calls.clear()
    monkeypatch.setattr(frame_memory, "_prefix_statistics", lambda *args: None)
    fresh = [t.detach().requires_grad_() for t in tensors]
    baseline = scan(fresh)
    baseline_grads = torch.autograd.grad(baseline[1][:, :, -2:].square().sum(), fresh)
    assert len(calls) > reuse_calls
    torch.testing.assert_close(reused[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(reused[1], baseline[1], rtol=0, atol=0)
    torch.testing.assert_close(grads, baseline_grads, rtol=2e-5, atol=2e-7)
    for actual, expected in zip(reused[4], baseline[4]):
        actual, expected = dict(actual), dict(expected)
        actual.pop("prefix_statistics_reused")
        expected.pop("prefix_statistics_reused")
        assert actual == expected


def test_prefix_summary_detaches_at_episode_boundary_and_invalidates_on_new_prefix():
    cfg = config(memory_start_frame=3)
    prior, q, k, v, beta = [t.requires_grad_() for t in data(frames=4)]
    retained = frame_memory.frame_correct(prior, q, k, v, beta, torch.ones(1, 8, dtype=torch.bool),
        torch.arange(4)[None], (4, 1, 2), cfg)[3]
    detached = memory_cache.detach_cache(retained)
    assert detached.prefix_statistics is not None
    assert not any(t.requires_grad for t in detached.prefix_statistics[1])
    # Changing the protected-prefix definition must discard the old statistics.
    changed = memory_cache.update_cache(retained, k[:, :, :2], v[:, :, :2], beta[:, :, :2],
        q[:, :, :2], torch.ones(1, 2, dtype=torch.bool), torch.tensor([[4]]), (1, 1, 2),
        capacity_frames=5, prefix_frames=2, recent_frames=1)
    assert changed.prefix_statistics is None


def test_prefix_reuse_with_ragged_batch_thresholds_and_autocast(monkeypatch):
    cfg = config(memory_start_frame=3)
    frames = 19
    prior, q, k, v, beta = [t.repeat(2, *([1]*(t.ndim-1))) for t in data(frames=frames, dtype=torch.float32)]
    ids = torch.arange(frames)[None] + torch.tensor([[0], [2]])
    write = torch.ones(2, frames*2, dtype=torch.bool)
    write[1, 6:10] = False
    def scan():
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return frame_memory.frame_correct(prior, q, k, v, beta, write, ids, (frames, 1, 2), cfg)
    actual = scan()
    monkeypatch.setattr(frame_memory, "_prefix_statistics", lambda *args: None)
    reference = scan()
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for H20/D112, 121-frame fullgrad gate")
def test_cuda_real_head_dimension_121_frames_prefix_reuse_and_late_credit(monkeypatch):
    """Real recurrent dimensions on two spatial tokens, not a full-model memory gate."""
    cfg = replace(config(), heads=20, head_dim=112, memory_frame_kappa=49.,
        memory_prefix_frames=10, memory_capacity_frames=16, memory_recent_frames=4, memory_start_frame=13)
    frames, spatial, device = 121, 2, torch.device("cuda")
    torch.manual_seed(129)
    prior = torch.randn(1, 20, 112, 112, device=device) * .01
    q, k, v = [torch.randn(1, 20, frames * spatial, 112, device=device) * .1 for _ in range(3)]
    beta = torch.rand(1, 20, frames * spatial, device=device) + .1
    tensors = prior, q, k, v, beta
    write = torch.ones(1, frames * spatial, dtype=torch.bool, device=device)
    ids = torch.arange(frames, device=device)[None]

    def scan(values):
        incoming, query, key, value, weight = values
        return frame_memory.frame_correct(incoming, query, key, value, weight, write, ids,
                                          (frames, 1, spatial), cfg)

    live = [value.detach().requires_grad_() for value in tensors]
    reused = scan(live)
    assert reused[3].prefix_statistics is not None
    assert reused[3].prefix_statistics[0] == 10
    assert reused[3].prefix_statistics[1][-2].grad_fn is not None
    retained_ids = reused[3].observation.frame_ids[0]
    assert torch.isin(torch.arange(10, device=device), retained_ids).all()
    assert retained_ids.numel() <= cfg.memory_capacity_frames * spatial
    gradients = torch.autograd.grad(reused[1][:, :, -spatial:].square().sum(), live)
    assert all(torch.isfinite(gradient).all() and gradient.norm() > 0 for gradient in gradients)
    assert gradients[2][:, :, :spatial].norm() > 0, "last frame must train first-frame K"
    assert gradients[3][:, :, :spatial].norm() > 0, "last frame must train first-frame V"
    reused_state, reused_output = reused[0].detach(), reused[1].detach()
    del reused, live

    monkeypatch.setattr(frame_memory, "_prefix_statistics", lambda *args: None)
    fresh = [value.detach().requires_grad_() for value in tensors]
    baseline = scan(fresh)
    baseline_gradients = torch.autograd.grad(baseline[1][:, :, -spatial:].square().sum(), fresh)
    torch.testing.assert_close(reused_state, baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(reused_output, baseline[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients, baseline_gradients, rtol=2e-5, atol=2e-7)
