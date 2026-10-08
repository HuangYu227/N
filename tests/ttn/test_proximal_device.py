"""Single-GPU numerical gates; these do not substitute for LTU FSDP2 acceptance."""
import copy

import pytest
import torch

from test_alignment_detail import cached_sana
from test_full_training import EulerOracle, long_inputs
from test_proximal import inputs, observation
from test_proximal_runtime import proximal_model
from worldttn import training
from worldttn.checkpoint import make_optimizer, rng_state, restore_rng
from worldttn.proximal import proximal_correct

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_d112_cuda_solve_and_outer_gradients_match_fp64_cpu():
    p, k, v, beta, mask = inputs(batch=1, heads=2, tokens=24, dim=112)
    tensors = [p, k, v, beta, k[:, :, :7].clone(), v[:, :, :7].clone(), beta[:, :, :7].clone()]
    def run(device, dtype):
        a = [t.to(device=device, dtype=dtype).detach().requires_grad_() for t in tensors]
        state, _, stats = proximal_correct(*a[:4], mask.to(device),
            history=(observation(*a[4:]),), collect_stats=True)
        gradients = torch.autograd.grad(state.square().mean(), a)
        assert torch.isfinite(state).all()
        assert all(g.norm() > 0 and torch.isfinite(g).all() for g in gradients)
        return state.detach().cpu(), tuple(g.detach().cpu() for g in gradients), stats
    expected, expected_grad, _ = run("cpu", torch.float64)
    actual, actual_grad, stats = run("cuda", torch.float32)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=3e-7)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a.double(), b, rtol=2e-4, atol=2e-7)
    assert max(stats["per_head"]["solve_condition_upper_bound"][0]) <= 8.00001


def test_cuda_bf16_anchor_training_cpu_offload_preserves_update(monkeypatch, cached_sana):
    monkeypatch.setattr(training, "_history_scheduler", EulerOracle)
    base = proximal_model(cached_sana).cuda()
    offloaded = copy.deepcopy(base)
    clean, noise, t, camera = [v.cuda() for v in long_inputs()]
    initial_rng = copy.deepcopy(rng_state())
    def run(model, mode):
        restore_rng(initial_rng)  # Pair the independent generated-history noise as well as flow noise.
        phases = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = training.train_clip(model, clean, torch.zeros(1, 1, 2, 8, device="cuda"), camera,
                make_optimizer(model), training.linear_flow_loss, t, noise, width=100, height=100,
                tbptt=4, activation_offload=mode, memory_callback=lambda p, **i: phases.append((p, i)),
                history_training={"source": "generated", "steps": 2, "cached_chunks": 2})
        assert result["runtime"].verify_memory_prefix()
        assert not result["runtime"].world_state.requires_grad
        assert result["runtime"].transition_fast.count_nonzero() == 0
        assert all(not c.observation.key.requires_grad for c in result["runtime"].memory_caches)
        effects = [a["proximal"]["output_effect"] for c in result["chunks"] for a in c["anchors"]]
        assert all(e["delta_norm"] > 0 and e["changed_fraction"] > 0 for e in effects)
        return result, phases
    reference, _ = run(base, "none")
    actual, phases = run(offloaded, "cpu")
    assert actual["loss"] == reference["loss"]
    for (name, a), (_, b) in zip(base.named_parameters(), offloaded.named_parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=3e-7, msg=name)
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-7, msg=name)
    assert any(i["offload_saved_tensors"]["packed_tensor_bytes"] > 0 for p, i in phases if p == "backward_begin")
