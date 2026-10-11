"""Checkpoint inputs remain live while all nested history reaches storage hooks."""
import copy
import gc
import weakref
from dataclasses import dataclass

import pytest
import torch

from test_sdpa_policy import text_attention


@dataclass(frozen=True)
class History:
    state: torch.Tensor
    summary: tuple


def test_checkpoint_does_not_keep_input_tensors_in_a_recursive_closure_cycle():
    from worldttn.sequence import _checkpoint
    enabled = gc.isenabled()
    gc.disable()
    try:
        leaf = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x.detach().clone(), lambda x: x):
            early = leaf.sin()
            witness = weakref.ref(early)
            out = _checkpoint(lambda x: x.cos(), (early,), enabled=True)
            del early
            assert witness() is None, "checkpoint closure must not retain the original GPU input until cyclic GC"
            out.sum().backward()
        torch.testing.assert_close(leaf.grad, -leaf.sin().sin()*leaf.cos())
    finally:
        if enabled:
            gc.enable()


def test_nested_checkpoint_inputs_reach_pack_hooks_and_keep_gradients():
    from worldttn.sequence import _checkpoint
    torch.manual_seed(29)
    early = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    current = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    condition = torch.randn(7, dtype=torch.float64, requires_grad=True)
    history = History(early, (early.square(),))
    inputs = (current, [history], {"camera": condition})
    shapes = []
    def call(x, caches, options):
        return (x + caches[0].state.mean(0) + caches[0].summary[0].mean(0) + options["camera"]).sin()
    expected = call(*inputs)
    reference = torch.autograd.grad(expected.square().sum(), (early, current, condition), retain_graph=True)
    with torch.autograd.graph.saved_tensors_hooks(
            lambda value: (shapes.append(tuple(value.shape)), value.detach().clone())[1], lambda value: value):
        actual = _checkpoint(call, inputs, enabled=True)
    gradients = torch.autograd.grad(actual.square().sum(), (early, current, condition))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradients, reference, rtol=0, atol=0)
    assert shapes.count((5, 7)) == 2, "both live history and its cached sufficient statistics must be packed"
    assert (7,) in shapes, "conditioning cannot remain hidden in the checkpoint closure"


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_text_kv_reuse_preserves_outputs_and_all_gradients(text_attention, amp, qk_norm, record_property):
    torch.manual_seed(13)
    dtype = torch.float32 if amp else torch.float64
    ordinary = text_attention(16, 2, qk_norm=qk_norm).to(dtype)
    reused = copy.deepcopy(ordinary)
    for module in (ordinary, reused): module.set_sdpa_backend("math")
    x = torch.randn(1, 9, 16, dtype=dtype, requires_grad=True)
    cond = torch.randn(1, 1, 5, 16, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    results, counts = [], []
    for module, reuse in ((ordinary, False), (reused, True)):
        count = []
        module.kv_linear.register_forward_hook(lambda *args: count.append(1))
        module.ttn_activation_checkpointing = True
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
            prepared = module.prepare_kv(cond, x.shape[0]) if reuse else None
            out = torch.cat([module(piece, cond, mask, prepared_kv=prepared) if reuse else module(piece, cond, mask)
                             for piece in x.split((4, 5), dim=1)], dim=1)
            loss = out.square().sum()
        gradients = torch.autograd.grad(loss, (x, cond, *module.parameters()))
        results.append((out, gradients))
        counts.append(len(count))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    # Shared backward accumulates group gradients before RMSNorm/Linear; BF16
    # changes that reduction's rounding, never its mathematical derivative.
    torch.testing.assert_close(results[0][1], results[1][1], rtol=3e-2 if amp else 1e-5 if qk_norm else 1e-10,
                               atol=1e-2 if amp else 5e-7 if qk_norm else 1e-10)
    record_property("max_relative_gradient_l2", max(float((a-r).norm()/r.norm().clamp_min(1e-8))
                                                   for r, a in zip(results[0][1], results[1][1])))
    record_property("max_abs_gradient_error", max(float((a-r).abs().max())
                                                for r, a in zip(results[0][1], results[1][1])))
    for reference, actual in zip(results[0][1], results[1][1]):
        assert (actual-reference).norm() <= (1e-2 if amp else 1e-6)*reference.norm().clamp_min(1e-8)
    assert counts[1] == 1 and counts[0] >= 2


def test_text_preparation_preserves_packed_batch_layout(text_attention):
    module = text_attention(16, 2).double()
    module.set_sdpa_backend("math")
    x = torch.randn(2, 3, 16, dtype=torch.float64)
    cond = torch.randn(1, 10, 16, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    prepared = module.prepare_kv(cond, x.shape[0])
    assert prepared[0].shape == (2, 5, 2, 8)
    torch.testing.assert_close(module(x, cond, mask), module(x, cond, mask, prepared_kv=prepared), rtol=0, atol=0)
