import copy
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer
from worldttn.training import train_clip, linear_flow_loss
from worldttn.fullgradient import checkpoint_tree


def model(checkpoint=True):
    torch.manual_seed(31)
    m = configure_train_scope(TinyWorldModel(local_update=True, persistent_meta=True), "dit")
    m.ttn_fullgrad_checkpoint = checkpoint
    return m


def update(m, k=0, offload="none", callback=None, loss_fn=linear_flow_loss):
    clean, noise, t, camera = inputs()
    return train_clip(m, clean, torch.zeros(1, 1, 2, 8), camera, make_optimizer(m),
                      loss_fn, t, noise, width=100, height=100, tbptt=k,
                      activation_offload=offload, on_prediction=callback)


@pytest.mark.parametrize("offload", ["none", "cpu", "cpu-pageable"])
def test_checkpoint_and_offload_preserve_fullgrad_update(offload):
    a, b = model(False), model(True)
    left, right = update(a), update(b, offload=offload)
    assert left["loss"] == right["loss"]
    assert right["sequence"]["backward_calls"] == 1
    assert right["sequence"]["temporal_detaches"] == 0
    assert right["runtime"].commit_count == 5
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p, q, atol=2e-10, rtol=2e-6, msg=name)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-10, rtol=2e-6, msg=name)


def test_fullgrad_forward_matches_legacy_before_optimizer():
    old, full = model(False), model(True)
    old_outputs, new_outputs = [], []
    a = update(old, k=2, callback=lambda _, value: old_outputs.append(value.detach().clone()))
    b = update(full, callback=lambda _, value: new_outputs.append(value.detach().clone()))
    assert len(new_outputs) == len(old_outputs) == 4
    for left, right in zip(old_outputs, new_outputs):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for key in ("world_state", "transition_fast"):
        torch.testing.assert_close(getattr(a["runtime"], key), getattr(b["runtime"], key), atol=0, rtol=0)


def test_last_chunk_loss_reaches_first_clean_state_and_psi():
    m = model(False)
    captured = []
    original = m.blocks[3].forward
    def record(*args, **kw):
        output = original(*args, **kw)
        context = next((a for a in args if hasattr(a, "clean_mode")), kw.get("ttn_chunk_context"))
        if context is not None and context.clean_mode and not context.prefill_mode and not captured:
            state, gradient, _ = context.candidates[0]
            state.retain_grad(); gradient.retain_grad()
            captured.append((state, gradient))
        return output
    m.blocks[3].forward = record
    def last_only(session, clean, t, noise, y, context, cache, start, end, mask, info, lm, on_prediction=None):
        loss = linear_flow_loss(session, clean, t, noise, y, context, cache, start, end, mask, info, lm, on_prediction)
        return loss if start == 10 else loss*0
    result = update(m, loss_fn=last_only)
    assert result["outer_grad_norm"] > 0
    assert captured
    for value in captured[0]:
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.norm() > 0


def test_checkpoint_restores_zero_stride_inputs():
    base = torch.randn(1, 1, requires_grad=True)
    x = base.expand(3, 4)
    reference = (x*x).sum()
    def operation(v):
        assert v.stride() == x.stride()
        return (v*v).sum()
    actual = checkpoint_tree(operation, (x,))
    expected = torch.autograd.grad(reference, base)[0]
    torch.testing.assert_close(torch.autograd.grad(actual, base)[0], expected)
