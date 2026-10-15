"""Author-algebra oracle and real cached FFN integration on a tiny grid."""
import copy

import pytest
import torch
from torch.nn import functional as F

from worldttn.inplace_ttt import InPlaceFFNTTT
from test_legacy_sequence import cached_ffn


def inputs():
    torch.manual_seed(71)
    return torch.randn(2, 7, 5), torch.randn(2, 7, 3), torch.randn(2, 3, 5) * .02


def learned():
    model = InPlaceFFNTTT(3, 5)
    with torch.no_grad():
        model.ttt_conv.weight.normal_(std=.03)
        model.ttt_proj.weight.copy_(torch.tensor([[.2, .5, -.1], [-.3, .1, .7], [.4, -.2, .3]]))
    return model


def test_author_nonsymmetric_projection_and_negative_inner_gradient():
    z, target, delta = inputs()
    model = learned()
    mask = torch.ones(z.shape[:2], dtype=torch.bool)
    correction, candidate, stats = model(z, target, delta, mask)
    # Mirrors author's `c h,c d,d e -> e h`, deliberately using nonsymmetric P.
    conv = F.conv1d(target.transpose(1, 2), model.ttt_conv.weight, padding=2, groups=3).transpose(1, 2)
    values = torch.einsum('bnd,de->bne', conv, model.ttt_proj.weight)
    update = model.lr * torch.einsum('bne,bnh->beh', values, z)
    torch.testing.assert_close(correction, torch.einsum('beh,bnh->bne', delta, z))
    torch.testing.assert_close(candidate, delta + update)
    variable = delta.detach().requires_grad_()
    loss = -(torch.bmm(z, variable.transpose(1, 2)) * values).sum()
    gradient, = torch.autograd.grad(loss, variable)
    torch.testing.assert_close(-model.lr * gradient, update)
    assert not stats['raw_update_norm'].requires_grad


def test_zero_initialization_preserves_output_and_starts_conv_learning():
    z, target, delta = inputs()
    model = InPlaceFFNTTT(3, 5)
    zeros = torch.zeros_like(delta)
    output, candidate, _ = model(z, target, zeros, torch.ones(z.shape[:2], dtype=torch.bool))
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(candidate) == 0
    assert torch.count_nonzero(model.ttt_proj.weight - model.ttt_proj.weight.diag().diag()) == 0
    assert torch.count_nonzero(model.ttt_proj.weight.diag()) == 3
    future, _, _ = model(z.flip(1), target, candidate)
    (future - 1).square().sum().backward()
    assert torch.count_nonzero(model.ttt_conv.weight.grad) > 0
    assert torch.count_nonzero(model.ttt_proj.weight.grad) == 0


def test_sequential_reads_match_exclusive_prefix_outputs_and_gradients():
    torch.manual_seed(83)
    model, oracle = learned(), None
    oracle = copy.deepcopy(model)
    z = torch.randn(2, 4, 7, 5, requires_grad=True)
    target = torch.randn(2, 4, 7, 3, requires_grad=True)
    z_ref, t_ref = z.detach().clone().requires_grad_(), target.detach().clone().requires_grad_()
    state = torch.zeros(2, 3, 5)
    outputs = []
    for i in range(4):
        out, state, _ = model(z[:, i], target[:, i], state, torch.ones(2, 7, dtype=torch.bool))
        outputs.append(out)
    sequential = torch.stack(outputs, dim=1)
    tconv = F.conv1d(t_ref.reshape(8, 7, 3).transpose(1, 2), oracle.ttt_conv.weight,
                     padding=2, groups=3).transpose(1, 2).reshape(2, 4, 7, 3)
    updates = oracle.lr * torch.einsum('btch,btcd,de->bteh', z_ref, tconv, oracle.ttt_proj.weight)
    exclusive = torch.cat((torch.zeros_like(updates[:, :1]), updates[:, :-1]), dim=1).cumsum(1)
    reference = torch.einsum('bteh,btch->btce', exclusive, z_ref)
    torch.testing.assert_close(sequential, reference, rtol=2e-5, atol=2e-6)
    sequential.square().mean().backward(); reference.square().mean().backward()
    for left, right in [(z.grad, z_ref.grad), (target.grad, t_ref.grad),
                        (model.ttt_conv.weight.grad, oracle.ttt_conv.weight.grad),
                        (model.ttt_proj.weight.grad, oracle.ttt_proj.weight.grad)]:
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)


def test_current_chunk_target_does_not_change_current_read():
    z, target, delta = inputs()
    model = learned()
    mask = torch.ones(2, 7, dtype=torch.bool)
    out, state, _ = model(z, target, delta, mask)
    altered, new_state, _ = model(z, target + 4, delta, mask)
    torch.testing.assert_close(out, altered, rtol=0, atol=0)
    assert not torch.allclose(state, new_state)
    assert not torch.allclose(model(z, target, state)[0], model(z, target, new_state)[0])


def test_mask_and_per_sample_inference_clip():
    z, target, delta = inputs()
    model = learned()
    mask = torch.ones(2, 7, dtype=torch.bool); mask[1] = False
    _, state, stats = model(z, target, delta, mask, delta_clip=1e-5)
    assert stats['applied_update_norm'][0] <= 1.00001e-5
    assert stats['raw_update_norm'][0] > 1e-5
    torch.testing.assert_close(state[1], delta[1], rtol=0, atol=0)
    assert stats['clip_scale'][1] == 1
    with pytest.raises(ValueError, match='float32'):
        model(z, target, delta.bfloat16())


class Context:
    def __init__(self, delta, *, clean=True, prefill=False, enabled=True):
        self.ffn_fast = {3: delta}
        self.ffn_enabled, self.clean_mode, self.prefill_mode = enabled, clean, prefill
        self.ffn_delta_clip = None
        self.staged = []

    def token_masks(self, n):
        mask = torch.ones(self.ffn_fast[3].shape[0], n, dtype=torch.bool)
        return mask, mask

    def stage_ffn(self, *values):
        self.staged.append(values)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_cached_ffn_old_behavior_and_checkpoint_single_stage(cached_ffn, checkpointed):
    torch.manual_seed(9)
    old = cached_ffn(3, 5, norm=(None, None, None))
    adapted = copy.deepcopy(old)
    adapted.inplace_ttt = InPlaceFFNTTT(3, 5)
    adapted.ttn_ffn_index = 3
    adapted.ttn_activation_checkpointing = checkpointed
    x = torch.randn(2, 8, 3, requires_grad=True)
    ctx = Context(torch.zeros(2, 3, 5))
    result = adapted(x, HW=(2, 2, 2), ttn_chunk_context=ctx)
    torch.testing.assert_close(result, old(x, HW=(2, 2, 2)), rtol=0, atol=0)
    assert len(ctx.staged) == 1
    (result.square().sum() + ctx.staged[0][1].sum()).backward()
    assert len(ctx.staged) == 1
    assert adapted.inplace_ttt.ttt_conv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('prefill,enabled', [(True, True), (False, False)])
def test_prefill_and_ablation_never_write(cached_ffn, prefill, enabled):
    torch.manual_seed(19)
    ffn = cached_ffn(3, 5, norm=(None, None, None))
    ffn.inplace_ttt = learned(); ffn.ttn_ffn_index = 3
    delta = torch.randn(2, 3, 5) * .01
    ctx = Context(delta, prefill=prefill, enabled=enabled)
    ffn(torch.randn(2, 8, 3), HW=(2, 2, 2), ttn_chunk_context=ctx)
    assert len(ctx.staged) == 1
    assert ctx.staged[0][1] is delta


def test_fp32_residual_survives_bf16_weight_rounding():
    model = InPlaceFFNTTT(2, 2)
    base_weight = torch.ones(2, 2, dtype=torch.bfloat16)
    delta = torch.full((1, 2, 2), 1e-5)
    z = torch.tensor([[[256., -256.]]], dtype=torch.bfloat16)
    delta[:, :, 1] = -1e-5
    rounded = F.linear(z, (base_weight.float() + delta[0]).bfloat16())
    correction, _, _ = model(z, torch.zeros(1, 1, 2), delta)
    residual = (F.linear(z, base_weight).float() + correction).bfloat16()
    assert torch.count_nonzero(rounded) == 0
    assert torch.count_nonzero(residual) == 2
