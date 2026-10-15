"""In-Place FFN state follows the actual observed/noisy/clean layer scan."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_alignment_detail import cached_sana
from test_sdpa_policy import text_attention
from test_legacy_sequence import cached_ffn, sequence_model
from test_training import inputs
from worldttn.core import ANCHORS
from worldttn.inplace_ttt import InPlaceFFNTTT
from worldttn.session import TTNSession
from worldttn.training import chunk_ranges, train_clip, linear_flow_loss
from worldttn.legacy_sequence_training import LegacySequenceSession, LegacySequenceTraining
from worldttn.runtime import TTNChunkContext


@pytest.fixture
def inplace_sequence_model(sequence_model):
    model = sequence_model
    config = replace(model.ttn_system.config, ttt_backend='inplace-ffn', state_ttt_rank=0,
                     local_update=False, persistent_meta=False, eta_psi=0., inplace_lr=.03)
    model.ttn_system.config = config
    if hasattr(model.ttn_system, 'local_eta_logits'):
        del model.ttn_system.local_eta_logits
    for index in ANCHORS:
        block = model.blocks[index]
        block.attn.config = config
        if hasattr(block.attn, 'state_ttt'):
            del block.attn.state_ttt
        block.mlp.ttn_ffn_index = index
        block.mlp.inplace_ttt = InPlaceFFNTTT(16, 32, lr=config.inplace_lr)
        # A learned, nonzero target tests recurrence; zero-init has separate algebra tests.
        torch.nn.init.normal_(block.mlp.inplace_ttt.ttt_conv.weight, std=.01)
    model.ttn_training_protocol = 'inplace-ffn-clean-fullgrad-v1'
    return model


def inplace_sequence_forward(model, clean, noisy, times, camera, *, cpu_inputs=False):
    b, _, frames, _, _ = clean.shape
    session = LegacySequenceSession(model, camera, 100, 100)
    runtime = session.reset(b)
    session.clean = clean
    state = runtime.world_state.repeat(3, 1, 1, 1, 1)
    psi = runtime.transition_fast.repeat(3, 1, 1, 1)
    ids = torch.arange(frames, device=clean.device).expand(3*b, -1)
    valid = torch.ones_like(ids, dtype=torch.bool)
    context = TTNChunkContext(state, state, psi, psi, model.ttn_system, ids,
        camera[:, :, :16].reshape(b, frames, 4, 4).repeat(3, 1, 1, 1),
        valid, valid, 0, id(runtime), sequence_mode=True, ffn_fast=runtime.ffn_fast,
        ffn_delta_clip=model.ttn_system.config.inplace_delta_clip, checkpoint_cpu_inputs=cpu_inputs)
    outputs = []
    def prediction_loss(session, clean, t, noise, y, ctx, cache, start, end, mask, info, lm, on_prediction):
        output, _ = session.forward(noisy, t, y, ctx, cache, start, end, mask, info)
        outputs.append(output)
        return output.square().flatten(1).mean(-1)
    episode = SimpleNamespace(session=session, context=context, clean=clean, timesteps=times, noise=noisy,
        y=torch.ones(b, 1, 2, 16, device=clean.device),
        mask=torch.ones(b, 2, dtype=torch.bool, device=clean.device),
        data_info=None, loss_mask=clean.new_ones(b, 1, frames, 1, 1), on_prediction=None)
    LegacySequenceTraining(model, prediction_loss)(episode)
    return outputs[0], context


@pytest.mark.parametrize('scaled', [False, True])
def test_inplace_layer_scan_matches_sequential_clean_transactions(inplace_sequence_model, scaled):
    model = inplace_sequence_model
    if scaled:
        model.ttn_system.config = replace(model.ttn_system.config,
            inplace_reduction='mean', inplace_delta_clip=.01, inplace_lr=3.)
        for index in ANCHORS:
            model.blocks[index].mlp.inplace_ttt.reduction = 'mean'
            model.blocks[index].mlp.inplace_ttt.lr = 3.
    reference = copy.deepcopy(model)
    clean, noisy, times, camera = inputs()
    noisy[:, :, :1] = clean[:, :, :1]
    y, mask = torch.ones(1, 1, 2, 16), torch.ones(1, 2, dtype=torch.bool)
    session = TTNSession(reference, camera, 100, 100, live_cache=True)
    session.reset(1)
    session.runtime.ffn_delta_clip = model.ttn_system.config.inplace_delta_clip
    session.prefill(clean[:, :, :1], y, mask)
    assert all(value.count_nonzero() == 0 for value in session.runtime.ffn_fast.values())
    cache, predictions = [[None]*10 for _ in model.blocks], []
    for start, end in chunk_ranges(clean.shape[2]):
        context = session.begin_chunk(start, end)
        prediction, _ = session.forward(noisy[:, :, start:end], times[:, :, start:end],
            y, context, cache, start, end, mask)
        predictions.append(prediction)
        _, cache = session.clean_forward(clean[:, :, start:end], y, context,
            cache, start, end, mask, cached_chunks=2)
    actual, context = inplace_sequence_forward(model, clean, noisy, times, camera)
    torch.testing.assert_close(actual, torch.cat(predictions, 2), rtol=3e-5, atol=3e-6)
    assert set(context.ffn_candidates) == set(ANCHORS)
    for index in ANCHORS:
        delta, records = context.ffn_candidates[index]
        torch.testing.assert_close(delta, session.runtime.ffn_fast[index], rtol=3e-5, atol=3e-6)
        assert records[0] is None
        assert len(records) == 1 + len(chunk_ranges(clean.shape[2]))


@pytest.mark.parametrize('cpu_inputs', [False, True])
@pytest.mark.parametrize('reduction', ['sum', 'mean'])
def test_inplace_checkpoint_preserves_full_history_credit(inplace_sequence_model, cpu_inputs, reduction):
    if reduction == 'mean':
        inplace_sequence_model.ttn_system.config = replace(inplace_sequence_model.ttn_system.config,
            inplace_reduction='mean', inplace_delta_clip=.01, inplace_lr=3.)
        for index in ANCHORS:
            module = inplace_sequence_model.blocks[index].mlp.inplace_ttt
            module.reduction, module.lr = 'mean', 3.
    clean, noisy, times, camera = inputs()
    indices = torch.cat((torch.arange(13), torch.arange(1, 13)))
    clean, noisy, times, camera = clean[:, :, indices], noisy[:, :, indices], times[:, :, indices], camera[:, indices]
    results = []
    for checkpointed in (False, True):
        model = copy.deepcopy(inplace_sequence_model)
        for block in model.blocks:
            block.ttn_fullgrad_checkpoint = checkpointed
            block.mlp.ttn_activation_checkpointing = checkpointed
        history = clean.clone().requires_grad_()
        output, context = inplace_sequence_forward(model, history, noisy, times, camera, cpu_inputs=cpu_inputs)
        output[:, :, -3:].square().mean().backward()
        assert history.grad[:, :, 1:4].norm() > 0
        assert history.grad[:, :, -3:].count_nonzero() == 0
        for index in ANCHORS:
            module = model.blocks[index].mlp.inplace_ttt
            assert module.ttt_conv.weight.grad is not None and module.ttt_conv.weight.grad.norm() > 0
            assert module.ttt_proj.weight.grad is not None and module.ttt_proj.weight.grad.norm() > 0
        results.append((output.detach(), history.grad,
                        [parameter.grad for parameter in model.parameters()],
                        {index: value[0].detach() for index, value in context.ffn_candidates.items()}))
    torch.testing.assert_close(results[0], results[1], rtol=5e-5, atol=5e-6)


def test_inplace_training_disables_psi_and_releases_fast_graph(inplace_sequence_model, monkeypatch):
    import json
    import worldttn.anchor as anchor
    from worldttn.anchor import configure_train_scope
    from worldttn.checkpoint import make_optimizer
    def forbidden(*args, **kwargs):
        raise AssertionError('inplace-ffn must not execute the legacy psi update')
    monkeypatch.setattr(anchor, 'analytic_psi_gradient', forbidden)
    monkeypatch.setattr(anchor, 'coefficient_gradient', forbidden)
    model = configure_train_scope(inplace_sequence_model, 'dit')
    clean, noisy, times, camera = inputs()
    result = train_clip(model, clean, torch.ones(1, 1, 2, 16), camera, make_optimizer(model),
        linear_flow_loss, times, noisy, width=100, height=100, tbptt=0, cached_chunks=2)
    assert result['sequence']['protocol'] == 'inplace-ffn-clean-fullgrad-v1'
    assert result['sequence']['ffn_clean_commits'] == len(chunk_ranges(clean.shape[2]))
    assert result['runtime'].ffn_commit_count == len(chunk_ranges(clean.shape[2]))
    assert result['sequence']['psi_updates'] is False
    assert result['runtime'].transition_fast.count_nonzero() == 0
    for value in result['runtime'].ffn_fast.values():
        assert value.grad_fn is None
    for chunk in result['chunks']:
        assert {row['block'] for row in chunk['ffn_ttt']} == set(ANCHORS)
        json.dumps(chunk['ffn_ttt'], allow_nan=False)
        for row in chunk['anchors']:
            assert row['persistent']['update_enabled'] is False
            assert row['local']['enabled'] is False
