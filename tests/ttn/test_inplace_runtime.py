import copy
from dataclasses import replace

import pytest
import torch

from test_anchor import ProjectionContract
from test_meta_runtime import begin
from worldttn.anchor import TTNAnchor, install_ttn, configure_train_scope, is_ttn_new_parameter, offline_state_dict
from worldttn.core import ANCHORS, TTNConfig
from worldttn.runtime import TTNRuntimeState, TTNSystem


def episode():
    cfg = TTNConfig(heads=2, head_dim=8, generators=3, stage='C', eta_psi=0,
                    ttt_backend='inplace-ffn')
    system = TTNSystem(cfg)
    runtime = TTNRuntimeState.create(cfg, 2, 'cpu')
    return system, runtime, [TTNAnchor(ProjectionContract(), i, cfg) for i in range(5)]


def stage_all(system, runtime, anchors):
    context = begin(system, runtime).for_clean()
    for i, anchor in enumerate(anchors):
        anchor(torch.randn(2, 8, 16), HW=(2, 2, 2), ttn_chunk_context=context)
        block = ANCHORS[i]
        context.stage_ffn(block, context.ffn_fast[block] + (i+1)*.001, {})
    return context


def test_config_identity_roundtrip_and_no_legacy_backend_activation():
    legacy = TTNConfig(stage='C').to_dict()
    assert not any(key.startswith('inplace') or key == 'ttt_backend' for key in legacy)
    cfg = episode()[0].config
    assert TTNConfig(**cfg.to_dict()) == cfg
    for flags in ({'state_ttt_rank': 1}, {'local_update': True}, {'persistent_meta': True}, {'eta_psi': .1}):
        with pytest.raises(ValueError, match='replaces'):
            replace(cfg, **flags)


def test_scaled_update_config_is_shared_by_train_and_inference_without_changing_old_identities():
    system, _, _ = episode()
    legacy = system.config
    assert 'inplace_reduction' not in legacy.to_dict() and 'inplace_delta_clip' not in legacy.to_dict()
    assert legacy.inference_ffn_clip() == 1e-5
    cfg = replace(legacy, inplace_reduction='mean', inplace_delta_clip=.01)
    assert TTNConfig(**cfg.to_dict()) == cfg
    runtime = TTNRuntimeState.create(cfg, 2, 'cpu')
    assert runtime.ffn_delta_clip == cfg.inplace_delta_clip == cfg.inference_ffn_clip()
    assert TTNRuntimeState.create(cfg, 2, 'cpu', ffn_delta_clip=0).ffn_delta_clip is None
    assert cfg.inference_ffn_clip(.02) == .02
    for flags in ({'inplace_reduction': 'invalid'}, {'inplace_delta_clip': -1}, {'inplace_delta_clip': float('nan')}):
        with pytest.raises(ValueError): replace(cfg, **flags)


def test_clean_commits_ffn_atomically_and_never_calls_legacy_psi(monkeypatch):
    monkeypatch.setattr('worldttn.anchor.analytic_psi_gradient', lambda *a, **k: pytest.fail('legacy psi called'))
    system, runtime, anchors = episode()
    context = stage_all(system, runtime, anchors)
    assert all(not hasattr(a, 'state_ttt') for a in anchors)
    runtime.commit_chunk(context)
    assert runtime.ffn_commit_count == 1
    assert not runtime.transition_fast.count_nonzero()
    assert runtime.ffn_fast[3].data_ptr() != runtime.ffn_fast[7].data_ptr()
    assert not torch.equal(runtime.ffn_fast[3], runtime.ffn_fast[7])
    with pytest.raises(RuntimeError, match='current clean'):
        runtime.commit_chunk(context)


def test_bad_ffn_candidate_does_not_commit_s_or_other_ffns():
    system, runtime, anchors = episode()
    context = stage_all(system, runtime, anchors)
    before_s, before_ffn = runtime.world_state.clone(), copy.deepcopy(runtime.ffn_fast)
    bad = context.ffn_candidates[19][0].clone()
    bad[1, 0, 0] = float('nan')
    context.ffn_candidates[19] = (bad, {})
    with pytest.raises(ValueError, match='invalid FFN'):
        runtime.commit_chunk(context)
    assert runtime.commit_count == runtime.ffn_commit_count == 0
    torch.testing.assert_close(runtime.world_state, before_s)
    for i, value in before_ffn.items(): torch.testing.assert_close(runtime.ffn_fast[i], value)


def test_ffn_transaction_is_invalidated_by_another_begin_and_for_clean_preserves_options():
    system, runtime, anchors = episode()
    context = stage_all(system, runtime, anchors)
    context.ffn_enabled, context.ffn_delta_clip = False, None
    clean = context.for_clean()
    assert not clean.ffn_enabled and clean.ffn_delta_clip is None and not clean.ffn_candidates
    assert clean.ffn_fast is context.ffn_fast
    begin(system, runtime)
    with pytest.raises(RuntimeError, match='superseded'):
        runtime.commit_chunk(context)


def test_runtime_reset_isolates_cfg_samples_and_layers():
    _, runtime, _ = episode()
    runtime.ffn_fast[3][0].fill_(1)
    assert not runtime.ffn_fast[3][1].count_nonzero()
    assert not runtime.ffn_fast[7].count_nonzero()
    runtime.reset()
    assert not any(value.count_nonzero() for value in runtime.ffn_fast.values())
    assert runtime.ffn_commit_count == 0
    assert is_ttn_new_parameter('blocks.3.mlp.inplace_ttt.ttt_proj.weight')
    assert not is_ttn_new_parameter('blocks.3.mlp.point_conv.conv.weight')


def test_install_preserves_pretrained_ffn_and_exports_new_targets(tmp_path):
    from torch import nn
    from worldttn.checkpoint import save_checkpoint, load_checkpoint, make_optimizer

    def base():
        model = nn.Module()
        model.blocks = nn.ModuleList()
        for _ in range(20):
            block = nn.Module()
            block.attn = ProjectionContract()
            block.mlp = nn.Module()
            block.mlp.point_conv = nn.Module()
            block.mlp.point_conv.conv = nn.Conv2d(32, 16, 1)
            model.blocks.append(block)
        return model

    torch.manual_seed(41)
    original = base()
    inherited = {name: value.clone() for name, value in original.state_dict().items() if '.mlp.' in name}
    cfg = episode()[0].config
    model = configure_train_scope(install_ttn(original, cfg), 'dit')
    for name, value in inherited.items(): torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    for index in ANCHORS:
        assert model.blocks[index].mlp.ttn_ffn_index == index
        assert not model.blocks[index].mlp.inplace_ttt.ttt_conv.weight.count_nonzero()
    assert sum('.mlp.inplace_ttt.' in name for name in offline_state_dict(model)) == 10
    optimizer = make_optimizer(model)
    new_names = {id(p) for group in optimizer.param_groups if group['name'] == 'ttn_new' for p in group['params']}
    assert all(id(p) in new_names for i in ANCHORS for p in model.blocks[i].mlp.inplace_ttt.parameters())
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, optimizer, 2)
    other = configure_train_scope(install_ttn(base(), cfg), 'dit')
    assert load_checkpoint(path, other, make_optimizer(other), resume=True) == 2
    for name, value in model.state_dict().items(): torch.testing.assert_close(other.state_dict()[name], value)
