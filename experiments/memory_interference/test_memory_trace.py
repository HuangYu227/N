"""Exercise hooks against actual production anchors, StateTTT and transactions."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT if (ROOT/'worldttn').is_dir() else ROOT/'ttn-dual-psi-fp8'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(SOURCE / 'tests/ttn'))
from test_training import TinyWorldModel, inputs
from worldttn.core import ANCHORS
from worldttn.session import TTNSession
from memory_math import check_clean_pair, clone_tree
from memory_trace import MemoryTrace
from run_diagnostic import state_identity, validate_config


@torch.no_grad()
def episode(mode):
    torch.manual_seed(314)
    model = TinyWorldModel(local_update=True, persistent_meta=True,
                           state_ttt_rank=4, state_ttt_gain=.5, eta_psi=0).eval()
    model.requires_grad_(False)
    clean, noise, times, camera = inputs()
    # Multiple spatial tokens exercise the actual checkerboard support/query split.
    clean = clean[:, :, :7].expand(-1, -1, -1, 2, 2).contiguous()
    y = torch.zeros(1, 1, 2, 16)
    session = TTNSession(model, camera[:, :7], 2, 2, ablation=mode)
    session.reset(1)
    trace = MemoryTrace(model, ANCHORS, 3)
    trace.phase = 'prefill'
    session.prefill(clean[:, :, :1], y)
    result = dict(prefix_hashes=list(trace.prefix_hashes.values()), clean_state_hashes=[])
    cumulative = torch.eye(8).expand_as(session.runtime.world_state).clone()
    cache = [[None] * 10 for _ in model.blocks]
    try:
        for chunk, (start, end) in enumerate(((0, 4), (4, 7)), 1):
            context = session.begin_chunk(start, end)
            trace.phase, trace.chunk = 'noisy', chunk
            rotation = context.live_factors.right(torch.eye(8).expand_as(cumulative))
            cumulative = cumulative @ rotation
            trace.transport = cumulative
            before = state_identity(session.runtime)
            session.forward(clean[:, :, start:end] + .1, times[:, :, start:end], y,
                            context, clone_tree(cache), start, end)
            assert state_identity(session.runtime) == before
            trace.phase = 'clean'
            _, cache = session.clean_forward(clean[:, :, start:end], y, context, cache,
                                              start, end, cached_chunks=2)
            result['clean_state_hashes'].append(state_identity(session.runtime))
        result.update(rows=trace.rows, calls=dict(trace.calls))
    finally:
        trace.close()
    assert not any(model.blocks[i].attn._forward_pre_hooks for i in ANCHORS)
    return result


def test_actual_model_four_modes_execute_expected_branches_and_keep_clean_history_isolated():
    records = {mode: episode(mode) for mode in ('full', 'no-local', 'no-persistent', 'no-ttt')}
    check_clean_pair(records['full'], records['no-local'])
    check_clean_pair(records['no-persistent'], records['no-ttt'])
    assert records['full']['calls'] == {'clean': 10, 'local': 10}
    assert records['no-local']['calls'] == {'clean': 10, 'local': 0}
    assert records['no-persistent']['calls'] == {'clean': 0, 'local': 10}
    assert records['no-ttt']['calls'] == {'clean': 0, 'local': 0}
    assert len(records['full']['rows']) == 15
    active = [row for row in records['full']['rows'] if row['chunk']]
    assert all(row['inner']['loss_after'] <= row['inner']['loss_before'] + 1e-7 for row in active)
    assert any(abs(row['ttt_old_binding_delta']) > 1e-8 for row in active)
    assert all(row['ttt_old_binding_delta'] == 0 for row in records['no-ttt']['rows'] if row['chunk'])


def test_wrong_checkpoint_family_is_rejected_before_running_ablation():
    from worldttn.core import TTNConfig
    with pytest.raises(ValueError, match='direct State-TTT'):
        validate_config(TTNConfig(stage='C'))


@torch.no_grad()
def test_production_head_dimension_and_state_rank_preserve_metric_contract():
    from worldttn.core import TTNConfig, correct, write_weights
    from worldttn.state_ttt import StateTTT
    from memory_math import make_reference, retention_metrics
    config = TTNConfig(stage='C', local_update=True, persistent_meta=True,
                       state_ttt_rank=64, state_ttt_gain=.5, eta_psi=0)
    validate_config(config)
    torch.manual_seed(270)
    module = StateTTT(config, 0)
    q, k, v = [torch.randn(1, 20, 128, 112)*.1 for _ in range(3)]
    state = torch.randn(1, 20, 112, 112)*.01
    selected = torch.ones(1, 128, dtype=torch.bool)
    beta = torch.full((1, 20, 128), .5)
    weights = write_weights(beta, selected, config.eps)
    updated, stats = module(state, k, v, weights, selected, clean=True,
                            eps=config.eps, clip=config.inner_clip)
    assert torch.isfinite(updated).all()
    assert (stats['loss_after'] <= stats['loss_before'] + 1e-7).all()
    candidate = correct(updated, k, v, beta, selected, config.alpha_s, config.eps)[0]
    metrics = retention_metrics(candidate, make_reference(q, k, v, state), torch.eye(112))
    assert metrics['old_query_response_drift'] > 0
    assert len(metrics['per_token_key_nmse']) == 64
