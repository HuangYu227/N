"""State-TTT mathematical gradients, full credit and checkpoint identity."""
import copy
from dataclasses import replace

import pytest
import torch

from test_legacy_sequence import sequence_model, cached_sana, cached_ffn, text_attention
from test_training import inputs
from worldttn.core import ANCHORS, TTNConfig
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint
from worldttn.training import train_clip, linear_flow_loss


def adapted_model(model,policy='curvature'):
    from worldttn.runtime import TTNSystem
    from worldttn.state_ttt import StateTTT
    config=replace(model.ttn_system.config,state_ttt_rank=4,state_ttt_policy=policy,
                   state_ttt_gain=.5 if policy=='curvature' else 1.)
    model.ttn_system=TTNSystem(config)
    for index,block in enumerate(ANCHORS):
        model.blocks[block].attn.config=config
        model.blocks[block].attn.state_ttt=StateTTT(config,index)
    return configure_train_scope(model,'dit')


def test_default_config_preserves_legacy_identity_and_rejects_invalid_state_rank():
    assert not any(k.startswith('state_ttt') for k in TTNConfig().to_dict())
    with pytest.raises(ValueError):TTNConfig(stage='C',state_ttt_rank=113)
    with pytest.raises(ValueError):TTNConfig(stage='C',state_ttt_rank=16,state_ttt_gain=2.)


@pytest.mark.parametrize('policy',['fixed','curvature'])
def test_state_update_is_live_and_descends_support_objective(policy):
    from worldttn.state_ttt import StateTTT
    config=TTNConfig(heads=2,head_dim=8,generators=3,stage='C',state_ttt_rank=4,
                    state_ttt_policy=policy,state_ttt_gain=.5)
    m=StateTTT(config,0).double()
    torch.manual_seed(191)
    s=torch.randn(1,2,8,8,dtype=torch.float64,requires_grad=True)*.1
    k,v=[torch.randn(1,2,6,8,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    w=torch.ones(1,2,6,dtype=torch.float64);mask=torch.ones(1,6,dtype=torch.bool)
    out,stats=m(s,k,v,w,mask,clean=False,eps=1e-6,clip=1.)
    assert out.dtype==torch.float64 and (stats['loss_after']<=stats['loss_before']).all()
    basis_grad,eta_grad=torch.autograd.grad(out.square().mean(),(m.raw_basis,m.step_logits),create_graph=True)
    assert basis_grad.norm()>0 and eta_grad.norm()>0
    second=torch.autograd.grad(basis_grad.square().sum(),s)[0]
    assert torch.isfinite(second).all()
    empty,_=m(s,k,v,w,mask&False,clean=False,eps=1e-6,clip=1.)
    torch.testing.assert_close(empty,s,rtol=0,atol=0)


@pytest.mark.parametrize('policy',['fixed','curvature'])
def test_complete_sequence_backward_and_exact_resume(sequence_model,tmp_path,policy):
    model=adapted_model(copy.deepcopy(sequence_model),policy)
    optimizer=make_optimizer(model)
    clean,noise,times,camera=inputs()
    def update(m,o):
        return train_clip(m,clean,torch.zeros(1,1,2,16),camera,o,linear_flow_loss,times,noise,
            width=100,height=100,tbptt=0,cached_chunks=2,audit_update=True)
    result=update(model,optimizer)
    assert result['sequence']['backward_calls']==1 and result['sequence']['temporal_detaches']==0
    assert not result['optimizer_updates']['missing_core_gradients']
    signals=result['optimizer_updates']['state_ttt_parameters']
    assert len(signals)==10
    assert all(row['grad_norm']>0 and row['delta_norm']>0 and row['optimizer_state'] for row in signals.values())
    assert not hasattr(model.ttn_system,'local_eta_logits')
    assert torch.count_nonzero(result['runtime'].transition_fast)==0
    for block in ANCHORS:
        module=model.blocks[block].attn.state_ttt
        assert module.raw_basis.grad is not None and module.raw_basis.grad.norm()>0
        assert module.step_logits.grad is not None and module.step_logits.grad.norm()>0
        assert all('state_ttt' in row['anchors'][ANCHORS.index(block)] for row in result['chunks'])
    path=tmp_path/'state.pt';save_checkpoint(path,model,optimizer,1)
    other=adapted_model(copy.deepcopy(sequence_model),policy);other_opt=make_optimizer(other)
    assert load_checkpoint(path,other,other_opt,resume=True)==1
    a,b=update(model,optimizer),update(other,other_opt)
    assert a['loss']==b['loss']
    for left,right in zip(model.parameters(),other.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    legacy=copy.deepcopy(sequence_model)
    with pytest.raises(ValueError,match='identity'):load_checkpoint(path,legacy)


def test_first_update_guard_reports_disconnected_state_parameters(sequence_model):
    from worldttn.training_health import FirstUpdateProbe
    model=adapted_model(copy.deepcopy(sequence_model))
    report=FirstUpdateProbe(model).report(make_optimizer(model))
    expected={f'blocks.{i}.attn.state_ttt.{p}' for i in ANCHORS for p in ('raw_basis','step_logits')}
    assert expected<=set(report['missing_core_gradients'])
    assert set(report['state_ttt_parameters'])==expected
    assert all(row['grad_norm'] is None and not row['optimizer_state'] for row in report['state_ttt_parameters'].values())


@pytest.mark.parametrize('clean', [False, True])
def test_curvature_update_matches_independent_autograd_oracle(clean):
    from worldttn.state_ttt import StateTTT
    torch.manual_seed(37)
    config = TTNConfig(heads=1, head_dim=8, stage='C', state_ttt_rank=4,
                       state_ttt_gain=.5)
    module = StateTTT(config, 0).double()
    state = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    keys, values = [torch.randn(1, 1, 7, 8, dtype=torch.float64) for _ in range(2)]
    weights = torch.rand(1, 1, 7, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, True, False, True, True]])
    basis = torch.linalg.qr(module.raw_basis).Q.detach()
    factors = torch.zeros(1, 1, 4, 8, dtype=torch.float64, requires_grad=True)
    selected = mask[:, None, :]
    weights = torch.where(selected, weights, 0.)
    denominator = (2 * 8 * mask.sum(-1)[:, None] if clean else
                   1e-6 + (weights * values.square().sum(-1)).sum(-1))

    def objective(b):
        residual = keys @ (state + basis @ b) - values
        return ((weights * residual.square().sum(-1)).sum(-1) / denominator).sum()

    gradient = torch.autograd.grad(objective(factors), factors)[0]
    # One value column has the same curvature matrix as every other column.
    def column_objective(column):
        b = torch.cat((column.reshape(1, 1, 4, 1), factors.detach()[..., 1:]), -1)
        return objective(b)

    hessian = torch.autograd.functional.hessian(column_objective, torch.zeros(4, dtype=torch.float64))
    gain = 2 * module.step_logits[int(clean)].sigmoid().detach()
    eta = gain / (hessian.trace() + 1e-6)
    expected = state - eta[..., None, None] * (basis @ gradient)
    actual, stats = module(state, keys, values, weights, mask, clean=clean, eps=1e-6, clip=1e6)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(stats['curvature_trace'], hessian.trace().reshape(1, 1))


def test_real_head_dimensions_write_content_from_zero_state():
    from worldttn.state_ttt import StateTTT
    config = TTNConfig(stage='C', state_ttt_rank=64, state_ttt_gain=.5)
    module = StateTTT(config, 4)
    torch.manual_seed(51)
    state = torch.zeros(1, 20, 112, 112)
    keys, values = [torch.randn(1, 20, 5, 112) for _ in range(2)]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual, stats = module(state, keys, values, torch.ones(1, 20, 5),
                               torch.ones(1, 5, dtype=torch.bool), clean=False, eps=1e-6, clip=1.)
    assert actual.dtype == torch.float32 and actual.norm() > 0
    assert all(torch.isfinite(value).all() for value in stats.values())
    actual.square().mean().backward()
    assert module.raw_basis.grad.norm() > 0 and module.step_logits.grad.norm() > 0


def test_reused_basis_preserves_noisy_clean_outputs_and_gradients():
    from worldttn.state_ttt import StateTTT
    config = TTNConfig(heads=2, head_dim=8, stage='C', state_ttt_rank=4,
                       state_ttt_gain=.5)
    torch.manual_seed(27)
    baseline = StateTTT(config, 0).double()
    reused = copy.deepcopy(baseline)
    state = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    keys, values = [torch.randn(1, 2, 6, 8, dtype=torch.float64) for _ in range(2)]
    weights = torch.rand(1, 2, 6, dtype=torch.float64)
    mask = torch.ones(1, 6, dtype=torch.bool)

    def two_calls(module, basis=None):
        local = module(state, keys, values, weights, mask, clean=False,
                       eps=1e-6, clip=1., basis=basis)[0]
        clean = module(local, keys, values, weights, mask, clean=True,
                       eps=1e-6, clip=1., basis=basis)[0]
        return clean.square().mean()

    old = two_calls(baseline)
    new = two_calls(reused, reused.orthogonal_basis(torch.float64))
    torch.testing.assert_close(old, new, rtol=0, atol=0)
    old.backward(); new.backward()
    for left, right in zip(baseline.parameters(), reused.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-12)


def test_batched_state_telemetry_preserves_values_and_detaches():
    from worldttn.state_ttt import state_stats
    values = {
        "effective_eta": torch.tensor([[.2, .5]], requires_grad=True),
        "clip_scale": torch.tensor([[[[1.]], [[.25]]]], requires_grad=True),
        "gradient_norm": torch.tensor([[3., 5.]], requires_grad=True),
    }
    report = state_stats(values, clean=True, policy="curvature")
    assert report["branch"] == "clean"
    assert report["clip_fraction"] == .5
    assert report["effective_eta_min"] == pytest.approx(.2)
    assert report["effective_eta_max"] == pytest.approx(.5)
    for name, value in values.items():
        assert report[name] == pytest.approx(float(value.detach().double().mean()))
        assert report["per_head"][name] == value.detach().tolist()
