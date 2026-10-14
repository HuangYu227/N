"""Native FP8 on two GPUs: actual GEMMs, live dual-psi gradients and exact resume."""
import copy
import json
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch import nn

from test_training import inputs
from test_legacy_sequence import sequence_model, cached_sana, cached_ffn, text_attention
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer, load_checkpoint
from worldttn.distributed import ParallelTraining
from worldttn.fp8 import configure_training_precision, precision_identity
from worldttn.training import train_clip, linear_flow_loss

pytestmark = [pytest.mark.skipif('FP8_ACCEPTANCE_OUTPUT' not in os.environ or not torch.cuda.is_available(),
                               reason='requires isolated native two-GPU FP8 acceptance'),
              pytest.mark.filterwarnings('ignore:torch.get_autocast_gpu_dtype.*is deprecated:DeprecationWarning')]


def test_native_fp8_projection_contract():
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    m = nn.Module(); block = nn.Module(); block.cross_attn = nn.Module()
    block.cross_attn.q_linear = nn.Linear(256, 256, device=device)
    m.blocks = nn.ModuleList([block])
    baseline = copy.deepcopy(block.cross_attn.q_linear)
    report = configure_training_precision(m, 'fp8')
    x = torch.randn(64, 256, device=device, requires_grad=True)
    control = x.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        expected = baseline(control)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            actual = block.cross_attn.q_linear(x)
            actual.square().mean().backward()
        expected.square().mean().backward()
    assert actual.dtype == torch.bfloat16
    names = {event.key for event in profile.key_averages()}
    assert any('_scaled_mm' in name for name in names), names
    differences = {}
    for name, a, b in [('output', actual, expected), ('input_gradient', x.grad, control.grad),
                       ('weight_gradient', block.cross_attn.q_linear.weight.grad, baseline.weight.grad)]:
        assert torch.isfinite(a).all()
        ratio = (a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)
        differences[name] = float(ratio.detach())
        assert ratio < .15, (name, ratio)
    print('[FP8 native] '+json.dumps({'recipe':report['recipe'], 'relative_differences':differences,
                                     'quality_claim':False}), flush=True)


def test_real_width_ffn_fp8_forward_backward(cached_ffn):
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    m=nn.Module();block=nn.Module();block.mlp=cached_ffn(2240,6720).to(device)
    m.blocks=nn.ModuleList([block])
    reference=copy.deepcopy(block.mlp)
    report=configure_training_precision(m,'fp8')
    assert len(report['modules'])==2
    x=torch.randn(1,2640,2240,device=device,requires_grad=True)
    other=x.detach().clone().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=block.mlp(x,HW=(3,22,40))
        expected=reference(other,HW=(3,22,40))
        actual.float().square().mean().backward()
        expected.float().square().mean().backward()
    differences={}
    for name,a,b in [('output',actual,expected),('input_gradient',x.grad,other.grad)]:
        assert torch.isfinite(a).all()
        differences[name]=float(((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)).detach())
        assert differences[name]<.2,(name,differences[name])
    assert all(torch.isfinite(p.grad).all() for p in block.mlp.parameters() if p.grad is not None)
    print('[FP8 real FFN] '+json.dumps({'width':2240,'hidden':6720,'rows':2640,
                                      'relative_differences':differences,'quality_claim':False}),flush=True)


def test_two_gpu_dual_psi_checkpoint_and_resume(sequence_model):
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', init_method='env://', device_id=device)
    try:
        assert dist.get_world_size() == 2
        def build(fp8=True):
            model = configure_train_scope(copy.deepcopy(sequence_model).to(device), 'dit')
            model.ttn_fullgrad_checkpoint = True
            for block in model.blocks: block.ttn_fullgrad_checkpoint = True
            if fp8:
                report = configure_training_precision(model, 'fp8')
                assert len(report['modules']) == 100
            return model
        model = build()
        parallel = ParallelTraining(model, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        optimizer = make_optimizer(model)
        clean, noise, times, camera = [value.to(device) for value in inputs()]
        clean = clean+rank*.01
        def update(m, opt, par):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = train_clip(m, clean, torch.zeros(1,1,2,16,device=device), camera, opt,
                    linear_flow_loss, times, noise, width=100, height=100, tbptt=0,
                    cached_chunks=2, parallel=par, activation_offload='cpu-pageable', audit_update=True)
            assert result['sequence']['backward_calls'] == 1
            assert result['sequence']['temporal_detaches'] == 0
            assert not result['optimizer_updates']['missing_core_gradients']
            assert result['runtime'].world_state.dtype == torch.float32
            assert result['runtime'].transition_fast.dtype == torch.float32
            for p in m.parameters():
                if p.grad is not None:
                    value=p.grad.to_local() if hasattr(p.grad,'to_local') else p.grad
                    assert torch.isfinite(value).all()
            return result
        update(model, optimizer, parallel)
        update(model, optimizer, parallel)
        output = Path(os.environ['FP8_ACCEPTANCE_OUTPUT'])
        identity={'tbptt':0, 'compute_precision':precision_identity('fp8')}
        from worldttn.parallel_checkpoint import save_training_checkpoint, restore_training_checkpoint
        save_training_checkpoint(output/'last.pt', parallel, optimizer, 2, {'position':2}, identity)
        ordinary = build(False)
        # A checkpoint can also initialize a normal BF16-compute inference model.
        assert load_checkpoint(output/'last.pt', ordinary) == 0
        configure_training_precision(ordinary, 'fp8')
        resumed = ParallelTraining(ordinary, linear_flow_loss, 'fsdp2', activation_offload='cpu-pageable')
        resumed_optimizer=make_optimizer(ordinary)
        assert restore_training_checkpoint(output/'last.pt', resumed, resumed_optimizer, identity) == (2,{'position':2})
        a=update(model,optimizer,parallel); b=update(ordinary,resumed_optimizer,resumed)
        assert a['loss'] == pytest.approx(b['loss'],abs=1e-7)
        for (name,p),(_,q) in zip(model.named_parameters(), ordinary.named_parameters()):
            p=p.to_local() if hasattr(p,'to_local') else p
            q=q.to_local() if hasattr(q,'to_local') else q
            torch.testing.assert_close(p,q,rtol=3e-5,atol=3e-7,msg=name)
        if rank == 0:
            print('[FP8 acceptance] '+json.dumps({'status':'passed','world_size':2,
                'loss':a['loss'],'commits':a['runtime'].commit_count,
                'saved_step':2,'real_model_capacity_claim':False,'quality_claim':False}),flush=True)
    finally:
        dist.destroy_process_group()
