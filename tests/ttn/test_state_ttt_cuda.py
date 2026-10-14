"""Native two-GPU FP8 plus state-TTT higher-order gradients and exact resume."""
import copy
import os
from pathlib import Path

import pytest
import torch
from torch import distributed as dist

from test_state_ttt import adapted_model, sequence_model, cached_sana, cached_ffn, text_attention
from test_training import inputs
from worldttn.core import ANCHORS
from worldttn.checkpoint import make_optimizer,load_checkpoint
from worldttn.distributed import ParallelTraining
from worldttn.fp8 import configure_training_precision,precision_identity
from worldttn.training import train_clip,linear_flow_loss

pytestmark=pytest.mark.skipif('STATE_TTT_ACCEPTANCE_OUTPUT' not in os.environ or not torch.cuda.is_available(),
    reason='isolated L20 two-GPU native FP8 acceptance required')


def test_two_gpu_state_ttt_fullgrad_save_and_resume(sequence_model):
    rank=int(os.environ['RANK']);device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device)
    try:
        assert dist.get_world_size()==2
        def build():
            m=adapted_model(copy.deepcopy(sequence_model).to(device))
            m.to(device)
            m.ttn_fullgrad_checkpoint=True
            for b in m.blocks:b.ttn_fullgrad_checkpoint=True
            report=configure_training_precision(m,'fp8')
            assert len(report['modules'])==100
            return m
        model=build();parallel=ParallelTraining(model,linear_flow_loss,'fsdp2',activation_offload='cpu-pageable')
        optimizer=make_optimizer(model)
        clean,noise,times,camera=[x.to(device) for x in inputs()];clean=clean+rank*.01
        def update(m,p,o):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                r=train_clip(m,clean,torch.zeros(1,1,2,16,device=device),camera,o,linear_flow_loss,times,noise,
                    width=100,height=100,tbptt=0,cached_chunks=2,parallel=p,activation_offload='cpu-pageable',audit_update=True)
            assert r['sequence']['backward_calls']==1 and r['sequence']['temporal_detaches']==0
            assert not r['optimizer_updates']['missing_core_gradients']
            for block in ANCHORS:
                for parameter in m.blocks[block].attn.state_ttt.parameters():
                    assert parameter.grad is not None
                    g=parameter.grad.to_local() if hasattr(parameter.grad,'to_local') else parameter.grad
                    assert torch.isfinite(g).all() and g.norm()>0
            return r
        update(model,parallel,optimizer);update(model,parallel,optimizer)
        output=Path(os.environ['STATE_TTT_ACCEPTANCE_OUTPUT'])
        identity={'state_ttt':'rank4-curvature-.5','compute_precision':precision_identity('fp8'),'tbptt':0}
        from worldttn.parallel_checkpoint import save_training_checkpoint,restore_training_checkpoint
        save_training_checkpoint(output/'last.pt',parallel,optimizer,2,{'position':2},identity)
        other=build();load_checkpoint(output/'last.pt',other)
        resumed=ParallelTraining(other,linear_flow_loss,'fsdp2',activation_offload='cpu-pageable');other_opt=make_optimizer(other)
        assert restore_training_checkpoint(output/'last.pt',resumed,other_opt,identity)==(2,{'position':2})
        a,b=update(model,parallel,optimizer),update(other,resumed,other_opt)
        assert a['loss']==pytest.approx(b['loss'],abs=1e-7)
        for (name,x),(_,y) in zip(model.named_parameters(),other.named_parameters()):
            x=x.to_local() if hasattr(x,'to_local') else x;y=y.to_local() if hasattr(y,'to_local') else y
            torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-7,msg=name)
        print('STATE_TTT_FP8_ACCEPTANCE_PASSED',rank,flush=True)
    finally:dist.destroy_process_group()
