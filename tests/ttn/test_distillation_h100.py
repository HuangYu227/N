"""Portable recipe, accumulation and launcher failure/recovery checks."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from worldttn.core import TTNConfig
from worldttn.distillation.adapters import GradientAccumulator, optimizer, clip_grad, parameters
from worldttn.distillation.checkpoint import EMA, save, restore, digest
from worldttn.distillation.recipe import allocation_budget, teacher_identity
from tools.distillation import launch_h100


def teacher_file(path,step=300):
    settings=TTNConfig(stage='C',state_ttt_rank=64,state_ttt_policy='curvature',eta_psi=0,
                       local_update=True,persistent_meta=True)
    torch.save({'format':'TTN-SANA-WM-v0.1','step':step,'train_scope':'dit','weight_scope':'dit',
                'config':settings.to_dict(),'base_sha256':'base-sha',
                'distributed':{'checkpoint_id':'checkpoint-300'},'adapter':{}},path)


def test_teacher_snapshot_step_hash_and_original_independence(tmp_path):
    source=tmp_path/'last.pt'; teacher_file(source)
    pinned=tmp_path/'teacher/model.pt'
    identity=launch_h100.pin_teacher(source,pinned)
    assert identity['step']==300 and identity['sha256']==digest(pinned)
    teacher_file(source,325)
    assert teacher_identity(pinned,300,identity['sha256'],'checkpoint-300')[0]['step']==300
    with pytest.raises(ValueError,match='step300'): teacher_identity(source,300)
    with pytest.raises(ValueError,match='SHA256'): teacher_identity(pinned,300,'incorrect')
    with pytest.raises(ValueError,match='checkpoint ID'): teacher_identity(pinned,300,checkpoint_id='wrong')
    with pytest.raises(FileExistsError): launch_h100.pin_teacher(source,pinned)


def test_find_teacher_does_not_substitute_latest_step(tmp_path):
    teacher_file(tmp_path/'last.pt',325)
    with pytest.raises(ValueError,match='No saved step300'): launch_h100.find_teacher(tmp_path)
    bundle=tmp_path/'last-resume-00000300-test';bundle.mkdir()
    teacher_file(bundle/'model.pt')
    assert launch_h100.find_teacher(tmp_path)==bundle/'model.pt'


def test_budget_is_bytes_not_l20_fraction():
    assert allocation_budget(80*2**30,70,8)==70*2**30
    assert allocation_budget(76*2**30,70,8)==68*2**30
    for budget in (0,-1,float('nan')):
        with pytest.raises(ValueError): allocation_budget(80*2**30,budget,8)


def test_accumulated_gradient_matches_average_and_exact_next_update(tmp_path):
    torch.manual_seed(31)
    model=nn.Sequential(nn.Linear(4,4),nn.Tanh(),nn.Linear(4,2))
    reference=copy.deepcopy(model)
    inputs=[torch.randn(2,4),torch.randn(2,4)]
    opt=optimizer(model,1e-4); ema=EMA(model)
    def update():
        opt.zero_grad(set_to_none=True)
        gradients=GradientAccumulator(model)
        for value in inputs:
            (model(value).square().mean()/2).backward()
            gradients.capture()
            assert all(p.grad is None for p in model.parameters())
        gradients.apply()
        clip_grad(model);opt.step();ema.update(model)
    sum(reference(value).square().mean()/2 for value in inputs).backward()
    gradients=GradientAccumulator(model)
    for value in inputs:
        (model(value).square().mean()/2).backward();gradients.capture()
    gradients.apply()
    for p,q in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-6,atol=1e-7)
    update()
    models={'student':model,'teacher':copy.deepcopy(model).requires_grad_(False)}
    save(tmp_path,models,{'student':opt},ema,{'student':1,'critic':5},{'effective_batch':2})
    update()
    expected={n:p.detach().clone() for n,p in parameters(model).items()}
    expected_ema={n:p.clone() for n,p in ema.shadow.items()}
    restore(tmp_path,models,{'student':opt},ema,{'effective_batch':2})
    update()
    for name,p in parameters(model).items():
        torch.testing.assert_close(p,expected[name],rtol=0,atol=0)
        torch.testing.assert_close(ema.shadow[name],expected_ema[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='identity changed'):
        restore(tmp_path,models,{'student':opt},ema,{'effective_batch':1})


def prepared_run(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    settings=OmegaConf.create({'code_sha256':'code','capacity_proof':str(run/'capacity/capacity-passed.json')})
    OmegaConf.save(settings,run/'config.yaml')
    (run/'launch.json').write_text(json.dumps({'config_sha256':digest(run/'config.yaml'),
        'resources':{'memory_scope_gib':128,'gpu_uuid':'gpu5'}}))
    monkeypatch.setattr(launch_h100,'code_identity',lambda root:{'sha256':'code'})
    return run


def test_capacity_failure_never_starts_formal_job(tmp_path,monkeypatch):
    run=prepared_run(tmp_path,monkeypatch)
    commands=[]
    class Child:
        def __init__(self,command,**kwargs): commands.append(command)
        def wait(self): return 1 if '--capacity-only' in commands[-1] else 0
    monkeypatch.setattr(launch_h100.subprocess,'Popen',Child)
    assert launch_h100.execute(run)==1
    assert len(commands)==2 and '--capacity-only' in commands[-1]
    state=json.loads((run/'status.json').read_text())
    assert state['status']=='failed' and state['phase']=='capacity'
    assert not (run/'runner.lock').exists()


def test_capacity_success_then_fresh_training_and_resume_skip(tmp_path,monkeypatch):
    run=prepared_run(tmp_path,monkeypatch)
    commands=[]
    class Child:
        def __init__(self,command,**kwargs): commands.append(command)
        def wait(self):
            if '--capacity-only' in commands[-1]:
                (run/'capacity/run-config.json').write_text('{"identity":{"test":1}}')
                (run/'capacity/capacity-passed.json').write_text('{"test":1}')
            return 0
    monkeypatch.setattr(launch_h100.subprocess,'Popen',Child)
    assert launch_h100.execute(run)==0
    assert len(commands)==3 and '--capacity-only' not in commands[-1]
    assert str(run/'capacity') in commands[1] and str(run/'training') in commands[2]
    (run/'capacity/capacity-passed.json').write_text('{}')
    monkeypatch.setattr(launch_h100,'resources',lambda *args:{'memory_scope_gib':128,'gpu_uuid':'gpu5'})
    commands.clear()
    assert launch_h100.execute(run,resume=True)==0
    assert len(commands)==1 and '--capacity-only' not in commands[0]


def test_launcher_does_not_accept_config_or_code_drift(tmp_path,monkeypatch):
    run=prepared_run(tmp_path,monkeypatch)
    monkeypatch.setattr(launch_h100,'code_identity',lambda root:{'sha256':'other'})
    with pytest.raises(ValueError,match='Code changed'): launch_h100.execute(run)
    (run/'config.yaml').write_text('changed: true\n')
    with pytest.raises(ValueError,match='config changed'): launch_h100.execute(run)


def test_data_index_keeps_two_samples_per_effective_batch():
    from worldttn.distillation.trainer import StateTTTTrainer
    trainer=object.__new__(StateTTTTrainer)
    trainer.rank=0;trainer.world=1;trainer.accumulation=2
    trainer.data=SimpleNamespace(index=lambda counter,rank,world:counter*world+rank,
                                 batch=lambda index,frames:(index,None))
    assert list(trainer.batches(121,0))==[0,1]
    assert list(trainer.batches(121,1))==[2,3]


def test_trainer_accumulates_then_steps_and_counts_once(monkeypatch):
    import worldttn.distillation.trainer as module
    trainer=object.__new__(module.StateTTTTrainer)
    trainer.models={'student':nn.Linear(4,4),'fake':nn.Linear(4,4)}
    trainer.teacher=nn.Identity();trainer.device=torch.device('cpu')
    trainer.world=1;trainer.accumulation=2;trainer.shift=9.8
    trainer.config=OmegaConf.create({'cfg':4.5,'critic_ratio':5,'capacity_only':False})
    trainer.counts={'student':0,'critic':0}
    trainer.begin_update=lambda:0.
    trainer.check_gradients=lambda model,role:{}
    rows=[];trainer.record=lambda row,axis:rows.append(row)
    trainer.generate=lambda batch,model,collect=False:(model(batch),[])
    trainer.ema=EMA(trainer.models['student'])
    class Opt(torch.optim.AdamW):
        def step(self,*args,**kwargs):
            self.steps=getattr(self,'steps',0)+1
            self.last_grad=[p.grad.clone() for group in self.param_groups for p in group['params']]
            return super().step(*args,**kwargs)
    trainer.optimizers={key:Opt(model.parameters(),lr=1e-4) for key,model in trainer.models.items()}
    class DMD:
        def __init__(self,teacher,fake,batch,**kwargs): self.fake=fake
        def critic_loss(self,generated): return self.fake(generated).square().mean(),{}
        def student_loss(self,generated):
            return generated.square().mean(),{key:torch.tensor(.1) for key in
                ('dmdtrain_gradient_norm','normalizer_min','normalizer_mean')}
    monkeypatch.setattr(module,'StateDMD',DMD)
    reference=copy.deepcopy(trainer.models['student'])
    values=[torch.randn(2,4),torch.randn(2,4)]
    sum(reference(value).square().mean()/2 for value in values).backward()
    clip_grad(reference)
    trainer.critic_update(iter(values))
    trainer.student_update(iter(values))
    assert trainer.counts=={'student':1,'critic':1}
    assert all(opt.steps==1 for opt in trainer.optimizers.values())
    for observed,expected in zip(trainer.optimizers['student'].last_grad,reference.parameters()):
        torch.testing.assert_close(observed,expected.grad,rtol=1e-6,atol=1e-7)
    assert all(row['microbatches']==2 and row['effective_batch']==2 for row in rows)


def test_zero_exit_without_capacity_receipt_is_rejected(tmp_path,monkeypatch):
    run=prepared_run(tmp_path,monkeypatch)
    commands=[]
    class Child:
        def __init__(self,command,**kwargs): commands.append(command)
        def wait(self):
            if '--capacity-only' in commands[-1]:
                (run/'capacity/run-config.json').write_text('{"identity":{"test":1}}')
            return 0
    monkeypatch.setattr(launch_h100.subprocess,'Popen',Child)
    with pytest.raises(RuntimeError,match='acceptance receipt'): launch_h100.execute(run)
    assert len(commands)==2
    assert json.loads((run/'status.json').read_text())['status']=='failed'


@pytest.mark.parametrize('stage4',[False,True])
def test_trainer_constructor_preserves_code_identity_across_teacher_config_and_stage4(tmp_path,monkeypatch,stage4):
    """Run the real constructor; replace hardware/model IO, not identity assembly."""
    from dataclasses import dataclass
    import worldttn.distillation.trainer as module
    import worldttn.distillation.evaluate as evaluation
    @dataclass
    class Settings:
        origin: str = 'default'
        inference_flow_shift: float = 9.8
    teacher=tmp_path/'teacher.pt';teacher_file(teacher)
    payload=torch.load(teacher,weights_only=False)
    payload['distributed']['training_config']={name:{'origin':'teacher-'+name} for name in
                                               ('sana_model','text_encoder','scheduler','data')}
    torch.save(payload,teacher)
    fingerprint=digest(teacher)
    base=tmp_path/'base.safetensors';base.write_bytes(b'base')
    payload['base_sha256']=digest(base)
    torch.save(payload,teacher);fingerprint=digest(teacher)
    config=OmegaConf.load(module.Path(__file__).resolve().parents[2]/
                          'configs/distillation/longlive_ttn_step300_h100_8step.yaml')
    config.update({'teacher':str(teacher),'teacher_sha256':fingerprint,
                   'teacher_checkpoint_id':'checkpoint-300','base_weights':str(base),
                   'dataset_root':str(tmp_path/'data'),'logdir':str(tmp_path/'training'),
                   'code_sha256':'code-sha','disable_wandb':True})
    if stage4:
        stage=tmp_path/'stage8';stage.mkdir()
        source=stage/'best-adapters.pt'
        torch.save({'format':'TTN-DMD-adapters-v1','identity':{
            'checkpoint_sha256':fingerprint,'base_sha256':digest(base),
            'dataset_sha256':'dataset-sha','config':{'student_steps':8,'lora_rank':128,'cfg':4.5}},
            'weights':{'student':{},'fake':{}},'counts':{'student':25}},source)
        review=stage/'review.json'
        review.write_text(json.dumps({'video_review':True,'source_sha256':digest(source)}))
        metrics=stage/'eval-000025';metrics.mkdir()
        (metrics/'metrics.json').write_text('{"results":[]}')
        config.update({'student_steps':4,'stage4_source':str(source),'stage4_review':str(review)})
        monkeypatch.setattr(evaluation,'video_review_ready',lambda *a:True)
        monkeypatch.setattr(evaluation,'phase4_gate',lambda *a:True)
        monkeypatch.setattr(module,'load_adapter_weights',lambda *a:None)
    for name,value in {'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0','CUDA_VISIBLE_DEVICES':'5'}.items():
        monkeypatch.setenv(name,value)
    for name in ('set_device','set_per_process_memory_fraction','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'H100-test')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(total_memory=80*2**30))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a:0)
    monkeypatch.setattr(module.dist,'init_process_group',lambda *a,**k:None)
    monkeypatch.setattr(module.dist,'new_group',lambda *a,**k:None)
    monkeypatch.setattr(module,'code_identity',lambda root:{'sha256':'code-sha','commit':'commit-sha'})
    monkeypatch.setattr(module,'memory_report',lambda device:{})
    monkeypatch.setattr(module,'load_sana_config',lambda path:SimpleNamespace(**{
        name:Settings() for name in ('model','text_encoder','scheduler','data')}))
    monkeypatch.setattr(module,'Data',lambda *a:SimpleNamespace(identity='dataset-sha',
        partition={'train':[0],'validation':[1]}))
    def build(*a,**kw):
        model=nn.Linear(2,2)
        model.blocks=nn.ModuleList([nn.Identity()])
        return model
    monkeypatch.setattr(module,'build_sana',build)
    for name in ('configure_activation_checkpointing','configure_cross_attention',
                 'configure_execution','apply_checkpoint_weights','install','shard'):
        monkeypatch.setattr(module,name,lambda *a,**kw:None)
    monkeypatch.setattr(module,'read_checkpoint',lambda *a:payload)
    trainer=module.StateTTTTrainer(config)
    assert trainer.identity['code_sha256']=='code-sha'
    assert trainer.sana_config.data.origin=='teacher-data'
    recorded=json.loads((tmp_path/'training/run-config.json').read_text())
    assert recorded['code_commit']=='commit-sha'
    assert recorded['identity']==trainer.identity
    assert ('stage4_source_sha256' in trainer.identity)==stage4
