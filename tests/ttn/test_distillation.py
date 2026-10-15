import copy
import pytest
import torch
from torch import nn
from dataclasses import replace
from test_legacy_sequence import sequence_model, cached_sana, cached_ffn, text_attention
from test_state_ttt import adapted_model
from test_training import inputs
from worldttn.distillation.sampler import Transactions, cfg_velocity, flow_x0, shifted_schedule
from worldttn.distillation.objective import StateDMD, FlowScheduler
from worldttn.distillation.data import split, video_id
from worldttn.distillation.adapters import install, optimizer, parameters, clip_grad, load_adapter_weights
from worldttn.distillation.checkpoint import EMA, save, restore
from worldttn.distillation.evaluate import phase4_gate, video_review_ready


def test_extensionless_hf_safetensors_blob(tmp_path):
    from safetensors.torch import save_file
    from tools.download import find_model
    path = tmp_path/'sha256-blob-without-suffix'
    save_file({'weight':torch.arange(4)}, str(path))
    state = find_model(str(path))['state_dict']
    torch.testing.assert_close(state['weight'],torch.arange(4))


def test_stage4_requires_video_review_and_both_quality_gates():
    teacher = {'lpips':.30,'tail':{'latent_mse':.40},'fps':2.}
    student = {'lpips':.31,'tail':{'latent_mse':.41},'fps':3.}
    rows = [{'case':'case-a','method':'teacher20','metrics':teacher},
            {'case':'case-a','method':'student8','metrics':student}]
    assert not phase4_gate(rows,False)
    assert phase4_gate(rows,True)
    assert not phase4_gate(rows[:-1]+[dict(rows[-1],metrics={**student,'lpips':.32})],True)
    assert not phase4_gate(rows[:-1]+[dict(rows[-1],metrics={**student,'fps':1.})],True)


def test_stage4_preparation_checks_review_and_metrics(tmp_path, monkeypatch):
    import json
    from omegaconf import OmegaConf
    from tools.distillation import prepare_stage4
    root = tmp_path/'repo'
    config = root/'configs/distillation/longlive_ttn_step200_8step.yaml'
    config.parent.mkdir(parents=True)
    config.write_text('backend: ttn_state_ttt\nstudent_steps: 8\n')
    monkeypatch.setattr(prepare_stage4,'ROOT',root)
    eight = tmp_path/'eight'; eight.mkdir()
    best = eight/'best-adapters.pt'
    torch.save({'counts':{'student':25}},best)
    review = tmp_path/'review.json'
    review.write_text(json.dumps({'source_sha256':prepare_stage4.sha256(best),'video_review':False}))
    with pytest.raises(ValueError,match='reviewed'):
        prepare_stage4.prepare(eight,review)
    review.write_text(json.dumps({'source_sha256':prepare_stage4.sha256(best),'video_review':True}))
    with pytest.raises(ValueError,match='videos'):
        prepare_stage4.prepare(eight,review)
    videos = eight/'review-videos'; videos.mkdir()
    methods = ('teacher20','undistilled8','student8','student8-ttt-off')
    rows = [{'seconds':duration,'method':method}
            for duration in (30,60) for method in methods]
    for row in rows:
        (videos/f'{row["seconds"]}s-{row["method"]}.mp4').write_bytes(b'mp4')
    (videos/'manifest.json').write_text(json.dumps({'status':'completed',
        'best_sha256':prepare_stage4.sha256(best),'videos':rows}))
    assert video_review_ready(eight,prepare_stage4.sha256(best))
    assert not video_review_ready(eight,'another-checkpoint')
    evaluation = eight/'eval-000025';evaluation.mkdir()
    evaluation.joinpath('metrics.json').write_text(json.dumps({'score':[.31,.41],
        'results':[{'case':'case-a','method':'teacher20','metrics':{'lpips':.30,
                   'tail':{'latent_mse':.40},'fps':2.}},
                   {'case':'case-a','method':'student8','metrics':{'lpips':.31,
                   'tail':{'latent_mse':.41},'fps':3.}}]}))
    staged = prepare_stage4.prepare(eight,review)
    assert OmegaConf.load(staged).student_steps == 4
    assert OmegaConf.load(staged).stage4_source == str(best)


def make_case(model):
    model = adapted_model(copy.deepcopy(model))
    config = replace(model.ttn_system.config, eta_psi=0.)
    model.ttn_system.config = config
    for block in model.blocks:
        if hasattr(block.attn,'config'): block.attn.config = config
    clean, noise, _, camera = inputs()
    clean, noise, camera = clean[:,:,:7], noise[:,:,:7], camera[:,:7]
    batch = dict(initial_latent=clean[:,:,:1], camera_conditions=camera,
                 y=torch.zeros(1,1,2,16), uncondition=torch.zeros(1,1,2,16), width=100,height=100)
    return model,batch,noise


def test_sigma_cfg_and_scheduler_sign():
    noisy = torch.ones(1,2,4,1,1)
    torch.testing.assert_close(flow_x0(noisy,torch.tensor(.25),noisy*2), noisy*.5)
    torch.testing.assert_close(cfg_velocity(torch.tensor([[1.],[3.]]),4.5),torch.tensor([[10.]]))
    sigma,_ = shifted_schedule(8,9.8,'cpu')
    assert len(sigma)==9 and sigma[-1]==0 and (sigma[:-1]>sigma[1:]).all()
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=9.8)
    scheduler.set_timesteps(8)
    actual = scheduler.step(noisy*2,scheduler.timesteps[0],noisy,return_dict=False)[0]
    torch.testing.assert_close(actual,noisy+(sigma[1]-sigma[0])*noisy*2)
    mixed = FlowScheduler().add_noise(noisy,noisy*3,torch.tensor([250.]))
    torch.testing.assert_close(mixed,noisy*1.5)


def test_video_disjoint_split():
    rows = [dict(dataset_name='set',key=f'{v}_000001_000962_s{i:06d}')
            for v in ('video_a','video_b','video_c') for i in range(4)]
    partition = split(rows)
    assert split(rows)==partition and partition['validation']
    assert not {video_id(rows[i]) for i in partition['train']} & {
        video_id(rows[i]) for i in partition['validation']}
    with pytest.raises(ValueError): split(rows[:4])


def test_lora_zero_and_temporal_credit(sequence_model):
    model,batch,noise = make_case(sequence_model)
    baseline,_ = Transactions(model,batch,checkpointed=False).generate(noise,2,9.8,4.5)
    install(model,2)
    assert len([n for n in parameters(model) if '.state_ttt.' in n])==10
    written = []
    def retain(module,args,kwargs,result):
        context = kwargs['ttn_chunk_context']
        if context.clean_mode:
            value = context.candidates[0][0]
            if value.requires_grad: value.retain_grad(); written.append(value)
    handle = model.register_forward_hook(retain,with_kwargs=True)
    txn = Transactions(model,batch,checkpointed=False)
    output,state = txn.generate(noise,2,9.8,4.5)
    torch.testing.assert_close(output,baseline,rtol=2e-6,atol=3e-7)
    assert len(written)==3  # prefill + two prediction chunks, never per solver step
    output[:,:,-3:].square().mean().backward()
    assert written[0].grad is not None and written[0].grad.norm()>0
    assert all(p.grad is not None and p.grad.norm()>0 for n,p in parameters(model).items() if '.state_ttt.' in n)
    assert any(p.grad is not None and p.grad.norm()>0 for n,p in parameters(model).items() if 'lora_B' in n)
    handle.remove()
    # A separate role cannot retain or mutate the generator's S/cache.
    before = state.state.detach().clone()
    other = copy.deepcopy(model)
    Transactions(other,batch,checkpointed=False).score(output.detach(),torch.full((1,7),.5),output.detach())
    torch.testing.assert_close(state.state,before,rtol=0,atol=0)


def test_stage4_transfers_lora_and_state_ttt_weights(sequence_model):
    source,_,_ = make_case(sequence_model)
    target = copy.deepcopy(source)
    install(source,2);install(target,2)
    with torch.no_grad():
        for name,param in parameters(source).items():
            param.add_(.125 if '.state_ttt.' in name else .25)
    saved = {name:param.detach().clone() for name,param in parameters(source).items()}
    load_adapter_weights(target,saved)
    for name,param in parameters(target).items():
        torch.testing.assert_close(param,saved[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='schema'):
        load_adapter_weights(target,{})


def test_checkpoint_recompute_output_and_gradients(sequence_model):
    model,batch,noise = make_case(sequence_model); install(model,2)
    other = copy.deepcopy(model)
    outputs = []
    for m,checkpointed in ((model,False),(other,True)):
        generated,_ = Transactions(m,batch,checkpointed=checkpointed).generate(noise,2,9.8,4.5)
        generated[:,:,-3:].square().mean().backward()
        outputs.append(generated.detach())
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for name,p in parameters(model).items():
        q = parameters(other)[name]
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-8)


class FixedScore(nn.Module):
    def __init__(self,conditional,unconditional):
        super().__init__(); self.c,self.u=conditional,unconditional
    def forward(self,noisy_image_or_video,timestep,conditional_dict,**kwargs):
        value = self.c if conditional_dict['conditional'] else self.u
        return noisy_image_or_video*0, torch.full_like(noisy_image_or_video,value)


def test_upstream_dmd_direction_mask_and_nonfinite(sequence_model):
    m,batch,_ = make_case(sequence_model)
    dmd=StateDMD(m,copy.deepcopy(m),batch)
    dmd.real_score=FixedScore(2,0); dmd.fake_score=FixedScore(1,1)
    generated=torch.full((1,16,7,1,1),2.,requires_grad=True)
    loss,logs=dmd.student_loss(generated); loss.backward()
    assert torch.equal(generated.grad[:,:,:1],torch.zeros_like(generated.grad[:,:,:1]))
    expected = torch.full_like(generated.grad[:,:,1:],(-8/7)/(16*6))
    torch.testing.assert_close(generated.grad[:,:,1:],expected)
    dmd.real_score=FixedScore(float('nan'),0)
    with pytest.raises(FloatingPointError): dmd.student_loss(generated)


def test_adapter_only_resume_exact_next_update(sequence_model,tmp_path):
    m,batch,noise=make_case(sequence_model); install(m,2)
    models={'student':m,'fake':copy.deepcopy(m)}
    opts={n:optimizer(v,1e-5) for n,v in models.items()}; ema=EMA(m)
    def update():
        for role,model in models.items():
            opts[role].zero_grad(set_to_none=True)
            output,_=Transactions(model,batch,checkpointed=False).generate(noise,2,9.8,4.5)
            output.square().mean().backward(); clip_grad(model); opts[role].step()
        ema.update(m)
    update()
    save(tmp_path,models,opts,ema,{'student':1,'critic':5},{'test':'identity'})
    update()
    expected={role:{n:p.detach().clone() for n,p in parameters(model).items()} for role,model in models.items()}
    counts=restore(tmp_path,models,opts,ema,{'test':'identity'})
    assert counts=={'student':1,'critic':5}
    update()
    for role,model in models.items():
        for n,p in parameters(model).items(): torch.testing.assert_close(p,expected[role][n],rtol=0,atol=0)
