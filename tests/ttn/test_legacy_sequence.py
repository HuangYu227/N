"""Real SANA blocks on a tiny grid: legacy scan equivalence and full credit."""
import ast
import copy
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from test_alignment_detail import cached_sana
from test_sdpa_policy import text_attention
from test_training import inputs
from worldttn.anchor import install_ttn, configure_train_scope
from worldttn.core import TTNConfig
from worldttn.session import TTNSession
from worldttn.training import train_clip, linear_flow_loss

@pytest.fixture
def cached_ffn():
    # Execute the released CPU-capable definitions without optional CUDA imports.
    root = Path(__file__).resolve().parents[2]
    namespace = dict(torch=torch, nn=nn, checkpoint=checkpoint,
                     build_act=runpy.run_path(str(root/"diffusion/model/act.py"))["build_act"],
                     build_norm=runpy.run_path(str(root/"diffusion/model/norms.py"))["build_norm"],
                     _INT32_SAFE_CONV_ELEMENTS=1 << 30)
    for filename, names in [("diffusion/model/utils.py", {"val2list", "val2tuple", "get_same_padding", "checkpoint_preserving_strides"}),
                            ("diffusion/model/nets/basic_modules.py",
                             {"ConvLayer", "GLUMBConv", "GLUMBConvTemp", "CachedGLUMBConvTemp"})]:
        source = root/filename
        nodes = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        for node in nodes: node.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["CachedGLUMBConvTemp"]

class NativeRecurrence(nn.Module):
    """CPU recurrence fixture; verifies layer scan scheduling, not fused GDN numerics."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, x, kv_cache, ttn_live_cache=False, save_kv_cache=False, **kw):
        current = self.proj(x)
        prior = kv_cache[0]
        if prior is not None:
            current = current + .1 * prior
        if save_kv_cache:
            kv_cache[0] = current.mean(1, keepdim=True)
            if not ttn_live_cache:
                kv_cache[0] = kv_cache[0].detach()
        return current, kv_cache

@pytest.fixture
def sequence_model(cached_sana, cached_ffn, text_attention):
    path = Path(__file__).resolve().parents[2] / "diffusion/model/nets/sana_multi_scale_video_camctrl.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.ClassDef) and n.name == "SanaVideoMSCamCtrlBlock")
    namespace = dict(torch=torch, nn=nn, Optional=Optional,
                     t2i_modulate=lambda x, shift, scale: x * (1+scale) + shift)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cls = namespace["SanaVideoMSCamCtrlBlock"]
    camera_globals = cached_sana._cached_cam_branch_softmax.__globals__
    camera_globals["prepare_prope_fns"] = lambda **kw: (lambda x: x, lambda x: x, lambda x: x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            for _ in range(20):
                block = cls.__new__(cls); nn.Module.__init__(block)
                block.attn = cached_sana()
                block.norm1, block.norm2 = nn.LayerNorm(16), nn.LayerNorm(16)
                block.scale_shift_table = nn.Parameter(torch.randn(6, 16)*.05)
                block.drop_path = nn.Identity()
                block.cross_attn = text_attention(16, 2)
                block.cross_attn.set_sdpa_backend("math")
                block.mlp = cached_ffn(16, 32)
                nn.init.normal_(block.mlp.t_conv.weight, std=.05)
                block.flash_attn_additional = None
                block.cross_attn_image_embeds = False
                block.chunk_size = 3
                block.ttn_fullgrad_checkpoint = False
                self.blocks.append(block)
            install_ttn(self, TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana", local_update=True, persistent_meta=True))
            for i, block in enumerate(self.blocks):
                if i not in (3, 7, 11, 15, 19):
                    block.attn = NativeRecurrence()
            self.ttn_training_protocol = "dual-psi-clean-fullgrad-v1"
            self.forward_calls = 0

        def forward(self, x, t, y, kv_cache, ttn_chunk_context, **kw):
            self.forward_calls += 1
            b, c, f, h, w = x.shape
            z = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
            if t.ndim == 1:
                t = t[:, None, None].expand(b, 1, f)
            t0 = t[..., None].expand(-1, -1, -1, 96) / 10000
            plucker = kw.get("chunk_plucker")
            plucker = plucker.permute(0, 2, 3, 4, 1).reshape(b, -1, c).contiguous() if plucker is not None else None
            caches = []
            for block, cache in zip(self.blocks, kv_cache):
                z, cache = block(z, y, t0, mask=kw.get("mask"), THW=(f, h, w), kv_cache=cache,
                    save_kv_cache=kw.get("save_kv_cache", False), ttn_chunk_context=ttn_chunk_context,
                    ttn_live_cache=kw.get("ttn_live_cache", False),
                    frame_valid_mask=kw.get("frame_valid_mask"), camera_conditions=kw.get("camera_conditions"),
                    plucker_emb=plucker)
                caches.append(cache)
            return z.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3), caches
    torch.manual_seed(103)
    return Model()


def sequence_forward(model, clean, noisy, times, camera):
    from worldttn.legacy_sequence_training import LegacySequenceSession, LegacySequenceTraining
    from worldttn.runtime import TTNChunkContext
    b, _, frames, _, _ = clean.shape
    session = LegacySequenceSession(model, camera, 100, 100)
    runtime = session.reset(b); session.clean = clean
    state = runtime.world_state.repeat(3,1,1,1,1)
    psi = runtime.transition_fast.repeat(3,1,1,1)
    ids = torch.arange(frames, device=clean.device).expand(3*b,-1)
    valid = torch.ones_like(ids,dtype=torch.bool)
    context = TTNChunkContext(state,state,psi,psi,model.ttn_system,ids,
        camera[:,:,:16].reshape(b,frames,4,4).repeat(3,1,1,1),valid,valid,0,id(runtime),sequence_mode=True)
    # Execute the actual training wrapper to prepare camera-only coefficients.
    result={}
    def prediction_loss(session, clean, t, noise, y, ctx, cache, start, end, mask, info, lm, on_prediction):
        output,_ = session.forward(noisy,t,y,ctx,cache,start,end,mask,info)
        result['output']=output
        return output.square().flatten(1).mean(-1)
    episode=SimpleNamespace(session=session,context=context,clean=clean,timesteps=times,noise=noisy,
        y=torch.ones(b,1,2,16,device=clean.device),mask=torch.ones(b,2,dtype=torch.bool,device=clean.device),
        data_info=None,loss_mask=clean.new_ones(b,1,frames,1,1),on_prediction=None)
    LegacySequenceTraining(model,prediction_loss)(episode)
    return result['output'],context


def test_legacy_layer_scan_matches_chunk_forward_and_clean_state(sequence_model):
    from worldttn.training import chunk_ranges
    m=sequence_model; checked=copy.deepcopy(m)
    clean,noise,t,camera=inputs()
    clean,noise=clean.expand(-1,-1,-1,1,2).clone(),noise.expand(-1,-1,-1,1,2).clone()
    y=torch.ones(1,1,2,16);mask=torch.ones(1,2,dtype=torch.bool)
    noisy=noise.clone();noisy[:,:,:1]=clean[:,:,:1]
    # Match the observed frame on both reference and production paths.
    reference=copy.deepcopy(m)
    session=TTNSession(reference,camera,100,100,live_cache=True);session.reset(1);session.prefill(clean[:,:,:1],y,mask)
    cache=[[None]*10 for _ in m.blocks];outputs=[]
    for start,end in chunk_ranges(clean.shape[2]):
        ctx=session.begin_chunk(start,end)
        out,_=session.forward(noisy[:,:,start:end],t[:,:,start:end],y,ctx,cache,start,end,mask)
        outputs.append(out)
        _,cache=session.clean_forward(clean[:,:,start:end],y,ctx,cache,start,end,mask,cached_chunks=2)
    actual,ctx=sequence_forward(checked,clean,noisy,t,camera)
    torch.testing.assert_close(actual,torch.cat(outputs,2),rtol=2e-5,atol=2e-6)
    for i in range(5):
        torch.testing.assert_close(ctx.candidates[i][0],session.runtime.world_state[:,i],rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(ctx.candidates[i][1],session.runtime.transition_fast[:,i],rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('frames', [7,25])
def test_legacy_layer_checkpoint_and_early_credit(sequence_model,frames):
    clean,noise,t,camera=inputs()
    if frames == 25:
        indices=torch.cat((torch.arange(7),torch.arange(4,7).repeat(6)))
        clean,noise,t,camera=clean[:,:,indices],noise[:,:,indices],t[:,:,indices],camera[:,indices]
    clean,noise=clean.expand(-1,-1,-1,1,2).clone(),noise.expand(-1,-1,-1,1,2).clone()
    results=[]
    for recompute in (False,True):
        m=copy.deepcopy(sequence_model)
        for block in m.blocks:block.ttn_fullgrad_checkpoint=recompute
        history=clean.clone().requires_grad_();noisy=noise.clone().requires_grad_()
        out,ctx=sequence_forward(m,history,noisy,t,camera)
        out[:,:,-3:].square().mean().backward()
        assert torch.isfinite(history.grad).all()
        assert history.grad[:,:,:1].norm()>0 and history.grad[:,:,1:4].norm()>0
        assert history.grad[:,:,-3:].count_nonzero()==0
        results.append((out.detach(),history.grad,[p.grad for p in m.parameters()]))
    torch.testing.assert_close(results[0],results[1],rtol=3e-5,atol=3e-6)


def test_legacy_scan_gradients_match_untruncated_chunk_oracle(sequence_model):
    from worldttn.checkpoint import make_optimizer
    clean,noise,t,camera=inputs()
    clean,noise=clean.expand(-1,-1,-1,1,2).clone(),noise.expand(-1,-1,-1,1,2).clone()
    results=[]
    for scan in (False,True):
        m=configure_train_scope(copy.deepcopy(sequence_model),'dit')
        m.ttn_training_protocol='dual-psi-clean-fullgrad-v1' if scan else 'clean-tbptt'
        m.ttn_fullgrad_checkpoint=False
        result=train_clip(m,clean,torch.ones(1,1,2,16),camera,make_optimizer(m),linear_flow_loss,t,noise,
            width=100,height=100,mask=torch.ones(1,2,dtype=torch.bool),tbptt=0,cached_chunks=2)
        results.append((result['loss'],[p.grad for p in m.parameters()],m.state_dict(),result['outer_grad_norm']))
    assert results[0][0]==pytest.approx(results[1][0],rel=2e-5,abs=2e-6)
    torch.testing.assert_close(results[0][1:3],results[1][1:3],rtol=5e-5,atol=3e-6)


def test_full_sequence_record_matches_production_summary_schema(sequence_model):
    from worldttn.checkpoint import make_optimizer
    from worldttn.stability import stability_rows, progress_line
    model=configure_train_scope(sequence_model,'dit')
    clean,noise,t,camera=inputs()
    result=train_clip(model,clean,torch.ones(1,1,2,16),camera,make_optimizer(model),linear_flow_loss,t,noise,
        width=100,height=100,tbptt=0,cached_chunks=2)
    record=dict(step=1,stage='C',train_scope='dit',config=model.ttn_system.config.to_dict(),
        loss=result['loss'],outer_grad_norm=result['outer_grad_norm'],seconds=1,
        ranks=[dict(rank=0,prefill=result['prefill'],chunks=result['chunks'])])
    rows=list(stability_rows(record))
    assert len(rows)==5*(1+len(result['chunks']))
    assert {r['chunk'] for r in rows}=={-1,*range(len(result['chunks']))}
    assert '[TTN progress] step 1/500' in progress_line(record,500)
    assert len(result['prediction_diagnostics']['per_frame_velocity_mse'][0])==clean.shape[2]


def test_completed_episode_context_is_released(sequence_model,monkeypatch):
    import weakref
    from worldttn.legacy_sequence_training import LegacySequenceSession
    from worldttn.checkpoint import make_optimizer
    contexts=[]
    original=LegacySequenceSession.forward
    def capture(self,x,t,y,context,*args,**kwargs):
        contexts.append(weakref.ref(context))
        return original(self,x,t,y,context,*args,**kwargs)
    monkeypatch.setattr(LegacySequenceSession,'forward',capture)
    model=configure_train_scope(sequence_model,'dit');clean,noise,t,camera=inputs()
    result=train_clip(model,clean,torch.ones(1,1,2,16),camera,make_optimizer(model),linear_flow_loss,t,noise,
        width=100,height=100,tbptt=0,cached_chunks=2)
    assert contexts and all(reference() is None for reference in contexts)
    assert result['runtime'].world_state.grad_fn is None
