"""Real SANA block/FFN/camera code on a tiny grid; native GDN separately needs CUDA."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
from torch import nn

from test_activation_checkpoint import cached_ffn, text_attention
from test_alignment_detail import cached_sana
from test_frame_memory import config
from test_training import inputs
from worldttn.anchor import install_ttn
from worldttn.session import TTNSession
from worldttn.training import train_clip, linear_flow_loss


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
                block.ttn_sequence_checkpoint = False
                self.blocks.append(block)
            install_ttn(self, config())
            for i, block in enumerate(self.blocks):
                if i not in (3, 7, 11, 15, 19):
                    block.attn = NativeRecurrence()
            self.ttn_training_protocol = "frame-noisy-fullgrad-v1"
            self.forward_calls = 0

        def forward(self, x, t, y, kv_cache, ttn_chunk_context, **kw):
            self.forward_calls += 1
            b, c, f, h, w = x.shape
            z = x.permute(0, 2, 3, 4, 1).reshape(b, -1, c)
            if t.ndim == 1:
                t = t[:, None, None].expand(b, 1, f)
            t0 = t[..., None].expand(-1, -1, -1, 96) / 10000
            plucker = kw.get("chunk_plucker")
            plucker = plucker.permute(0, 2, 3, 4, 1).reshape(b, -1, c) if plucker is not None else None
            caches = []
            for block, cache in zip(self.blocks, kv_cache):
                z, cache = block(z, y, t0, mask=kw.get("mask"), THW=(f, h, w), kv_cache=cache,
                    save_kv_cache=kw.get("save_kv_cache", False), ttn_chunk_context=ttn_chunk_context,
                    frame_valid_mask=kw.get("frame_valid_mask"), camera_conditions=kw.get("camera_conditions"),
                    plucker_emb=plucker)
                caches.append(cache)
            return z.reshape(b, f, h, w, c).permute(0, 4, 1, 2, 3), caches
    torch.manual_seed(103)
    return Model()


def batch():
    clean, noise, t, cam = inputs()
    return clean[:, :, :7].expand(-1, -1, -1, 1, 2).clone(), noise[:, :, :7].expand(-1, -1, -1, 1, 2).clone(), t[:, :, :7], cam[:, :7]


def run_forward(model, x, t, cam, *, sequence=True):
    session = TTNSession(model, cam, 100, 100)
    session.reset(1)
    context = session.begin_chunk(0, x.shape[2])
    context.sequence_mode = sequence
    y = torch.randn(1, 1, 2, 16)
    out, _ = session.forward(x, t, y, context, [[None]*10 for _ in model.blocks], 0, x.shape[2],
                             torch.ones(1, 2, dtype=torch.bool), save=True)
    return out, context, session


def test_sequence_recompute_preserves_outputs_gradients_and_single_publication(sequence_model):
    model = sequence_model
    checked = copy.deepcopy(model)
    for block in checked.blocks:
        block.ttn_sequence_checkpoint = True
    clean, _, t, cam = batch()
    results = []
    for m in (model, checked):
        torch.manual_seed(123)
        x = clean.clone().requires_grad_()
        out, context, session = run_forward(m, x, t, cam)
        saved = {i: row[0].detach().clone() for i, row in context.candidates.items()}
        out[:, :, -3:].square().mean().backward()
        assert x.grad[:, :, :1].norm() > 0, "last group must train the observed prefill"
        assert x.grad[:, :, 1:4].norm() > 0, "last group must train earlier noisy history"
        assert session.runtime.commit_count == 0
        assert set(context.candidates) == set(range(5))
        for i in saved:
            torch.testing.assert_close(context.candidates[i][0], saved[i], rtol=0, atol=0)
        results.append((out.detach(), x.grad, [p.grad for p in m.parameters()]))
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-6)


def test_anchor_replay_is_bounded_without_truncating_early_credit(monkeypatch, sequence_model):
    from dataclasses import replace
    from worldttn import sequence
    model = sequence_model
    model.ttn_system.config = replace(model.ttn_system.config, memory_capacity_frames=16,
        memory_prefix_frames=10, memory_recent_frames=4, memory_start_frame=13)
    for index in (3, 7, 11, 15, 19):
        model.blocks[index].attn.config = model.ttn_system.config
    checked = copy.deepcopy(model)
    checked.blocks[19].ttn_sequence_checkpoint = True
    clean, _, t, cam = inputs()
    clean, t = [torch.cat((value, value[:, :, 1:]), 2) for value in (clean, t)]
    cam = torch.cat((cam, cam[:, 1:]), 1)
    clean = clean.expand(-1, -1, -1, 1, 2).clone()
    replay_frames = []
    original = sequence._checkpoint
    def counted(function, args, **kwargs):
        if function.__name__ == "run" and kwargs["enabled"]:
            replay_frames.append(args[0].shape[1] // 2)
        return original(function, args, **kwargs)
    monkeypatch.setattr(sequence, "_checkpoint", counted)
    results = []
    for candidate in (model, checked):
        torch.manual_seed(123)
        x = clean.clone().requires_grad_()
        out, context, session = run_forward(candidate, x, t, cam)
        out[:, :, -3:].square().mean().backward()
        assert x.grad[:, :, :1].norm() > 0
        assert x.grad[:, :, 1:4].norm() > 0
        assert session.runtime.commit_count == 0
        assert all(len(rows) == 9 for rows in context.memory_stats.values())
        results.append((out.detach(), x.grad, [p.grad for p in candidate.parameters()]))
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-6)
    assert replay_frames and max(replay_frames) <= 24, replay_frames
    assert sum(replay_frames) == 25


def test_sequence_chunk_causality_and_frozen_observed_prefill(sequence_model):
    x, _, t, cam = batch()
    changed = x.clone(); changed[:, :, 4:] += 13
    torch.manual_seed(6); first, _, _ = run_forward(sequence_model, x, t, cam)
    torch.manual_seed(6); second, _, _ = run_forward(sequence_model, changed, t, cam)
    torch.testing.assert_close(first[:, :, :4], second[:, :, :4], rtol=0, atol=0)
    changed = x.clone(); changed[:, :, 1:] -= 20
    torch.manual_seed(6); third, _, _ = run_forward(sequence_model, changed, t, cam)
    torch.testing.assert_close(first[:, :, :1], third[:, :, :1], rtol=0, atol=0)


def test_train_entry_one_forward_one_optimizer_update(sequence_model):
    clean, noise, t, cam = batch()
    for block in sequence_model.blocks: block.ttn_sequence_checkpoint = True
    opt = torch.optim.AdamW([p for p in sequence_model.parameters() if p.requires_grad], lr=1e-5)
    result = train_clip(sequence_model, clean, torch.ones(1, 1, 2, 16), cam, opt,
        linear_flow_loss, t, noise, width=100, height=100, tbptt=0, mask=torch.ones(1, 2, dtype=torch.bool))
    assert sequence_model.forward_calls == 1
    assert {int(state["step"]) for state in opt.state.values()} == {1}
    assert result["exposure"]["temporal_detaches"] == 0
    assert result["exposure"]["clean_history_forwards"] == 0
    assert result["runtime"].commit_count == 0
    assert len(result["chunks"]) == 2


def test_plucker_conditioning_survives_scan_and_recomputation(sequence_model):
    for block in sequence_model.blocks:
        block.plucker_proj = nn.Linear(16, 16)
    checked = copy.deepcopy(sequence_model)
    for block in checked.blocks: block.ttn_sequence_checkpoint = True
    clean, noise, t, cam = batch()
    results = []
    for model in (sequence_model, checked):
        plucker = clean.clone().requires_grad_()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        result = train_clip(model, clean, torch.ones(1, 1, 2, 16), cam, opt,
            linear_flow_loss, t, noise, width=100, height=100, tbptt=0,
            mask=torch.ones(1, 2, dtype=torch.bool), extras={"chunk_plucker": plucker})
        assert plucker.grad[:, :, :1].norm() > 0
        assert plucker.grad[:, :, 4:].norm() > 0
        assert all(block.plucker_proj.weight.grad.norm() > 0 for block in model.blocks)
        results.append((result["loss"], plucker.grad, [p.detach() for p in model.parameters()]))
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-6)


def test_frame_cli_preserves_shipped_camera_recipe_and_rejects_tbptt(monkeypatch, sequence_model):
    import yaml
    from worldttn import cli, sana
    source = yaml.safe_load((cli.ROOT / "configs/worldttn/sana_teacher_single_gpu.yaml").read_text())
    config = SimpleNamespace(model=SimpleNamespace(**source["model"]), data=SimpleNamespace(**source["data"]))
    monkeypatch.setattr(sana, "load_sana_config", lambda path: config)
    monkeypatch.setattr(sana, "build_sana", lambda *args, **kwargs: sequence_model)
    monkeypatch.setattr(sana, "configure_cross_attention", lambda *args, **kwargs: {})
    args = SimpleNamespace(config=str(cli.ROOT / "configs/worldttn/frame_fullgrad.json"),
        stage="C", sana_config=None, base_weights=None, device="cpu", train_scope="dit", tbptt=0)
    model, loaded, settings = cli.build(args)
    assert loaded.data.return_chunk_plucker and loaded.model.use_chunk_plucker_post_attn
    assert loaded.data.num_frames == 961 and settings["tbptt"] == 0
    assert model.ttn_training_protocol == "frame-noisy-fullgrad-v1"
    args.tbptt = 4
    with pytest.raises(ValueError, match="tbptt=0"):
        cli.build(args)


def test_native_camera_and_ffn_keep_early_gradients(cached_sana, cached_ffn):
    torch.manual_seed(17)
    camera = cached_sana()
    early = torch.randn(1, 2, 16, requires_grad=True)
    late = torch.randn(1, 2, 16, requires_grad=True)
    cache = [None]*10
    options = dict(HW=(1, 1, 2), camera_conditions=torch.ones(1, 1, 20), rotary_emb=None,
                   save_kv_cache=True, ttn_live_cache=True, prope_fns=(lambda x: x,)*3)
    camera._cached_cam_branch_softmax(early, kv_cache=cache, **options)
    out = camera._cached_cam_branch_softmax(late, kv_cache=list(cache), **options)
    assert torch.autograd.grad(out.square().sum(), early)[0].norm() > 0
    ffn = cached_ffn(16, 32)
    nn.init.normal_(ffn.t_conv.weight, std=.1)
    _, history = ffn(early, HW=(1, 1, 2), kv_cache=[None]*10, save_kv_cache=True, ttn_live_cache=True)
    out, _ = ffn(late, HW=(1, 1, 2), kv_cache=history, save_kv_cache=True, ttn_live_cache=True)
    assert torch.autograd.grad(out.square().sum(), early)[0].norm() > 0


def test_native_short_convolution_accepts_one_observed_frame_and_keeps_gradient():
    path = Path(__file__).resolve().parents[2] / "diffusion/model/ops/fused_streaming.py"
    node = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "_cached_temporal_short_conv")
    namespace = dict(torch=torch, ShortConvolution=nn.Module)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    short = namespace[node.name]
    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, 1, 4))
        def forward(self, x):
            value = torch.nn.functional.conv1d(torch.nn.functional.pad(x.transpose(1, 2), (3, 0)),
                                              self.weight, groups=4)
            return value.transpose(1, 2), None
    conv = Conv()
    observed = torch.randn(1, 2, 4, requires_grad=True)
    _, cache = short(observed, conv, (1, 1, 2), None, True, True)
    current = torch.randn(1, 6, 4, requires_grad=True)
    output, final = short(current, conv, (3, 1, 2), cache, True, True)
    assert output.shape == current.shape and final.shape == (2, 3, 4)
    assert torch.autograd.grad(output[:, -2:].sum(), observed)[0].norm() > 0


def test_first_inference_group_matches_isolated_training_prefill(sequence_model):
    clean, _, t, cam = batch()
    x, t, cam = clean[:, :, :4], t[:, :, :4], cam[:, :4]
    torch.manual_seed(123)
    y = torch.randn(1, 1, 2, 16)
    mask = torch.ones(1, 2, dtype=torch.bool)
    session = TTNSession(sequence_model, cam, 100, 100)
    session.reset(1)
    context = session.begin_chunk(0, 4); context.sequence_mode = True
    train_out, _ = session.forward(x, t, y, context, [[None]*10 for _ in sequence_model.blocks], 0, 4, mask, save=True)
    infer = TTNSession(sequence_model, cam, 100, 100)
    infer.prefill(x[:, :, :1], y, mask)
    before = infer.runtime.world_state.detach().clone()
    hashes = infer.runtime.verify_memory_prefix()
    current = infer.begin_chunk(0, 4)
    native = [[None]*10 for _ in sequence_model.blocks]
    for _ in range(2):
        infer_out, next_native = infer.forward(x, t, y, current, native, 0, 4, mask)
        torch.testing.assert_close(train_out, infer_out, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(infer.runtime.world_state, before, rtol=0, atol=0)
        assert infer.runtime.commit_count == 1 and not current.candidates
        assert infer.runtime.verify_memory_prefix() == hashes
        assert all(value is None for block in next_native for value in block)
        native = next_native
    infer.clean_forward(x, y, current, [[None]*10 for _ in sequence_model.blocks], 0, 4, mask, cached_chunks=2)
    assert infer.runtime.commit_count == 2
    assert infer.runtime.committed_frame_ids == [set(range(4))]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires GPU")
def test_cuda_sequence_checkpoint_and_offload_parity(sequence_model):
    clean, noise, t, cam = [value.cuda() for value in batch()]
    original = sequence_model.cuda()
    checked = copy.deepcopy(original)
    for block in checked.blocks: block.ttn_sequence_checkpoint = True
    results = []
    for model, offload in ((original, "none"), (checked, "cpu")):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            result = train_clip(model, clean, torch.ones(1, 1, 2, 16, device="cuda"), cam, opt,
                linear_flow_loss, t, noise, width=100, height=100, tbptt=0,
                mask=torch.ones(1, 2, dtype=torch.bool, device="cuda"), activation_offload=offload)
        results.append(result)
    assert results[0]["loss"] == pytest.approx(results[1]["loss"], rel=1e-5, abs=1e-6)
    for a, b in zip(original.parameters(), checked.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-3, atol=5e-5)
