"""Opt-in Linux GPU gate, launched by torchrun; includes real fused native GDN.

FRAME_DISTRIBUTED_TEST=1 FRAME_TEST_OUTPUT=/shared/fresh/path
torchrun --standalone --nproc_per_node=4 -m pytest -q -s tests/ttn/test_frame_distributed.py
"""
import copy
from dataclasses import replace
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from test_sequence_training import sequence_model
from test_training import inputs
from test_activation_checkpoint import cached_ffn, text_attention
from test_alignment_detail import cached_sana
from worldttn.anchor import configure_train_scope
from worldttn.checkpoint import make_optimizer, load_checkpoint
from worldttn.distributed import ParallelTraining, initialize
from worldttn.parallel_checkpoint import save_training_checkpoint, restore_training_checkpoint, _pack
from worldttn.training import train_clip, linear_flow_loss
from worldttn.session import TTNSession

pytestmark = pytest.mark.skipif(os.environ.get("FRAME_DISTRIBUTED_TEST") != "1",
    reason="requires an allocated Linux CUDA/FSDP2 environment; opt in explicitly")


def snapshot(model):
    return {name: (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().cpu().clone()
            for name, p in model.named_parameters()}


def test_native_fsdp_sequence_recompute_offload_and_resume(sequence_model):
    _, device = initialize("fsdp2", "cuda")
    try:
        from diffusion.model.nets.sana_gdn_camctrl_blocks import CachedChunkCausalGDNUCPESinglePathLiteLA
        torch.manual_seed(93)
        for i, block in enumerate(sequence_model.blocks):
            block.plucker_proj = torch.nn.Linear(16, 16)
            if i not in (3, 7, 11, 15, 19):
                # Triton MMA requires K >= 16; keep the same total hidden size.
                block.attn = CachedChunkCausalGDNUCPESinglePathLiteLA(16, 16, heads=1, dim=16,
                    cam_dim=16, cam_heads=1, patch_size=(1, 1, 1), conv_kernel_size=4,
                    qk_norm=True, use_bias=True)
                torch.nn.init.normal_(block.attn.out_proj_cam.weight, std=.02)
        template = configure_train_scope(sequence_model, "dit").to(device)
        template.ttn_system.config = replace(template.ttn_system.config, memory_capacity_frames=16,
            memory_prefix_frames=10, memory_recent_frames=4, memory_start_frame=13)
        for index in (3, 7, 11, 15, 19):
            template.blocks[index].attn.config = template.ttn_system.config
            assert template.blocks[index].attn.config.memory_prefix_frames == 10
        template.base_load_report = {"sha256": "synthetic-frame-gate"}
        clean, noise, t, camera = inputs()
        clean, noise, t = [torch.cat((value, value[:, :, 1:]), 2).to(device)
                           for value in (clean, noise, t)]
        camera = torch.cat((camera, camera[:, 1:]), 1).to(device)
        clean, noise = [value.expand(-1, -1, -1, 1, 2).clone() for value in (clean, noise)]
        y, mask = torch.ones(1, 1, 2, 16, device=device), torch.ones(1, 2, dtype=torch.bool, device=device)

        # Only a late loss can establish credit across the frame-13 history boundary.
        early = clean.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            session = TTNSession(template, camera, 100, 100)
            session.reset(1)
            context = session.begin_chunk(0, 25)
            context.sequence_mode = True
            output, _ = session.forward(early, t, y, context, [[None]*10 for _ in template.blocks],
                                        0, 25, mask, save=True)
            output[:, :, -3:].square().mean().backward()
        assert early.grad[:, :, :1].norm() > 0
        assert early.grad[:, :, 1:4].norm() > 0
        template.zero_grad(set_to_none=True)
        del session, context, output, early

        def update(model, optimizer, engine=None):
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                return train_clip(model, clean, y, camera, optimizer, linear_flow_loss, t, noise,
                    width=100, height=100, mask=mask, tbptt=0, parallel=engine,
                    activation_offload="cpu-pageable" if engine else "none", extras={"chunk_plucker": clean})

        baseline = copy.deepcopy(template)
        opt = make_optimizer(baseline)
        reference = update(baseline, opt)
        expected_weights = snapshot(baseline)
        del baseline, opt
        model = copy.deepcopy(template)
        for block in model.blocks: block.ttn_sequence_checkpoint = True
        engine = ParallelTraining(model, linear_flow_loss, "fsdp2", activation_offload="cpu-pageable")
        opt = make_optimizer(model)
        actual = update(model, opt, engine)
        assert actual["loss"] == pytest.approx(reference["loss"], rel=2e-4, abs=2e-5)
        for name, value in snapshot(model).items():
            torch.testing.assert_close(value, expected_weights[name], rtol=2e-4, atol=3e-6, msg=name)
        assert model.blocks[19].attn.beta_proj.weight.grad is not None
        assert actual["exposure"]["temporal_detaches"] == 0
        assert actual["exposure"]["frame_state_updates_per_anchor"] == 25
        for anchor in actual["chunks"][-1]["anchors"]:
            assert all(record["history_active"] == [True] for record in anchor["proximal"]["frames"])
        folder = Path(os.environ["FRAME_TEST_OUTPUT"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "last.pt"
        identity = {"tbptt": 0, "training_protocol": "frame-noisy-fullgrad-v1"}
        save_training_checkpoint(path, engine, opt, 1, {"rank": dist.get_rank()}, identity)
        expected = update(model, opt, engine)
        expected_weights, expected_optimizer = snapshot(model), copy.deepcopy(_pack(opt.state_dict()))
        del model, engine, opt
        resumed = copy.deepcopy(template)
        for block in resumed.blocks: block.ttn_sequence_checkpoint = True
        load_checkpoint(path, resumed)
        engine = ParallelTraining(resumed, linear_flow_loss, "fsdp2", activation_offload="cpu-pageable")
        opt = make_optimizer(resumed)
        step, cursor = restore_training_checkpoint(path, engine, opt, identity)
        assert step == 1 and cursor == {"rank": dist.get_rank()}
        actual = update(resumed, opt, engine)
        assert actual["loss"] == expected["loss"]
        from tools.ttn_compare_resume import identical
        identical(snapshot(resumed), expected_weights, "resumed model")
        identical(_pack(opt.state_dict()), expected_optimizer, "resumed optimizer")
    finally:
        dist.destroy_process_group()
