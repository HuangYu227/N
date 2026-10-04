"""Real camera/TTN gradients, frozen ownership and deliberate optimizer handoff."""
import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from test_alignment import Model
from test_alignment_detail import cached_sana
from test_training import inputs
from worldttn.anchor import install_ttn, configure_train_scope, is_ttn_camera_parameter
from worldttn.checkpoint import make_optimizer, load_checkpoint
from worldttn.core import ANCHORS, TTNConfig
from worldttn.distributed import ParallelTraining
from worldttn.parallel_checkpoint import save_training_checkpoint, restore_training_progress, validate_unfreeze_checkpoint
from worldttn.training import train_clip, linear_flow_loss
from worldttn.training_health import audit_training_parameters


def camera_model(source_class):
    torch.manual_seed(17)  # Same frozen SANA base on both sides of the handoff.
    model = Model()
    for i in ANCHORS: model.blocks[i].attn = source_class()
    install_ttn(model, TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana"))
    model.base_load_report = {"sha256": None}
    return model


def update(model, optimizer, engine=None):
    clean, noise, t, camera = inputs()
    return train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer, linear_flow_loss, t, noise,
                      width=100, height=100, tbptt=2, activation_offload="cpu", parallel=engine)


def test_visual_adaptation_updates_mappings_but_preserves_original_camera_and_backbone(cached_sana):
    model = camera_model(cached_sana)
    configure_train_scope(model, "ttn-visual")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = make_optimizer(model)
    audit = audit_training_parameters(model, optimizer)
    assert audit["train_scope"] == "ttn-visual" and audit["weight_scope"] == "ttn"
    assert optimizer.param_groups[0]["foreach"] is False
    result = update(model, optimizer)
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(before[n], p) and p.grad is None
        if is_ttn_camera_parameter(n): assert not p.requires_grad
    for i in ANCHORS:
        for group in ("qkv", "proj", "output_gate", "beta_proj"):
            parameter = getattr(model.blocks[i].attn, group).weight
            assert parameter.grad is not None and parameter.grad.norm() > 0
            assert not torch.equal(parameter, before[f"blocks.{i}.attn.{group}.weight"])
    assert result["runtime"].commit_count == result["runtime"].predict_count == 5
    assert result["runtime"].transition_fast.norm() > 0  # Stage C was active during adaptation.
    # Frozen camera operations still propagate gradients to their visual input.
    x = torch.randn(1, 4, 16, requires_grad=True)
    layer = model.blocks[3].attn
    camera = torch.ones(1, 4, 20)
    raw = layer._sana_camera_forward(layer, x, (4, 1, 1), camera, None, [None]*10, False,
                                    prope_fns=(lambda z: z, lambda z: z, lambda z: z))
    assert torch.autograd.grad(raw.square().sum(), x)[0].norm() > 0


def test_unfreeze_restores_progress_only_and_refuses_exact_resume_with_changed_scope(tmp_path, cached_sana):
    model = camera_model(cached_sana)
    configure_train_scope(model, "ttn-visual")
    engine = ParallelTraining(model, linear_flow_loss)
    optimizer = make_optimizer(model)
    update(model, optimizer, engine)
    path = tmp_path / "last.pt"
    old_identity = {"train_scope": "ttn-visual", "optimizer_foreach": False, "seed": 3407, "tbptt": 2}
    cursor = {"epoch": 9, "batch_in_epoch": 3}
    save_training_checkpoint(path, engine, optimizer, 50, cursor, old_identity)
    expected_rng = torch.get_rng_state().clone()
    model2 = camera_model(cached_sana)
    configure_train_scope(model2, "dit")
    load_checkpoint(path, model2)
    with pytest.raises(ValueError, match="train.scope"):
        load_checkpoint(path, model2, make_optimizer(model2), resume=True)
    engine2 = ParallelTraining(model2, linear_flow_loss)
    optimizer2 = make_optimizer(model2)
    joint_identity = {**old_identity, "train_scope": "dit", "backbone_lr": 1e-6}
    assert restore_training_progress(path, engine2, joint_identity) == (50, cursor)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert not optimizer2.state
    for n, p in model.named_parameters():
        torch.testing.assert_close(dict(model2.named_parameters())[n], p, rtol=0, atol=0)
    before_camera = model2.blocks[3].attn.q_proj_cam.weight.detach().clone()
    update(model2, optimizer2, engine2)
    assert not torch.equal(before_camera, model2.blocks[3].attn.q_proj_cam.weight)
    assert all(int(s["step"]) == 1 for s in optimizer2.state.values())
    payload = torch.load(path, weights_only=False)
    for change in ({"tbptt": 4}, {"seed": 1}):
        with pytest.raises(ValueError, match="training configuration"):
            validate_unfreeze_checkpoint(payload, "single", 1, {**joint_identity, **change})
    with pytest.raises(ValueError, match="backend and world"):
        validate_unfreeze_checkpoint(payload, "fsdp2", 4, joint_identity)
    wrong = copy.deepcopy(payload)
    wrong["config"]["camera_attention"] = "linear"
    with pytest.raises(ValueError, match="visual warmup"):
        validate_unfreeze_checkpoint(wrong, "single", 1, joint_identity)
    assert not any("world_state" in n or "transition_fast" in n for n in payload["adapter"])


def test_visual_builder_preserves_fp32_backbone_before_future_unfreeze(monkeypatch):
    from worldttn import cli, sana
    observed = []
    monkeypatch.setattr(cli, "read_reference", lambda *a: (TTNConfig(), {"sana_config": "unused.yaml"}))
    monkeypatch.setattr(sana, "load_sana_config", lambda *a: Namespace())
    monkeypatch.setattr(sana, "build_sana", lambda *a, **kw: observed.append(kw["dtype"]) or Namespace())
    monkeypatch.setattr(sana, "configure_cross_attention", lambda *a, **kw: {})
    cli.build(Namespace(config="unused", stage="C", sana_config=None, base_weights=None,
                        device="cpu", train_scope="ttn-visual"))
    assert observed == [torch.float32]


def _unfreeze_cli_worker(rank, size, uri, directory, mode):
    import torch.distributed as dist
    from worldttn import cli
    from test_parallel_cli import CPUFlowConfig
    from test_parallel import _rank_inputs
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=size)
    try:
        source = cached_sana.__wrapped__()
        def build(args):
            torch.manual_seed(17)
            return camera_model(source), Namespace(scheduler=CPUFlowConfig()), {}
        cli.build = build
        cli.SANAFlowLoss = lambda config: linear_flow_loss
        cli.timed_cuda = lambda call: (call(), {"seconds": 0., "peak_allocated_bytes": 0})
        first_joint = True
        expected_rng = None
        def train(model, config, batch, optimizer, k, parallel):
            nonlocal first_joint
            if model.ttn_train_scope == "dit" and first_joint:
                assert not optimizer.state
                assert torch.equal(torch.get_rng_state(), expected_rng)
                first_joint = False
            return train_clip(model, batch["clean_latents"], batch["y"], batch["camera_conditions"], optimizer,
                              linear_flow_loss, torch.ones(1, 1, 13)*500, torch.randn_like(batch["clean_latents"]),
                              width=100, height=100, tbptt=k, activation_offload="cpu", parallel=parallel)
        cli.train_update = train
        clean, _, _, camera = _rank_inputs(rank)
        root = Path(directory)
        torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera},
                   root / f"batch-{rank}.pt")
        args = Namespace(parallel=mode, seed=3407, batch_file=str(root/"batch-{rank}.pt"), device="cpu",
                         adapter=None, resume=False, unfreeze=False, output=str(root/"adapt"), max_steps=1,
                         save_every=1, tbptt=2, train_scope="ttn-visual", backbone_lr=1e-6, activation_offload="cpu")
        cli.train_command(args)
        args.adapter = str(root/"adapt/last.pt")
        args.resume, args.max_steps = True, 2
        cli.train_command(args)
        payload = torch.load(args.adapter, weights_only=False)
        shard = torch.load(root/"adapt"/payload["distributed"]["resume_dir"]/f"rank-{rank:05d}.pt", weights_only=False)
        expected_rng = shard["rng"]["torch"]
        args.resume, args.unfreeze, args.train_scope = False, True, "dit"
        args.output, args.max_steps = str(root/"joint"), 3
        cli.train_command(args)
        args.adapter = str(root/"joint/last.pt")
        args.resume, args.unfreeze, args.max_steps = True, False, 4
        cli.train_command(args)
        if rank == 0:
            for folder, steps in (("adapt", [1, 2]), ("joint", [3, 4])):
                rows = [json.loads(line) for line in (root/folder/"train.jsonl").read_text().splitlines()]
                assert [r["step"] for r in rows] == steps
                assert all(not rank_row["parameter_update"]["missing_core_gradients"] for r in rows for rank_row in r["ranks"])
            full = torch.load(args.adapter, weights_only=False)
            assert full["step"] == 4 and full["weight_scope"] == "dit"
            assert (root/"adapt/last.pt").is_file()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ddp", "fsdp2"])
def test_cli_visual_resume_unfreeze_and_joint_resume_keep_step_and_rng(tmp_path, mode):
    from test_parallel import _init_uri
    if mode == "fsdp2" and not hasattr(torch.cpu, "Stream"): pytest.skip("CPU FSDP2 unavailable")
    torch.multiprocessing.spawn(_unfreeze_cli_worker, args=(2, _init_uri(), str(tmp_path), mode), nprocs=2)
