"""Joint DiT updates and complete checkpoints, using the real TTN training loop."""
import copy
from argparse import Namespace
from pathlib import Path
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn import anchor
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint
from worldttn.training import train_clip, linear_flow_loss
from worldttn.training_health import FirstUpdateProbe, audit_training_parameters


def joint_model(stage="C"):
    model = TinyWorldModel(stage)
    anchor.configure_train_scope(model, "dit")
    return model


def update(model, optimizer, *, amp=False, parallel=None):
    clean, noise, t, camera = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        return train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer,
                          linear_flow_loss, t, noise, width=100, height=100,
                          tbptt=2, activation_offload="cpu", parallel=parallel)


@pytest.mark.parametrize("stage", ["A", "C"])
def test_joint_scope_and_optimizer_partition(stage):
    model = joint_model(stage)
    optimizer = make_optimizer(model, 1e-5, backbone_lr=1e-6)
    audit = audit_training_parameters(model, optimizer)
    assert audit["train_scope"] == audit["weight_scope"] == "dit"
    assert all(p.requires_grad and p.dtype == torch.float32 for b in model.blocks for p in b.parameters())
    assert all(p.requires_grad == (stage != "A") for p in model.ttn_system.parameters())
    assert [(g["name"], g["lr"], g["foreach"]) for g in optimizer.param_groups] == [
        ("ttn_new", 1e-5, False), ("sana_inherited", 1e-6, False)]
    names = audit["optimizer_parameter_names"]
    assert "blocks.3.attn.beta_proj.weight" in names[0]
    assert "blocks.3.attn.qkv.weight" in names[1]
    assert "blocks.3.ffn.weight" in names[1]
    assert not set(names[0]) & set(names[1])


def test_origin_policy_keeps_all_visual_and_camera_mappings_at_backbone_lr():
    from worldttn.anchor import is_ttn_new_parameter
    model = joint_model()
    optimizer = make_optimizer(model)
    audit = audit_training_parameters(model, optimizer)
    new, inherited = audit["optimizer_parameter_names"]
    assert all(is_ttn_new_parameter(n) for n in new)
    assert all(not is_ttn_new_parameter(n) for n in inherited)
    for i in (3, 7, 11, 15, 19):
        for group in ("qkv", "q_norm", "k_norm", "proj", "output_gate", "q_proj_cam", "k_proj_cam", "v_proj_cam", "out_proj_cam"):
            assert f"blocks.{i}.attn.{group}.weight" in inherited
    optimizer.param_groups[0]["params"], optimizer.param_groups[1]["params"] = (
        optimizer.param_groups[1]["params"], optimizer.param_groups[0]["params"])
    with pytest.raises(ValueError, match="origin"):
        audit_training_parameters(model, optimizer)


def test_legacy_joint_resume_policy_is_explicit_and_never_silently_regrouped(tmp_path):
    from worldttn.cli import resolve_optimizer_policy
    model = joint_model()
    optimizer = make_optimizer(model, policy="legacy")
    update(model, optimizer)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 1)
    payload = torch.load(path, weights_only=False)
    payload.pop("optimizer_policy")  # A genuine old checkpoint has no policy marker.
    torch.save(payload, path)
    args = Namespace(resume=True, train_scope="dit", optimizer_policy=None)
    resolve_optimizer_policy(args, payload)
    assert args.optimizer_policy == "legacy"
    restored = joint_model()
    assert load_checkpoint(path, restored, make_optimizer(restored, policy=args.optimizer_policy), resume=True) == 1
    wrong = joint_model()
    with pytest.raises(ValueError, match="policy"):
        load_checkpoint(path, wrong, make_optimizer(wrong), resume=True)
    args.optimizer_policy = "origin"
    with pytest.raises(ValueError, match="policy"):
        resolve_optimizer_policy(args, payload)


@pytest.mark.parametrize("amp", [False, True])
def test_real_joint_update_changes_anchor_and_backbone_without_changing_transactions(amp):
    model = joint_model()
    optimizer = make_optimizer(model)
    probe = FirstUpdateProbe(model)
    result = update(model, optimizer, amp=amp)
    report = probe.report(optimizer)
    assert not report["missing_core_gradients"]
    for group in ("blocks.3.attn.qkv", "blocks.3.attn.proj", "blocks.3.attn.output_gate", "blocks.0.ffn.weight"):
        assert report["groups"][group]["delta_norm"] > 0
    assert report["backbone"]["updated_parameters"] > 0
    assert report["backbone"]["by_component"]["ffn"]["updated_parameters"] > 0
    assert report["backbone"]["by_component"]["gdn"]["parameters_with_grad"] == 0
    # The fixture intentionally does not execute its 15 original attention modules.
    assert "blocks.0.attn.qkv.weight" in report["missing_backbone_gradients"]
    assert result["runtime"].commit_count == result["runtime"].predict_count == 5
    assert result["runtime"].committed_frame_ids == [set(range(13))]


def test_joint_full_checkpoint_inference_and_frozen_reexport(tmp_path):
    model = joint_model()
    optimizer = make_optimizer(model)
    update(model, optimizer)
    path = tmp_path / "full.pt"
    save_checkpoint(path, model, optimizer, 1)
    payload = torch.load(path, weights_only=False)
    assert payload["train_scope"] == payload["weight_scope"] == "dit"
    assert payload["adapter"].keys() == model.state_dict().keys()
    assert not any("world_state" in n or "transition_fast" in n for n in payload["adapter"])
    restored = TinyWorldModel().bfloat16()
    restored.blocks[3].attn.float()
    load_checkpoint(path, restored)
    assert restored.ttn_weight_scope == "dit"
    assert getattr(restored, "ttn_train_scope", "ttn") == "ttn"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    # A frozen-backbone run must retain and export the modified backbone, too.
    anchor.configure_train_scope(restored, "ttn")
    assert not restored.blocks[0].ffn.weight.requires_grad
    exported = tmp_path / "frozen.pt"
    save_checkpoint(exported, restored, make_optimizer(restored), 0)
    new = TinyWorldModel()
    load_checkpoint(exported, new)
    torch.testing.assert_close(new.blocks[0].ffn.weight, model.blocks[0].ffn.weight, rtol=0, atol=0)


def test_legacy_adapter_can_initialize_joint_but_cannot_exact_resume(tmp_path):
    old = TinyWorldModel()
    path = tmp_path / "old.pt"
    save_checkpoint(path, old, make_optimizer(old), 40)
    payload = torch.load(path, weights_only=False)
    payload.pop("train_scope", None)
    payload.pop("weight_scope", None)
    torch.save(payload, path)
    model = joint_model()
    before = model.blocks[0].ffn.weight.detach().clone()
    load_checkpoint(path, model)
    assert torch.equal(before, model.blocks[0].ffn.weight)
    assert model.ttn_weight_scope == "dit"
    with pytest.raises(ValueError, match="train.scope"):
        load_checkpoint(path, model, make_optimizer(model), resume=True)


def test_joint_resume_next_update_and_rejects_bad_metadata_before_mutation(tmp_path):
    model = joint_model()
    optimizer = make_optimizer(model)
    update(model, optimizer)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 1)
    expected_rng = torch.rand(4)
    update(model, optimizer)
    resumed = joint_model()
    new_optimizer = make_optimizer(resumed)
    assert load_checkpoint(path, resumed, new_optimizer, resume=True) == 1
    assert torch.equal(torch.rand(4), expected_rng)
    update(resumed, new_optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
    before = copy.deepcopy(resumed.state_dict())
    with pytest.raises(ValueError, match="learning rate"):
        load_checkpoint(path, resumed, make_optimizer(resumed, backbone_lr=2e-6), resume=True)
    for key in before: assert torch.equal(before[key], resumed.state_dict()[key])
    payload = torch.load(path, weights_only=False)
    for bad in ("missing_weight", "base", "scope"):
        broken = copy.deepcopy(payload)
        if bad == "missing_weight": del broken["adapter"]["blocks.0.ffn.weight"]
        elif bad == "base": broken["base_sha256"] = "bad"
        else: broken["weight_scope"] = "invalid"
        torch.save(broken, path)
        with pytest.raises(ValueError): load_checkpoint(path, resumed)
        for key in before: assert torch.equal(before[key], resumed.state_dict()[key])


def test_joint_resume_identity_tracks_scope_and_backbone_lr():
    from worldttn.cli import _training_identity
    from test_parallel_cli import CPUFlowConfig
    args = Namespace(seed=3407, batch_file="batch.pt", train_scope="ttn", backbone_lr=1e-6)
    config = Namespace(scheduler=CPUFlowConfig())
    old = _training_identity(args, config, {}, 2)
    assert "train_scope" not in old  # Existing frozen-backbone exact resumes stay valid.
    args.train_scope = "dit"
    joint = _training_identity(args, config, {}, 2)
    assert joint["train_scope"] == "dit" and joint["backbone_lr"] == 1e-6
    args.backbone_lr = 2e-6
    assert joint != _training_identity(args, config, {}, 2)


def test_joint_builder_loads_fp32_master_weights_without_bf16_round_trip(monkeypatch):
    from worldttn import cli, sana
    from worldttn.core import TTNConfig
    seen = []
    monkeypatch.setattr(cli, "read_reference", lambda *a: (TTNConfig(), {"sana_config": "unused.yaml"}))
    monkeypatch.setattr(sana, "load_sana_config", lambda *a: Namespace())
    def build(*args, **kwargs):
        seen.append(kwargs.get("dtype", torch.bfloat16))
        return TinyWorldModel()
    monkeypatch.setattr(sana, "build_sana", build)
    monkeypatch.setattr(sana, "configure_cross_attention", lambda *a, **kw: {})
    args = Namespace(config="unused.json", stage="C", sana_config=None, base_weights=None, device="cpu", train_scope="dit")
    cli.build(args)
    args.train_scope = "ttn"
    cli.build(args)
    assert seen == [torch.float32, torch.bfloat16]


def _joint_cli_worker(rank, size, uri, output, mode="ddp"):
    from worldttn import cli
    from test_parallel import _rank_inputs
    from test_parallel_cli import CPUFlowConfig
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=size)
    try:
        def build(args):
            torch.manual_seed(17)
            model = TinyWorldModel()
            model.base_load_report = {"sha256": None}
            return model, Namespace(scheduler=CPUFlowConfig()), {}
        def train(model, config, batch, optimizer, k, parallel):
            clean, noise, t, camera = _rank_inputs(rank)
            return train_clip(model, clean, batch["y"], camera, optimizer, linear_flow_loss,
                              t, noise, width=100, height=100, tbptt=k, parallel=parallel,
                              activation_offload="cpu")
        cli.build = build
        cli.SANAFlowLoss = lambda config: linear_flow_loss
        cli.train_update = train
        cli.timed_cuda = lambda call: (call(), {"seconds": 0., "peak_allocated_bytes": 0})
        clean, noise, t, camera = _rank_inputs(rank)
        batch = Path(output) / f"batch-{rank}.pt"
        torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera}, batch)
        args = Namespace(parallel=mode, seed=3407, batch_file=str(Path(output) / "batch-{rank}.pt"),
                         device="cpu", adapter=None, resume=False, output=str(Path(output) / "train"),
                         max_steps=1, save_every=1, tbptt=2, train_scope="dit", backbone_lr=1e-6,
                         activation_offload="cpu")
        cli.train_command(args)
        args.adapter = str(Path(args.output) / "last.pt")
        args.resume, args.max_steps = True, 2
        cli.train_command(args)
        if rank == 0:
            import json
            records = [json.loads(line) for line in (Path(args.output) / "train.jsonl").read_text().splitlines()]
            assert [r["step"] for r in records] == [1, 2]
            assert all(r["train_scope"] == "dit" and r["weight_scope"] == "dit" for r in records)
            assert all(not x["parameter_update"]["missing_core_gradients"] for r in records for x in r["ranks"])
            assert all(x["parameter_update"]["backbone"]["updated_parameters"] > 0 for r in records for x in r["ranks"])
            full = torch.load(args.adapter, weights_only=False)
            assert "blocks.0.ffn.weight" in full["adapter"]
            frozen = TinyWorldModel()
            load_checkpoint(args.adapter, frozen)
            assert torch.equal(frozen.blocks[0].ffn.weight, full["adapter"]["blocks.0.ffn.weight"])
            identity = dict(full["distributed"]["training_config"])
            from worldttn.parallel_checkpoint import validate_training_checkpoint
            identity["backbone_lr"] = 2e-6
            with pytest.raises(ValueError, match="training configuration"):
                validate_training_checkpoint(full, mode, size, identity)
    finally:
        dist.destroy_process_group()


def test_joint_ddp_cli_updates_and_full_checkpoint_resume(tmp_path):
    from test_parallel import _init_uri
    torch.multiprocessing.spawn(_joint_cli_worker, args=(2, _init_uri(), str(tmp_path)), nprocs=2)


@pytest.mark.parametrize("mode,size", [("ddp", 3), ("fsdp2", 2)])
def test_joint_parallel_export_and_exact_resume_next_update(tmp_path, mode, size):
    from test_parallel import _init_uri, _resume_worker
    from torch.distributed import fsdp
    if mode == "fsdp2" and (not hasattr(torch.cpu, "Stream") or not hasattr(fsdp, "fully_shard")):
        pytest.skip("CPU FSDP2 not available")
    torch.multiprocessing.spawn(_resume_worker, args=(size, _init_uri(), str(tmp_path), mode, True), nprocs=size)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_joint_rejects_bad_backbone_learning_rate(value):
    with pytest.raises(ValueError, match="learning rate"):
        make_optimizer(joint_model(), backbone_lr=value)


def test_joint_cli_scope_is_train_only_and_slurm_passes_flags():
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "infer", "--train-scope", "dit"],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "only supported for train" in result.stderr
    script = (root / "tools/ttn_slurm_train.sbatch").read_text()
    assert '--train-scope "${TRAIN_SCOPE:-ttn}"' in script
    assert '--backbone-lr "${BACKBONE_LR:-1e-6}"' in script
