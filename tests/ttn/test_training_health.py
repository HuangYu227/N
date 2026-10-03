import copy
import pytest
import torch
from test_training import TinyWorldModel, inputs
from worldttn.checkpoint import make_optimizer, save_checkpoint, load_checkpoint
from worldttn.training import train_clip, linear_flow_loss
from worldttn.training_health import audit_training_parameters, FirstUpdateProbe, optimizer_parameter_names


@pytest.mark.parametrize("stage", ["A", "B", "C"])
def test_actual_training_update_changes_core_parameters_without_probe_effect(stage):
    model = TinyWorldModel(stage)
    reference = copy.deepcopy(model)
    optimizer = make_optimizer(model)
    report = audit_training_parameters(model, optimizer)
    assert report["stage"] == stage
    assert report["trainable_numel"] > 0 and report["frozen_numel"] > 0
    names = sum(report["optimizer_parameter_names"], [])
    assert "blocks.3.attn.qkv.weight" in names
    assert ("ttn_system.generators.u" in names) == (stage != "A")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    probe = FirstUpdateProbe(model)
    clean, noise, t, camera = inputs()
    kw = dict(width=100, height=100, tbptt=2)
    train_clip(model, clean, torch.zeros(1, 1, 2, 8), camera, optimizer, linear_flow_loss, t, noise, **kw)
    health = probe.report(optimizer)
    train_clip(reference, clean, torch.zeros(1, 1, 2, 8), camera, make_optimizer(reference),
               linear_flow_loss, t, noise, **kw)
    assert not health["missing_core_gradients"] and not probe.before
    for block in (3, 7, 11, 15, 19):
        for group in ("qkv", "proj", "output_gate", "beta_proj"):
            values = health["groups"][f"blocks.{block}.attn.{group}"]
            assert values["grad_norm"] > 0 and values["delta_norm"] > 0
            assert values["changed_elements"] > 0 and values["optimizer_state_parameters"] > 0
    for name, p in model.named_parameters():
        assert torch.equal(p, dict(reference.named_parameters())[name])
        if not p.requires_grad: assert torch.equal(p, before[name])


@pytest.mark.parametrize("bad", ["frozen_qkv", "buffered_qkv", "beta_only_optimizer", "extra_backbone", "bf16"])
def test_training_audit_rejects_incomplete_or_wrong_trainable_configuration(bad):
    model = TinyWorldModel("C")
    if bad == "frozen_qkv": model.blocks[3].attn.qkv.requires_grad_(False)
    if bad == "buffered_qkv":
        module = model.blocks[3].attn.qkv
        weight = module.weight.detach()
        del module.weight
        module.register_buffer("weight", weight)
    if bad == "extra_backbone": model.blocks[0].ffn.requires_grad_(True)
    if bad == "bf16": model.blocks[3].attn.qkv.bfloat16()
    with pytest.raises(ValueError):
        optimizer = torch.optim.AdamW(model.blocks[3].attn.beta_proj.parameters()) if bad == "beta_only_optimizer" else make_optimizer(model)
        audit_training_parameters(model, optimizer)


def test_missing_core_backward_is_distinct_from_zero_initialized_controller_gradients():
    model = TinyWorldModel("C")
    optimizer = make_optimizer(model)
    probe = FirstUpdateProbe(model)
    model.blocks[3].attn.beta_proj.weight.grad = torch.ones_like(model.blocks[3].attn.beta_proj.weight)
    optimizer.step()
    health = probe.report(optimizer)
    assert "blocks.3.attn.qkv.weight" in health["missing_core_gradients"]
    assert not any(n.startswith("ttn_system.") for n in health["missing_core_gradients"])
    assert health["groups"]["blocks.3.attn.qkv"]["optimizer_state_parameters"] == 0


def test_checkpoint_saves_names_and_rejects_reordered_optimizer_without_loading_weights(tmp_path):
    model = TinyWorldModel("C")
    optimizer = make_optimizer(model)
    path = tmp_path / "last.pt"
    save_checkpoint(path, model, optimizer, 0)
    payload = torch.load(path, weights_only=False)
    assert payload["optimizer_parameter_names"] == optimizer_parameter_names(model, optimizer)
    optimizer.param_groups[0]["params"].reverse()
    with torch.no_grad(): model.blocks[3].attn.qkv.weight.add_(.2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    with pytest.raises(ValueError, match="names/order"):
        load_checkpoint(path, model, optimizer, resume=True)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_real_cli_refuses_beta_only_freezing_before_training(monkeypatch, tmp_path):
    from argparse import Namespace
    from test_parallel_cli import CPUFlowConfig
    from worldttn import cli
    model = TinyWorldModel("A").requires_grad_(False)
    for block in (3, 7, 11, 15, 19): model.blocks[block].attn.beta_proj.requires_grad_(True)
    model.base_load_report = {"sha256": None}
    monkeypatch.setattr(cli, "build", lambda args: (model, Namespace(scheduler=CPUFlowConfig()), {"learning_rate": 1e-5}))
    monkeypatch.setattr(cli, "SANAFlowLoss", lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, "train_update", lambda *args: pytest.fail("incomplete freezing reached training"))
    clean, _, _, camera = inputs()
    bundle = tmp_path / "batch.pt"
    torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera}, bundle)
    args = Namespace(parallel="single", seed=3407, batch_file=str(bundle), device="cpu", adapter=None,
                     resume=False, output=str(tmp_path / "train"), max_steps=1, save_every=1, tbptt=2)
    with pytest.raises(ValueError, match="trainable parameter mismatch"):
        cli.train_command(args)
