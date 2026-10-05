"""Telemetry must not change Meta-TTT or run in ordinary noisy inference."""
import copy
import json
from argparse import Namespace
import torch
from test_meta_runtime import episode, begin
from test_meta_training import model, update
from test_training import inputs
from worldttn.checkpoint import make_optimizer
from worldttn.session import record_noise
from worldttn.training import linear_flow_loss


def test_plain_noisy_inference_skips_noise_serialization_and_local_stats(monkeypatch):
    system, runtime, anchors = episode()
    ctx = begin(system, runtime)
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary inference must not compute Local telemetry")
    monkeypatch.setattr("worldttn.anchor.local_anchor_stats", forbidden)
    record_noise(ctx, torch.tensor([500.]))
    assert ctx.noise_info == {}
    anchors[0](torch.randn(1, 8, 16), HW=(2, 2, 2), ttn_chunk_context=ctx)
    assert ctx.local_stats == {}


def test_diagnostics_keep_each_solver_call_without_changing_output_or_gradient():
    system, runtime, anchors = episode()
    ordinary = begin(system, runtime)
    runtime.diagnostics = True
    diagnostic = begin(system, runtime)
    x = torch.randn(1, 8, 16)
    expected = anchors[0](x, HW=(2, 2, 2), ttn_chunk_context=ordinary)
    for step in (800., 200.):
        record_noise(diagnostic, torch.tensor([step]))
        for anchor in anchors:
            actual = anchor(x, HW=(2, 2, 2), ttn_chunk_context=diagnostic)
            if anchor.index == 0:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                diagnostic_output = actual
    clean = diagnostic.for_clean()
    for anchor in anchors:
        anchor(x, HW=(2, 2, 2), ttn_chunk_context=clean)
    runtime.commit_chunk(clean)
    for row in runtime.last_stats["anchors"]:
        trajectory = row["local_trajectory"]
        assert [call["call"] for call in trajectory] == [0, 1]
        torch.testing.assert_close(torch.tensor([call["noise_sigma"] for call in trajectory]), torch.tensor([[.8], [.2]]))
        assert all("raw_grad_norm" in call["per_head"] for call in trajectory)
        assert "loss_after_transport" in trajectory[0]["support"]
    reference_grad = torch.autograd.grad(expected.square().sum(), system.local_eta_logits, retain_graph=True)[0]
    diagnostic_grad = torch.autograd.grad(diagnostic_output.square().sum(), system.local_eta_logits)[0]
    torch.testing.assert_close(reference_grad, diagnostic_grad, atol=0, rtol=0)


def test_parameter_update_audit_is_opt_in_and_preserves_math(monkeypatch):
    audited = model()
    ordinary = copy.deepcopy(audited)
    from worldttn.training_health import FirstUpdateProbe
    calls = []
    def record(model):
        calls.append(1)
        return FirstUpdateProbe(model)
    monkeypatch.setattr("worldttn.training_health.FirstUpdateProbe", record)
    plain = update(ordinary, make_optimizer(ordinary))
    assert calls == [] and "optimizer_updates" not in plain
    checked = update(audited, make_optimizer(audited), audit=True)
    assert calls == [1] and checked["optimizer_updates"]["by_origin"]["ttn_new"]["delta_norm"] > 0
    assert plain["loss"] == checked["loss"]
    for name, parameter in ordinary.named_parameters():
        torch.testing.assert_close(parameter, dict(audited.named_parameters())[name], atol=0, rtol=0)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, dict(audited.named_parameters())[name].grad, atol=0, rtol=0)
    for field in ("world_state", "transition_fast"):
        torch.testing.assert_close(getattr(plain["runtime"], field), getattr(checked["runtime"], field), atol=0, rtol=0)


def test_meta_cli_audits_first_update_only_and_keeps_each_steps_stability(monkeypatch, tmp_path):
    from test_parallel_cli import CPUFlowConfig
    from worldttn import cli
    from worldttn.training_health import FirstUpdateProbe
    m = model()
    m.base_load_report = {"sha256": None}
    monkeypatch.setattr(cli, "build", lambda args: (m, Namespace(scheduler=CPUFlowConfig()), {"learning_rate": 1e-5}))
    monkeypatch.setattr(cli, "SANAFlowLoss", lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, "train_update", lambda model, config, batch, optimizer, k, parallel: update(model, optimizer))
    monkeypatch.setattr(cli, "timed_cuda", lambda fn: (fn(), {"seconds": 0., "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}))
    calls = []
    def record(model):
        calls.append(1)
        return FirstUpdateProbe(model)
    monkeypatch.setattr("worldttn.training_health.FirstUpdateProbe", record)
    clean, _, _, camera = inputs()
    batch = tmp_path / "batch.pt"
    torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera,
                "width": 100, "height": 100}, batch)
    output = tmp_path / "train"
    args = Namespace(parallel="single", seed=3407, batch_file=str(batch), device="cpu", adapter=None,
                     resume=False, output=str(output), max_steps=2, save_every=2, tbptt=2, train_scope="dit")
    cli.train_command(args)
    assert calls == [1]
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert ["parameter_update" in row["ranks"][0] for row in records] == [True, False]
    for record in records:
        for chunk in record["ranks"][0]["chunks"]:
            assert len(chunk["anchors"]) == 5
            assert all(anchor["local"]["recorded"] and anchor["persistent"] for anchor in chunk["anchors"])
