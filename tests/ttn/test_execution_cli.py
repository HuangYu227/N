import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_cli_exposes_execution_options_without_cuda_import():
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--ttn-core-backend" in result.stdout
    assert "--ttn-psi-backend" in result.stdout


def test_invalid_execution_combination_is_rejected_before_cuda_check():
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "train", "--ttn-psi-backend", "projected"],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "requires reuse or compiled" in result.stderr


def test_stage_evaluation_rejects_optimized_diagnostics_before_creating_output(tmp_path, monkeypatch):
    from worldttn.stage_evaluation import stage_evaluate_command
    args = SimpleNamespace(output=str(tmp_path / "eval"), training_run="snapshot", frames=61,
        fixed_cases="cases.pt", seed=3407, eval_cases=1, cross_attn_backend="math", device="cpu",
        steps=20, cfg_scale=4.5, cached_blocks=2, ttn_core_backend="compiled", ttn_psi_backend="projected")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[command.index("--ttn-core-backend") + 1] == "compiled"
        assert command[command.index("--ttn-psi-backend") + 1] == "projected"
        out = Path(command[command.index("--output") + 1]); out.mkdir()
        (out / "summary.json").write_text(json.dumps({"protocol": {"step": 1, "stage": "C",
            "checkpoint_sha256": "c", "fixed_cases_sha256": "f"}, "episodes": []}))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("worldttn.provenance.implementation_identity", lambda: {"git_commit": "test"})
    with pytest.raises(ValueError, match="reference/reference"):
        stage_evaluate_command(args)
    assert calls == [] and not Path(args.output).exists()


def test_layout_audit_deduplicates_and_includes_strides(tmp_path):
    import torch
    from worldttn.performance import ExecutionOptions, configure_execution, audit_layout
    from test_training import TinyWorldModel
    model = TinyWorldModel(); path = tmp_path / "layouts.jsonl"
    configure_execution(model, ExecutionOptions("reuse", "projected", layout_audit=str(path)))
    anchor = model.blocks[3].attn
    x = torch.randn(2, 3, 8).transpose(0, 1)
    for _ in range(2): audit_layout(anchor, "noisy", q=x)
    audit_layout(anchor, "noisy", q=x.contiguous())
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["inputs"]["q"]["stride"] == list(x.stride())


def test_precision_audit_is_read_only():
    import torch
    from worldttn.performance import precision_audit
    before = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    report = precision_audit()
    assert report["python"] == sys.executable
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == before


@pytest.mark.parametrize("ablation,diagnostics", [("identity", False), ("no-ttt", False), ("full", True)])
def test_runtime_rejects_optimized_mechanism_before_predict(ablation, diagnostics):
    import torch
    from test_training import TinyWorldModel, inputs
    from worldttn.session import TTNSession
    from worldttn.performance import ExecutionOptions, configure_execution
    model = TinyWorldModel("C"); configure_execution(model, ExecutionOptions("reuse", "projected"))
    with torch.no_grad():
        session = TTNSession(model, inputs()[3], 100, 100, ablation=ablation, diagnostics=diagnostics)
        session.reset(1)
        with pytest.raises(ValueError, match="Full Stage C"):
            session.begin_chunk(0, 1, prefill=True)
        assert session.runtime.predict_count == session.runtime.commit_count == 0


@pytest.mark.parametrize("stage", ["A", "B"])
def test_execution_is_stage_c_only(stage):
    from test_training import TinyWorldModel
    from worldttn.performance import ExecutionOptions, configure_execution
    with pytest.raises(ValueError, match="Full Stage C"):
        configure_execution(TinyWorldModel(stage), ExecutionOptions("reuse", "reference"))


@pytest.mark.parametrize("kind", ["throughput", "profile", "layouts", "ordinary"])
@pytest.mark.parametrize("optin", [True, False])
def test_checkpoint_io_is_explicit_for_benchmarks(kind, optin):
    from worldttn.cli import checkpoint_due
    args = SimpleNamespace(ttn_benchmark_stable_steps=10 if kind == "throughput" else 0,
        ttn_profiler_trace="trace" if kind == "profile" else None,
        ttn_layout_audit=kind == "layouts", benchmark_save_checkpoint=optin, max_steps=12, save_every=50)
    assert checkpoint_due(args, 12) == (kind == "ordinary" or optin)
    assert not checkpoint_due(args, 11)


def test_benchmark_derives_scope_and_refuses_explicit_mismatch(tmp_path):
    import torch
    from worldttn.cli import resolve_train_scope
    p = tmp_path / "checkpoint.pt"; torch.save({"train_scope": "dit"}, p)
    args = SimpleNamespace(adapter=str(p), ttn_benchmark_stable_steps=2, train_scope=None)
    resolve_train_scope(args); assert args.train_scope == "dit"
    args.train_scope = "ttn-visual"
    with pytest.raises(ValueError, match="exact resume"): resolve_train_scope(args)


@pytest.mark.parametrize("kind", ["throughput", "profile", "layouts"])
@pytest.mark.parametrize("optin", [False, True])
def test_real_training_loop_only_publishes_benchmark_bundle_when_opted_in(tmp_path, monkeypatch, kind, optin):
    from contextlib import nullcontext
    import torch
    from test_training import TinyWorldModel, inputs
    from test_parallel_cli import CPUFlowConfig
    from worldttn import cli, benchmark
    from worldttn.training import train_clip, linear_flow_loss
    from worldttn.checkpoint_integrity import audit_checkpoint
    monkeypatch.delenv("TORCH_LOGS", raising=False)
    monkeypatch.delenv("CUDA_LAUNCH_BLOCKING", raising=False)
    clean, noise, timestep, camera = inputs()
    bundle = tmp_path / "batch.pt"
    torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8),
                "camera_conditions": camera, "width": 100, "height": 100}, bundle)
    def build(args):
        torch.manual_seed(17)
        model = TinyWorldModel("C"); model.base_load_report = {"sha256": None}
        return model, SimpleNamespace(scheduler=CPUFlowConfig()), {"learning_rate": 1e-5}
    def update(model, config, batch, optimizer, k, parallel):
        return train_clip(model, batch["clean_latents"], batch["y"], camera, optimizer,
                          linear_flow_loss, timestep, noise, width=100, height=100, tbptt=k, parallel=parallel)
    monkeypatch.setattr(cli, "build", build)
    monkeypatch.setattr(cli, "SANAFlowLoss", lambda config: linear_flow_loss)
    monkeypatch.setattr(cli, "train_update", update)
    monkeypatch.setattr(cli, "timed_cuda", lambda call: (call(), {
        "seconds": 1., "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}))
    monkeypatch.setattr(benchmark, "profiler_update", lambda *a: nullcontext())
    output = tmp_path / "benchmark"
    args = SimpleNamespace(parallel="single", seed=3407, batch_file=str(bundle), device="cpu",
        adapter=None, resume=False, output=str(output), max_steps=1, save_every=1, tbptt=2,
        ttn_core_backend="reference", ttn_psi_backend="reference", ttn_profile=False,
        ttn_benchmark_stable_steps=1 if kind == "throughput" else 0,
        ttn_layout_audit=kind == "layouts", ttn_profiler_trace="trace" if kind == "profile" else None,
        benchmark_save_checkpoint=optin)
    cli.train_command(args)
    records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(records) == (2 if kind == "throughput" else 1)
    assert (output / "last.pt").exists() is optin
    assert bool(list(output.glob("last-resume-*"))) is optin
    if optin: assert audit_checkpoint(output / "last.pt")["step"] == records[-1]["step"]
    if kind == "throughput": assert (output / "performance.json").is_file()
