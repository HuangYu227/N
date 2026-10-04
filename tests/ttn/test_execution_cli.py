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


def test_stage_evaluation_forwards_backend_to_all_three_subprocesses(tmp_path, monkeypatch):
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
    stage_evaluate_command(args)
    assert len(calls) == 3


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
