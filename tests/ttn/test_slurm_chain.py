"""Test exact-resume launch wiring and failure boundaries without Slurm/GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def chain():
    path = Path(__file__).resolve().parents[2] / "tools/ttn_slurm_chain.py"
    spec = importlib.util.spec_from_file_location("ttn_slurm_chain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def plan(tmp_path):
    output = tmp_path / "formal"
    output.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    return {"project": str(tmp_path), "output": str(output), "source": str(source),
            "initial_step": 6, "target_step": 27, "segment_steps": 10, "partition": "short",
            "environment": {"ROOT": str(tmp_path), "PYTHON": str(tmp_path / "envs/worldttn/bin/python"),
                            "COMMAND": "train", "PARALLEL": "fsdp2", "STAGE": "C", "TRAIN_SCOPE": "dit",
                            "RESUME": "1", "SEED": "3407", "TBPTT": "2", "BACKBONE_LR": "1e-6",
                            "ACTIVATION_OFFLOAD": "cpu", "CROSS_ATTN_BACKEND": "math", "DATASET_ROOT": "shared/data"}}


def complete(run, step):
    run = Path(run)
    (run / "train.jsonl").write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
    (run / "last.pt").touch()


def test_submission_preserves_training_and_does_not_inherit_old_allocation(chain, plan, monkeypatch):
    for key in ("SLURM_JOB_ID", "SLURM_NTASKS", "SBATCH_ARRAY_INX", "MASTER_ADDR", "MASTER_PORT",
                "CUDA_VISIBLE_DEVICES", "RANK", "BATCH_FILE", "BASE_WEIGHTS", "CONFIG", "UNFREEZE", "DIAGNOSTIC_UNMASK_ALL_VALID"):
        monkeypatch.setenv(key, "stale")
    calls = []
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(stdout="98765;cluster\n")
    monkeypatch.setattr(chain.subprocess, "run", run)
    manifest = Path(plan["output"]) / "chain.json"
    assert chain.submit(plan, manifest, 6, "423072") == "98765"
    cmd, kw = calls[0]
    for flag in ("--nodes=4", "--ntasks=4", "--ntasks-per-node=1", "--gres=gpu:1",
                 "--mem=256G", "--time=01:00:00", "--dependency=afterok:423072", "--kill-on-invalid-dep=yes"):
        assert flag in cmd
    env = kw["env"]
    assert not any(name.startswith(("SLURM_", "SBATCH_")) for name in env)
    for key in ("MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES", "RANK", "BATCH_FILE", "BASE_WEIGHTS", "CONFIG", "UNFREEZE", "DIAGNOSTIC_UNMASK_ALL_VALID"):
        assert key not in env
    assert env["ADAPTER"] == str(Path(plan["source"]) / "last.pt")
    assert env["MAX_STEPS"] == "16" and env["SAVE_EVERY"] == "28"
    for key, value in plan["environment"].items():
        assert env[key] == value
    assert json.loads((Path(plan["output"]) / "jobs.jsonl").read_text())["stop_step"] == 16
    env = chain.environment(plan, 26)
    assert env["MAX_STEPS"] == "27" and env["ADAPTER"] == str(Path(plan["output"]) / "last.pt")


@pytest.mark.parametrize("active,accounting,expected", [
    ("RUNNING\n", "", "423072"), ("PENDING\n", "", "423072"),
    ("", "COMPLETED|0:0\n", None), ("", "FAILED|1:0\n", "error"),
    ("", "TIMEOUT|0:0\n", "error"), ("", "", "error"),
])
def test_parent_dependency_handles_completed_jobs_after_controller_purges_them(chain, monkeypatch, active, accounting, expected):
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=active if cmd[0] == "squeue" else accounting)
    monkeypatch.setattr(chain.subprocess, "run", run)
    if expected == "error":
        with pytest.raises(ValueError, match="succeeded"):
            chain.parent_dependency("423072")
    else:
        assert chain.parent_dependency("423072") == expected
    assert len(calls) == (1 if active else 2)


@pytest.mark.parametrize("outcome", ["success", "train-failure", "missing-checkpoint", "final"])
def test_worker_submits_only_one_successor_after_successful_save(chain, plan, monkeypatch, outcome):
    start = 26 if outcome == "final" else 6
    source = plan["source"] if start == 6 else plan["output"]
    complete(source, start)
    manifest = Path(plan["output"]) / "chain.json"
    manifest.write_text(json.dumps(plan))
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "new-nodes")
    monkeypatch.setenv("MASTER_ADDR", "stale-host")
    commands = []
    def run(cmd, **kw):
        commands.append((cmd, kw))
        if cmd[0] == "bash":
            assert kw["env"]["SLURM_JOB_NODELIST"] == "new-nodes"
            assert "MASTER_ADDR" not in kw["env"]
            if outcome == "train-failure":
                raise subprocess.CalledProcessError(1, cmd)
            if outcome != "missing-checkpoint":
                complete(plan["output"], int(kw["env"]["MAX_STEPS"]))
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(stdout="98766\n")
    monkeypatch.setattr(chain.subprocess, "run", run)
    if outcome == "train-failure":
        with pytest.raises(subprocess.CalledProcessError): chain.worker(manifest, start)
    elif outcome == "missing-checkpoint":
        with pytest.raises(ValueError, match="completed"): chain.worker(manifest, start)
    else:
        chain.worker(manifest, start)
    assert len(commands) == (2 if outcome == "success" else 1)
    if outcome == "success":
        assert "--dependency=afterok:98765" in commands[1][0]
        assert commands[1][1]["env"]["MAX_STEPS"] == "26"
        assert not any(k.startswith("SLURM_") for k in commands[1][1]["env"])


def test_read_partial_json_line_and_require_completed_checkpoint(chain, tmp_path):
    (tmp_path / "train.jsonl").write_text('{"step": 6}\n{"step":', encoding="utf-8")
    assert chain.last_step(tmp_path) == 6
    with pytest.raises(ValueError): chain.require_checkpoint(tmp_path, 6)
    (tmp_path / "last.pt").touch()
    chain.require_checkpoint(tmp_path, 6)
    with pytest.raises(ValueError): chain.require_checkpoint(tmp_path, 7)


def test_submission_failure_does_not_record_a_nonexistent_job(chain, plan, monkeypatch):
    def fail(cmd, **kw): raise subprocess.CalledProcessError(1, cmd, stderr="QOSMaxSubmitJobPerUserLimit")
    monkeypatch.setattr(chain.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        chain.submit(plan, Path(plan["output"]) / "chain.json", 6)
    assert not (Path(plan["output"]) / "jobs.jsonl").exists()


@pytest.mark.parametrize("pending", [True, False])
@pytest.mark.parametrize("unfreeze", [False, True])
def test_start_inherits_saved_profile_and_uses_source_final_target(chain, tmp_path, monkeypatch, pending, unfreeze):
    project = tmp_path / "WorldTTN"
    (project / "tools").mkdir(parents=True)
    monkeypatch.setattr(chain, "__file__", str(project / "tools/ttn_slurm_chain.py"))
    python = tmp_path / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv("ROOT", str(tmp_path))
    source = tmp_path / "source"
    source.mkdir()
    args = {"max_steps": 6, "stage": "C", "train_scope": "ttn-visual" if unfreeze else "dit", "tbptt": 2,
            "seed": 3407, "backbone_lr": 1e-6, "config": "configs/custom.json",
            "dataset_root": "shared/example", "data_dir": "shared/raw",
            "vae_cache_dir": "shared/cache", "base_weights": "shared/base.safetensors",
            "sana_config": "configs/sana.yaml", "batch_file": None,
            "text_encoder_device": "cpu", "cross_attn_backend": "math", "activation_offload": "cpu"}
    (source / "run_config.json").write_text(json.dumps({"parallel": "fsdp2", "world_size": 4, "arguments": args,
                                                      "training": {"train_scope": args["train_scope"]}}))
    complete(source, 3 if pending else 6)
    monkeypatch.setattr(chain, "parent_dependency", lambda job: job if pending else None)
    submissions = []
    monkeypatch.setattr(chain, "submit", lambda *values: submissions.append(values))
    chain.start_chain(SimpleNamespace(from_run=str(source), after_job="423072", target_step=500,
                                     segment_steps=10, partition="short", unfreeze=unfreeze))
    assert len(submissions) == 1
    plan, manifest, start, dependency = submissions[0]
    assert start == 6 and dependency == ("423072" if pending else None)
    assert plan["initial_step"] == 6 and plan["target_step"] == 500
    assert manifest.is_file() and Path(plan["output"]).is_dir()
    for field, env_name in chain.PROFILE.items():
        if args.get(field) is not None:
            assert plan["environment"][env_name] == str(args[field])
    assert plan["environment"]["PYTHON"] == str(python)
    assert "BATCH_FILE" not in plan["environment"]
    assert bool(plan.get("unfreeze")) == unfreeze


@pytest.mark.parametrize("change", [{"world_size": 3}, {"parallel": "ddp"}, {"stage": "A"}, {"train_scope": "ttn"}])
def test_start_rejects_incompatible_resume_profile_before_submission(chain, tmp_path, change):
    original = {"max_steps": 6, "stage": "C", "train_scope": "dit"}
    run = {"parallel": "fsdp2", "world_size": 4, "arguments": original}
    for key, value in change.items():
        (original if key in original else run)[key] = value
    (tmp_path / "run_config.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="FSDP2"):
        chain.start_chain(SimpleNamespace(from_run=str(tmp_path), after_job=None, target_step=500,
                                         segment_steps=10, partition="short"))


@pytest.mark.parametrize("hold", [False, True])
def test_fresh_original_ucpe_plan_starts_without_old_weights_and_clamps_warmup_boundary(chain, tmp_path, monkeypatch, hold):
    project = tmp_path/"WorldTTN"
    (project/"tools").mkdir(parents=True)
    config = project/"configs/worldttn/reference_sana_camera.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"ttn": {"stage": "C", "camera_attention": "sana"}}))
    python = tmp_path/"envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    (tmp_path/"datasets/sana-wm-example").mkdir(parents=True)
    monkeypatch.setattr(chain, "__file__", str(project/"tools/ttn_slurm_chain.py"))
    monkeypatch.setenv("ROOT", str(tmp_path))
    monkeypatch.setenv("ADAPTER", "/old/linear-camera.pt")
    monkeypatch.setenv("UNFREEZE", "1")
    calls = []
    monkeypatch.setattr(chain, "submit", lambda *args: calls.append(args))
    chain.fresh_chain(SimpleNamespace(output=str(tmp_path/"new"), dataset_root=None, config=None,
                    warmup_steps=25, target_step=100, segment_steps=10, partition="short", seed=3407,
                    tbptt=2, backbone_lr=1e-6, base_weights=None, warmup_only=hold))
    assert len(calls) == 1
    plan, manifest, start = calls[0]
    assert start == 0 and plan["target_step"] == 25 and plan["environment"]["TRAIN_SCOPE"] == "ttn-visual"
    assert bool(plan.get("joint_target_step")) == (not hold)
    env = chain.environment(plan, 0)
    assert "ADAPTER" not in env and "UNFREEZE" not in env and env["RESUME"] == "0"
    env = chain.environment(plan, 20)
    assert env["MAX_STEPS"] == "25" and env["RESUME"] == "1" and "UNFREEZE" not in env
    assert env["ADAPTER"] == str(Path(plan["output"])/"last.pt")
    assert manifest.is_file()


def test_joint_transition_resets_optimizer_only_in_first_segment(chain, plan):
    plan.update(initial_step=50, target_step=500, unfreeze=True)
    env = chain.environment(plan, 50)
    assert env["MAX_STEPS"] == "60" and env["RESUME"] == "0" and env["UNFREEZE"] == "1"
    assert env["ADAPTER"] == str(Path(plan["source"])/"last.pt")
    env = chain.environment(plan, 60)
    assert env["RESUME"] == "1" and "UNFREEZE" not in env and env["MAX_STEPS"] == "70"
    assert env["ADAPTER"] == str(Path(plan["output"])/"last.pt")


@pytest.mark.parametrize("fail", [False, True])
def test_warmup_submits_joint_phase_only_after_successful_final_checkpoint(chain, plan, monkeypatch, fail):
    plan.update(fresh=True, initial_step=0, target_step=25, joint_target_step=100,
                joint_output=str(Path(plan["output"])/"joint"))
    plan["environment"]["TRAIN_SCOPE"] = "ttn-visual"
    complete(plan["output"], 20)
    manifest = Path(plan["output"])/"chain.json"
    manifest.write_text(json.dumps(plan))
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    calls = []
    def train(cmd, **kw):
        assert kw["env"]["MAX_STEPS"] == "25" and kw["env"]["RESUME"] == "1"
        if fail: raise subprocess.CalledProcessError(1, cmd)
        complete(plan["output"], 25)
    monkeypatch.setattr(chain.subprocess, "run", train)
    monkeypatch.setattr(chain, "start_chain", lambda args: calls.append(args))
    if fail:
        with pytest.raises(subprocess.CalledProcessError): chain.worker(manifest, 20)
    else:
        chain.worker(manifest, 20)
        assert len(calls) == 1 and calls[0].unfreeze and calls[0].target_step == 100
        assert calls[0].from_run == plan["output"] and calls[0].after_job == "98765"
    assert bool(calls) == (not fail)
