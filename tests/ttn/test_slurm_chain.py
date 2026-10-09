"""Test exact-resume launch wiring and failure boundaries without Slurm/GPU."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


def test_batch_entry_uses_project_package_even_with_an_old_checkout_on_pythonpath(tmp_path):
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash not installed")
        bash = Path(candidate)
    project = Path(__file__).resolve().parents[2]
    old = tmp_path / "old checkout"
    for name in ("worldttn", "tools"):
        package = old / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text("# stale package without new checkpoint helpers\n")
    root = tmp_path / "shared root"
    python = root / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$@"\n', encoding="utf-8", newline="\n")
    python.chmod(0o755)
    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "chain.json"
    manifest.write_text(json.dumps({"output": str(output), "initial_step": 5, "target_step": 500}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(ROOT=root.as_posix(), PROJECT_ROOT=project.as_posix(), PYTHON=python.as_posix(),
               REAL_PYTHON=Path(sys.executable).as_posix(), PYTHONPATH=old.as_posix())
    for name in ("PYTHONSAFEPATH", "MSYS2_ENV_CONV_EXCL", "MSYS_NO_PATHCONV"):
        env.pop(name, None)
    result = subprocess.run([str(bash), str(project / "tools/ttn_slurm_chain.sbatch"), str(manifest), "4"],
                            cwd=old, env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=60)
    # Stop before any training/submission; reaching this guard proves local imports work.
    assert result.returncode != 0 and "Invalid segment start" in result.stderr, result.stderr
    status = json.loads((output / "chain-status-local.json").read_text())
    assert status["status"] == "failed" and status["phase"] == "source-validation"
    assert len(list(output.glob("failure-*.json"))) == 1


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


def complete(run, step, world=4):
    run = Path(run)
    (run / "train.jsonl").write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
    folder = run / f"last-resume-{step}"; folder.mkdir(exist_ok=True)
    meta = {"format": "TTN-parallel-resume-v1", "mode": "fsdp2", "world_size": world,
            "resume_dir": folder.name, "checkpoint_id": str(step)}
    torch.save({"format": "TTN-SANA-WM-v0.1", "step": step, "distributed": meta}, run / "last.pt")
    for rank in range(world):
        torch.save({**meta, "step": step, "rank": rank, "rng": {}, "data": {}, "optimizer": {}},
                   folder / f"rank-{rank:05d}.pt")


@pytest.mark.parametrize("world", [None, 3])
def test_submission_preserves_training_and_does_not_inherit_old_allocation(chain, plan, monkeypatch, world):
    if world is not None: plan["world_size"] = world
    expected_world = world or 4
    for key in ("SLURM_JOB_ID", "SLURM_NTASKS", "SBATCH_ARRAY_INX", "MASTER_ADDR", "MASTER_PORT", "TTN_RENDEZVOUS_FILE",
                "CUDA_VISIBLE_DEVICES", "RANK", "BATCH_FILE", "BASE_WEIGHTS", "CONFIG", "UNFREEZE", "DIAGNOSTIC_UNMASK_ALL_VALID",
                "TTN_ENTRY_MODULE", "META_TEST_OUTPUT"):
        monkeypatch.setenv(key, "stale")
    calls = []
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(stdout="98765;cluster\n")
    monkeypatch.setattr(chain.subprocess, "run", run)
    manifest = Path(plan["output"]) / "chain.json"
    assert chain.submit(plan, manifest, 6, "423072") == "98765"
    cmd, kw = calls[0]
    for flag in (f"--nodes={expected_world}", f"--ntasks={expected_world}", "--ntasks-per-node=1", "--gres=gpu:1",
                 "--mem=256G", "--time=01:00:00", "--dependency=afterok:423072", "--kill-on-invalid-dep=yes"):
        assert flag in cmd
    env = kw["env"]
    assert not any(name.startswith(("SLURM_", "SBATCH_")) for name in env)
    for key in ("MASTER_ADDR", "MASTER_PORT", "TTN_RENDEZVOUS_FILE", "CUDA_VISIBLE_DEVICES", "RANK", "BATCH_FILE", "BASE_WEIGHTS", "CONFIG", "UNFREEZE", "DIAGNOSTIC_UNMASK_ALL_VALID",
                "TTN_ENTRY_MODULE", "META_TEST_OUTPUT"):
        assert key not in env
    assert env["ADAPTER"] == str(Path(plan["source"]) / "last.pt")
    assert env["MAX_STEPS"] == "16" and env["SAVE_EVERY"] == "28"
    for key, value in plan["environment"].items():
        assert env[key] == value
    record = json.loads((Path(plan["output"]) / "jobs.jsonl").read_text())
    assert record["stop_step"] == 16 and record["world_size"] == expected_world
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
@pytest.mark.parametrize("world", [3, 4])
def test_worker_submits_only_one_successor_after_successful_save(chain, plan, monkeypatch, outcome, world):
    plan["world_size"] = world
    start = 26 if outcome == "final" else 6
    source = plan["source"] if start == 6 else plan["output"]
    complete(source, start, world)
    manifest = Path(plan["output"]) / "chain.json"
    manifest.write_text(json.dumps(plan))
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_NTASKS", str(world))
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
                complete(plan["output"], int(kw["env"]["MAX_STEPS"]), world)
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
        assert f"--nodes={world}" in commands[1][0] and f"--ntasks={world}" in commands[1][0]
        assert "--dependency=afterok:98765" in commands[1][0]
        assert commands[1][1]["env"]["MAX_STEPS"] == "26"
        assert not any(k.startswith("SLURM_") for k in commands[1][1]["env"])


def test_read_partial_json_line_and_require_completed_checkpoint(chain, tmp_path):
    (tmp_path / "train.jsonl").write_text('{"step": 6}\n{"step":', encoding="utf-8")
    assert chain.last_step(tmp_path) == 6
    with pytest.raises(ValueError): chain.require_checkpoint(tmp_path, 6)
    complete(tmp_path, 6)
    chain.require_checkpoint(tmp_path, 6)
    with pytest.raises(ValueError): chain.require_checkpoint(tmp_path, 7)


def test_submission_failure_does_not_record_a_nonexistent_job(chain, plan, monkeypatch):
    def fail(cmd, **kw): raise subprocess.CalledProcessError(1, cmd, stderr="QOSMaxSubmitJobPerUserLimit")
    monkeypatch.setattr(chain.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        chain.submit(plan, Path(plan["output"]) / "chain.json", 6)
    assert not (Path(plan["output"]) / "jobs.jsonl").exists()


@pytest.mark.parametrize("world", [1, 0, -3, True, "3", 3.5])
def test_invalid_world_stops_before_submission_or_fresh_output(chain, plan, monkeypatch, world):
    plan["world_size"] = world
    monkeypatch.setattr(chain.subprocess, "run", lambda *a, **kw: pytest.fail("invalid plan submitted"))
    with pytest.raises(ValueError, match="FSDP2"):
        chain.submit(plan, Path(plan["output"]) / "chain.json", 6)
    with pytest.raises(ValueError, match="FSDP2"):
        chain.fresh_chain(SimpleNamespace(world_size=world))
    assert not (Path(plan["output"]) / "jobs.jsonl").exists()


@pytest.mark.parametrize("failure", ["allocation", "source-world", "saved-world", "missing-rank"])
def test_three_rank_failure_never_submits_successor(chain, plan, monkeypatch, failure):
    plan["world_size"] = 3
    complete(plan["source"], 6, 4 if failure == "source-world" else 3)
    manifest = Path(plan["output"]) / "chain.json"
    manifest.write_text(json.dumps(plan))
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_NTASKS", "4" if failure == "allocation" else "3")
    calls = []
    def run(cmd, **kw):
        assert cmd[0] == "bash", "failed checkpoint submitted a successor"
        calls.append(cmd)
        complete(plan["output"], 16, 4 if failure == "saved-world" else 3)
        if failure == "missing-rank":
            (Path(plan["output"]) / "last-resume-16/rank-00002.pt").unlink()
    monkeypatch.setattr(chain.subprocess, "run", run)
    with pytest.raises((ValueError, FileNotFoundError)):
        chain.worker(manifest, 6)
    assert len(calls) == (0 if failure in ("allocation", "source-world") else 1)
    status = json.loads((Path(plan["output"]) / "chain-status-98765.json").read_text())
    assert status["status"] == "failed"
    assert not (Path(plan["output"]) / "jobs.jsonl").exists()


@pytest.mark.parametrize("failure", ["override", "checkpoint-world"])
def test_start_rejects_world_change_before_creating_output(chain, tmp_path, monkeypatch, failure):
    source = tmp_path / "source"
    source.mkdir()
    complete(source, 6, 4 if failure == "checkpoint-world" else 3)
    (source / "run_config.json").write_text(json.dumps({"parallel": "fsdp2", "world_size": 3,
        "arguments": {"stage": "C", "train_scope": "dit", "max_steps": 6}}))
    output = tmp_path / "new"
    monkeypatch.setattr(chain.subprocess, "run", lambda *a, **kw: pytest.fail("incompatible resume submitted"))
    with pytest.raises(ValueError, match="world"):
        chain.start_chain(SimpleNamespace(from_run=str(source), after_job=None, target_step=500,
            segment_steps=2, partition="day", output=str(output), world_size=4 if failure == "override" else 3))
    assert not output.exists()


@pytest.mark.parametrize("pending", [True, False])
@pytest.mark.parametrize("unfreeze", [False, True])
@pytest.mark.parametrize("world", [3, 4])
def test_start_inherits_saved_profile_and_uses_source_final_target(chain, tmp_path, monkeypatch, pending, unfreeze, world):
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
    (source / "run_config.json").write_text(json.dumps({"parallel": "fsdp2", "world_size": world, "arguments": args,
                                                      "training": {"train_scope": args["train_scope"]}}))
    complete(source, 3 if pending else 6, world)
    monkeypatch.setattr(chain, "parent_dependency", lambda job: job if pending else None)
    submissions = []
    monkeypatch.setattr(chain, "submit", lambda *values: submissions.append(values))
    chain.start_chain(SimpleNamespace(from_run=str(source), after_job="423072", target_step=500,
                                     segment_steps=10, partition="short", unfreeze=unfreeze,
                                     world_size=None if pending else world))
    assert len(submissions) == 1
    plan, manifest, start, dependency = submissions[0]
    assert start == 6 and dependency == ("423072" if pending else None)
    assert plan["initial_step"] == 6 and plan["target_step"] == 500
    assert plan["world_size"] == world
    assert manifest.is_file() and Path(plan["output"]).is_dir()
    for field, env_name in chain.PROFILE.items():
        if args.get(field) is not None:
            assert plan["environment"][env_name] == str(args[field])
    assert plan["environment"]["PYTHON"] == str(python)
    assert "BATCH_FILE" not in plan["environment"]
    assert bool(plan.get("unfreeze")) == unfreeze


@pytest.mark.parametrize("change", [{"world_size": 1}, {"parallel": "ddp"}, {"stage": "A"}, {"train_scope": "ttn"}])
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
@pytest.mark.parametrize("world", [3, 4])
def test_fresh_original_ucpe_plan_starts_without_old_weights_and_clamps_warmup_boundary(chain, tmp_path, monkeypatch, hold, world):
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
                    tbptt=2, backbone_lr=1e-6, base_weights=None, warmup_only=hold, world_size=world))
    assert len(calls) == 1
    plan, manifest, start = calls[0]
    assert start == 0 and plan["target_step"] == 25 and plan["environment"]["TRAIN_SCOPE"] == "ttn-new"
    assert plan["world_size"] == world
    assert plan["environment"]["OPTIMIZER_POLICY"] == "origin"
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


@pytest.mark.parametrize("world", [None, 3])
def test_zero_warmup_starts_joint_at_step_one_without_checkpoint_or_unfreeze(chain, tmp_path, monkeypatch, world):
    project = tmp_path / "WorldTTN"
    (project / "tools").mkdir(parents=True)
    config = project / "configs/worldttn/reference_sana_camera.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"ttn": {"stage": "C", "camera_attention": "sana"}}))
    python = tmp_path / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True); python.touch()
    (tmp_path / "datasets/sana-wm-example").mkdir(parents=True)
    monkeypatch.setattr(chain, "__file__", str(project / "tools/ttn_slurm_chain.py"))
    monkeypatch.setenv("ROOT", str(tmp_path))
    calls = []; monkeypatch.setattr(chain, "submit", lambda *args: calls.append(args))
    args = SimpleNamespace(output=str(tmp_path / "new"), dataset_root=None, config=None,
        warmup_steps=0, target_step=500, segment_steps=10, partition="short", seed=3407,
        tbptt=2, backbone_lr=1e-6, base_weights=None, warmup_only=False, eval_every=25)
    if world is not None: args.world_size = world
    chain.fresh_chain(args)
    plan = calls[0][0]
    assert plan["target_step"] == 500 and plan["environment"]["TRAIN_SCOPE"] == "dit"
    assert plan["world_size"] == (world or 4)
    assert "joint_target_step" not in plan
    assert plan["evaluation"]["keep_model_steps"] == [25, 50, 100, 250, 500]
    env = chain.environment(plan, 0)
    assert env["OPTIMIZER_POLICY"] == "origin" and env["RESUME"] == "0"
    assert "UNFREEZE" not in env and "ADAPTER" not in env
    assert chain.environment(plan, 10)["RESUME"] == "1"


def test_evaluation_snapshot_retention_preserves_key_points_and_releases_other_models(chain, plan, monkeypatch):
    import tools.ttn_eval_snapshot as snapshots
    calls = []
    monkeypatch.setattr(snapshots, "snapshot_training_run", lambda source, **kw: calls.append(kw) or Path(source) / "snapshot")
    plan["evaluation"] = {"every": 25, "seed": 3407, "steps": 20, "cases": 1,
        "fixed_cases": "fixed.pt", "output": str(Path(plan["output"]) / "eval"), "keep_model_steps": [25, 50, 100, 250, 500]}
    monkeypatch.setattr(chain.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="123456\n"))
    chain.submit_evaluation(plan, 75, "123455")
    chain.submit_evaluation(plan, 100, "123455")
    assert calls == [{"retain_model": False}, {"retain_model": True}]


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("world", [3, 4])
def test_warmup_submits_joint_phase_only_after_successful_final_checkpoint(chain, plan, monkeypatch, fail, world):
    plan.update(fresh=True, initial_step=0, target_step=25, joint_target_step=100,
                joint_output=str(Path(plan["output"])/"joint"), world_size=world)
    plan["environment"]["TRAIN_SCOPE"] = "ttn-visual"
    complete(plan["output"], 20, world)
    manifest = Path(plan["output"])/"chain.json"
    manifest.write_text(json.dumps(plan))
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_NTASKS", str(world))
    calls = []
    def train(cmd, **kw):
        assert kw["env"]["MAX_STEPS"] == "25" and kw["env"]["RESUME"] == "1"
        if fail: raise subprocess.CalledProcessError(1, cmd)
        complete(plan["output"], 25, world)
    monkeypatch.setattr(chain.subprocess, "run", train)
    monkeypatch.setattr(chain, "start_chain", lambda args: calls.append(args))
    if fail:
        with pytest.raises(subprocess.CalledProcessError): chain.worker(manifest, 20)
    else:
        chain.worker(manifest, 20)
        assert len(calls) == 1 and calls[0].unfreeze and calls[0].target_step == 100
        assert calls[0].from_run == plan["output"] and calls[0].after_job == "98765"
        assert calls[0].world_size == world
    assert bool(calls) == (not fail)


def test_periodic_boundaries_and_first_unfrozen_step_are_exact(chain, plan):
    plan["evaluation"] = {"every": 25}
    assert chain.segment_stop(plan, 20) == 25
    assert chain.environment(plan, 20)["MAX_STEPS"] == "25"
    plan.update(initial_step=50, target_step=500, unfreeze=True)
    assert chain.segment_stop(plan, 50) == 51
    assert chain.segment_stop(plan, 71) == 75
    assert chain.segment_stop(plan, 499) == 500


@pytest.mark.parametrize("long_training", [False, True])
def test_evaluation_submission_pins_snapshot_and_uses_separate_single_gpu(chain, plan, monkeypatch, long_training):
    import tools.ttn_eval_snapshot as snapshots
    snapshot = Path(plan["output"]) / "immutable-snapshot"
    monkeypatch.setattr(snapshots, "snapshot_training_run", lambda source, **kw: snapshot)
    plan["evaluation"] = {"every": 25, "seed": 3407, "steps": 20, "cases": 1,
                          "fixed_cases": "shared/fixed-cases.pt", "output": str(Path(plan["output"]) / "eval")}
    if long_training:
        plan.update(time_limit="12:00:00", memory="320G")
        plan["evaluation"].update(frames=121, cfg_scale=1., steps=4)
    monkeypatch.setenv("SLURM_JOB_ID", "old")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "old")
    monkeypatch.setenv("CAMERA_ATTENTION", "linear")
    monkeypatch.setenv("CAMERA_ABLATION", "1")
    calls = []
    def submit(cmd, **kw):
        calls.append((cmd, kw["env"]))
        return SimpleNamespace(stdout="123456\n")
    monkeypatch.setattr(chain.subprocess, "run", submit)
    chain.submit_evaluation(plan, 25, "123455")
    cmd, env = calls[0]
    assert "--nodes=1" in cmd and "--ntasks=1" in cmd and "--gres=gpu:1" in cmd
    assert "--dependency=afterok:123455" in cmd
    assert env["TRAINING_RUN"] == str(snapshot) and env["FIXED_CASES"] == "shared/fixed-cases.pt"
    assert env["COMMAND"] == "stage-evaluate" and env["PYTHON"] == plan["environment"]["PYTHON"]
    assert env["FRAMES"] == ("121" if long_training else "61")
    assert env["CFG_SCALE"] == ("1.0" if long_training else "4.5")
    assert ("--time=12:00:00" if long_training else "--time=01:00:00") in cmd
    assert not any(k in env for k in ("SLURM_JOB_ID", "ADAPTER", "CUDA_VISIBLE_DEVICES", "CAMERA_ATTENTION", "CAMERA_ABLATION"))
    row = json.loads((Path(plan["output"]) / "eval_jobs.jsonl").read_text())
    assert row["step"] == 25 and row["job"] == "123456"
    if long_training:
        chain.submit(plan, Path(plan["output"])/"chain.json", 6)
        assert "--time=12:00:00" in calls[1][0] and "--mem=320G" in calls[1][0]


def test_checkpoint_exists_but_saved_step_is_stale_stops_chain(chain, tmp_path):
    complete(tmp_path, 160)
    (tmp_path / "train.jsonl").write_text('{"step":170}\n')
    with pytest.raises(ValueError, match="checkpoint step 160"):
        chain.require_checkpoint(tmp_path, 170)


def test_start_recovers_from_saved_step_without_overwriting_original_logs(chain, tmp_path, monkeypatch):
    project = tmp_path / "WorldTTN"; (project / "tools").mkdir(parents=True)
    monkeypatch.setattr(chain, "__file__", str(project / "tools/ttn_slurm_chain.py"))
    python = tmp_path / "envs/worldttn/bin/python"; python.parent.mkdir(parents=True); python.touch()
    monkeypatch.setenv("ROOT", str(tmp_path))
    source = tmp_path / "old"; source.mkdir(); complete(source, 160)
    with (source / "train.jsonl").open("a") as stream: stream.write('{"step":170}\n')
    (source / "run_config.json").write_text(json.dumps({"parallel": "fsdp2", "world_size": 4,
        "arguments": {"max_steps": 170, "stage": "C", "train_scope": "dit"}}))
    calls = []; monkeypatch.setattr(chain, "submit", lambda *a: calls.append(a))
    chain.start_chain(SimpleNamespace(from_run=str(source), after_job=None, target_step=500,
                    segment_steps=10, partition="short"))
    plan = calls[0][0]
    assert plan["initial_step"] == 160 and plan["source_checkpoint"]["unsaved_steps"] == [161, 170]
    assert Path(plan["output"]) != source and chain.last_step(source) == 170


@pytest.mark.parametrize("damage", ["missing", "stage", "train_scope"])
def test_saved_training_record_must_exist_and_match_model_metadata(chain, tmp_path, damage):
    complete(tmp_path, 160)
    report = {"step": 160, "stage": "C", "train_scope": "dit"}
    row = {"step": 160, "stage": "C", "train_scope": "dit"}
    if damage == "missing": row["step"] = 170
    else: row[damage] = "wrong"
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="training record"):
        chain.verify_training_record(tmp_path, report)
