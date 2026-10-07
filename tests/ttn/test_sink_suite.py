"""Sink experiment plans are immutable, matched and single-GPU serial."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from worldttn.core import TTNConfig
from worldttn.evaluation import file_sha256


def snapshot(tmp_path):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    torch.save({"config": TTNConfig(stage="C", camera_attention="sana").to_dict(), "step": 100,
                "stage": "C", "base_sha256": "base"}, directory / "last.pt")
    (directory / "run_config.json").write_text(json.dumps({"base": {"sha256": "base"}}))
    (directory / "train.jsonl").write_text('{"step": 100, "stage": "C"}\n')
    (directory / "snapshot.json").write_text(json.dumps({"format": "TTN-evaluation-snapshot-v2",
                                                       "source": str(tmp_path), "step": 100, "retain_model": True}))
    return directory


def smoke_result(tmp_path, cfg_scale=4.5):
    from worldttn.sink import SinkOptions
    digest = "a"*64
    shifts = [12]*(2 if cfg_scale > 1 else 1)
    chunks = []
    for index, (start, end) in enumerate(((0, 4), (4, 7), (7, 10), (10, 13), (13, 16))):
        anchors = []
        for block in (3, 7, 11, 15, 19):
            sink = dict(active=index == 4, mode="protected", gain=.1)
            if index == 4:
                sink.update(position="temporal-realign", temporal_shift=shifts,
                            reference_source="observed-prefill", reference_sha256=digest,
                            effective_delta_norm=.2 if block == 19 else 0.)
            anchors.append(dict(block=block, sink=sink,
                sink_trajectory=[dict(sink, call=call) for call in (0, 2, 3)] if index == 4 else []))
        chunks.append(dict(chunk=index, start=start, end=end, anchors=anchors))
    protocol = dict(status="completed", step=100, stage="C", frames=16, raw_frames=121,
                    steps=4, cfg_scale=cfg_scale, camera_attention="sana", history_source="generated",
                    ttn_ablation="full", execution="reference/reference", state_diagnostics=True,
                    tla_sink=asdict(SinkOptions("protected")), checkpoint_sha256="b"*64)
    row = dict(method="ttn", commits=6, chunks=chunks,
               sink_reference_sha256=digest, sink_reference_verified=True)
    result = dict(protocol=protocol, episodes=[dict(method="sana"), row])
    (tmp_path / "summary.json").write_text(json.dumps(result))
    video = tmp_path / "videos"
    video.mkdir()
    (video / "comparison.json").write_text(json.dumps(dict(status="completed",
        encoded_video_validation={name: {"frames": 121} for name in ("sana.mp4", "ttn.mp4", "comparison.mp4")})))
    return result


@pytest.mark.parametrize("cfg_scale", [1., 4.5])
def test_smoke_checks_actual_noisy_effect_and_cli_publishes_report(tmp_path, cfg_scale, capsys):
    from tools.ttn_submit_sink import check_smoke, main
    smoke_result(tmp_path, cfg_scale)
    checked = check_smoke(tmp_path)
    assert checked["status"] == "passed" and checked["nonzero_noisy_anchors"] == [19]
    assert checked["chunks"][4]["temporal_shift"] == [12]*(2 if cfg_scale > 1 else 1)
    main(["--check-smoke", str(tmp_path)])
    assert json.loads((tmp_path / "smoke-validation.json").read_text()) == checked
    assert '"status": "passed"' in capsys.readouterr().out


def test_smoke_protocol_error_identifies_changed_and_missing_fields(tmp_path):
    from tools.ttn_submit_sink import main
    result = smoke_result(tmp_path)
    result["protocol"]["steps"] = 20
    result["protocol"].pop("state_diagnostics")
    (tmp_path / "summary.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="smoke protocol mismatch"):
        main(["--check-smoke", str(tmp_path)])
    failure = json.loads((tmp_path / "smoke-validation.json").read_text())
    mismatch = json.loads(failure["error"].split(": ", 1)[1])
    assert mismatch == {"steps": {"actual": 20, "expected": 4},
                        "state_diagnostics": {"actual": "<missing>", "expected": True}}
    assert failure["status"] == "failed" and failure["first_exception"]


@pytest.mark.parametrize("corrupt", ["early", "fifth", "missing-anchor", "shift", "mode", "gain", "position",
    "hash", "unverified", "zero-noisy", "missing-noisy", "nan", "commit", "video"])
def test_smoke_rejects_inactive_or_ineffective_sink_and_records_failure(tmp_path, corrupt):
    from tools.ttn_submit_sink import main
    result = smoke_result(tmp_path)
    row = result["episodes"][1]
    last = row["chunks"][-1]["anchors"]
    if corrupt == "early": row["chunks"][0]["anchors"][0]["sink"]["active"] = True
    elif corrupt == "fifth": last[0]["sink"]["active"] = False
    elif corrupt == "missing-anchor": last.pop()
    elif corrupt == "shift": last[0]["sink"]["temporal_shift"] = [11, 11]
    elif corrupt == "mode": last[0]["sink"]["mode"] = "zero"
    elif corrupt == "gain": last[0]["sink"]["gain"] = .25
    elif corrupt == "position": last[0]["sink"]["position"] = "absolute"
    elif corrupt == "hash": last[0]["sink"]["reference_sha256"] = "c"*64
    elif corrupt == "unverified": row["sink_reference_verified"] = False
    elif corrupt == "zero-noisy":
        for anchor in last:
            for call in anchor["sink_trajectory"]: call["effective_delta_norm"] = 0.
    elif corrupt == "missing-noisy": last[0]["sink_trajectory"] = []
    elif corrupt == "nan": last[0]["sink_trajectory"][0]["effective_delta_norm"] = float("nan")
    elif corrupt == "commit": row["commits"] = 7
    elif corrupt == "video":
        (tmp_path / "videos/comparison.json").write_text('{"status":"running"}')
    (tmp_path / "summary.json").write_text(json.dumps(result))
    with pytest.raises(ValueError): main(["--check-smoke", str(tmp_path)])
    failure = json.loads((tmp_path / "smoke-validation.json").read_text())
    assert failure["status"] == "failed" and failure["first_exception"]


def test_custom_suite_requires_pinned_snapshot_and_defines_scale_controls(tmp_path):
    from tools.ttn_submit_sink import prepare
    directory = snapshot(tmp_path)
    case = Path(__file__).resolve().parents[2] / "assets/worldttn/study_static/case.json"
    plan = prepare(training_run=directory, case=case, output=tmp_path / "suite")
    assert plan["kind"] == "custom" and plan["step"] == 100
    assert plan["checkpoint_sha256"] == file_sha256(directory / "last.pt")
    assert plan["variants"] == ["full", "zero-010", "absolute-010", "aligned-010", "zero-025", "aligned-025"]
    assert plan["interventions"]["zero-010"]["gain"] == plan["interventions"]["aligned-010"]["gain"] == .1
    assert plan["interventions"]["absolute-010"]["position"] == "absolute"
    assert plan["array"] == "1-5%1" and not Path(plan["output"]).exists()
    (directory / "snapshot.json").unlink()
    with pytest.raises(ValueError, match="snapshot"):
        prepare(training_run=directory, case=case, output=tmp_path / "suite")


def completed_suite(tmp_path):
    from tools.ttn_submit_sink import prepare
    plan = prepare(training_run=snapshot(tmp_path), output=tmp_path / "suite")
    root = Path(plan["output"])
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    identity = dict(case_id="custom/scene", seed=3407, input_sha256={"latent": "same"},
                    initial_noise_sha256="noise", base_sha256="base")
    for name in plan["variants"]:
        folder = root / name
        folder.mkdir()
        methods = ("sana", "ttn") if name == "full" else ("ttn",)
        protocol = dict(status="completed", checkpoint_sha256=plan["checkpoint_sha256"], step=100,
                        stage="C", frames=61, steps=20, seed=3407, cfg_scale=4.5, cached_blocks=2,
                        cross_attn_backend="math", history_source="generated", ttn_ablation="full",
                        camera_attention="sana", case=plan["case_data"], prompt=plan["prompt"], flow_shift=9.8,
                        tla_sink=plan["interventions"][name], execution="reference/reference")
        episodes = [dict(identity, method=method, chunks=[], timing={"seconds": 1}) for method in methods]
        for row in episodes:
            if row["method"] == "ttn" and plan["interventions"][name]["mode"] == "protected":
                row.update(sink_reference_sha256="a"*64, sink_reference_verified=True)
        (folder / "summary.json").write_text(json.dumps(dict(protocol=protocol, episodes=episodes, metrics=None)))
        (root / f"{name}-status.json").write_text('{"status":"completed"}')
    return root, plan


def test_collector_requires_complete_matching_protocols_and_preserves_null_gt(tmp_path):
    from tools.ttn_submit_sink import collect
    root, plan = completed_suite(tmp_path)
    result = collect(root)
    assert result["status"] == "completed" and result["metric_space"] == "none: no future GT"
    assert result["results"]["aligned-010"]["gt_metrics"] is None
    path = root / "aligned-010/summary.json"
    changed = json.loads(path.read_text())
    changed["episodes"][0]["initial_noise_sha256"] = "wrong"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="noise|identity"):
        collect(root)


def test_collector_requires_completed_readonly_reference_audit(tmp_path):
    from tools.ttn_submit_sink import collect
    root, _ = completed_suite(tmp_path)
    path = root / "aligned-010/summary.json"
    changed = json.loads(path.read_text())
    changed["episodes"][0]["sink_reference_verified"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reference audit"):
        collect(root)


def test_submission_protects_baseline_dependency_and_stops_on_failure(monkeypatch, tmp_path):
    from tools import ttn_submit_sink as tool
    plan = tool.prepare(training_run=snapshot(tmp_path), output=tmp_path / "suite")
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return type("Result", (), {"stdout": f"{901+len(calls)}\n"})()
    monkeypatch.setattr(tool.subprocess, "run", submit)
    monkeypatch.setattr(tool, "collect", lambda root: {"status": "running"})
    jobs = tool.submit(plan, partition="day", root=tmp_path)
    assert len(calls) == 2 and jobs["baseline"] == "902"
    assert "--dependency=afterok:902" in calls[1] and "--array=1-5%1" in calls[1]
    assert all("--partition=day" in command for command in calls)


def test_dataset_suite_reuses_exact_milestone_protocol(monkeypatch, tmp_path):
    from tools import ttn_submit_sink as tool
    directory = snapshot(tmp_path)
    source = dict(snapshot=str(directory), checkpoint_sha256=file_sha256(directory / "last.pt"), step=100,
                  steps=12, cfg_scale=3., cached_blocks=4, seed=22, eval_cases=2,
                  fixed_cases="fixed.pt", fixed_cases_sha256="cases", compile={}, cross_attn_backend="math")
    monkeypatch.setattr(tool, "prepare_mechanisms", lambda *a, **kw: source)
    plan = tool.prepare(evaluation=tmp_path / "evaluation", output=tmp_path / "suite")
    assert plan["kind"] == "dataset" and plan["steps"] == 12 and plan["cached_blocks"] == 4
    assert plan["seed"] == 22 and plan["eval_cases"] == 2


@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_summary_does_not_override_worker_completion_or_failure(tmp_path, status):
    from tools.ttn_submit_sink import collect
    root, _ = completed_suite(tmp_path)
    (root / "full-status.json").write_text(json.dumps({"status": status, "first_exception": "original trace"}))
    result = collect(root)
    assert result["status"] == ("failed" if status == "failed" else "running")
    assert result["variants"]["full"]["status"] == status
    assert result["variants"]["full"]["first_exception"] == "original trace"


@pytest.mark.parametrize("name", ["full", "aligned-010"])
@pytest.mark.parametrize("status", ["pending", "running", "failed"])
def test_parallel_collector_ignores_other_workers_partial_output(tmp_path, name, status):
    from tools.ttn_submit_sink import collect
    root, _ = completed_suite(tmp_path)
    (root / name / "summary.json").write_text('{"protocol":')
    (root / f"{name}-status.json").write_text(json.dumps({"status": status}))
    result = collect(root)
    assert result["status"] == ("failed" if status == "failed" else "running")
    assert name not in result["results"]


def test_worker_validates_own_result_before_completion_and_releases_failed_lock(tmp_path):
    from tools.ttn_submit_sink import _publish
    root, _ = completed_suite(tmp_path)
    (root / "aligned-010-status.json").write_text('{"status":"running"}')
    (root / "zero-025-status.json").write_text('{"status":"running"}')
    (root / "zero-025/summary.json").write_text('{"protocol":')
    result = _publish(root, validating="aligned-010")
    assert result["status"] == "running" and "aligned-010" in result["results"]
    assert "zero-025" not in result["results"]
    path = root / "aligned-010/summary.json"
    invalid = json.loads(path.read_text())
    invalid["episodes"][0]["sink_reference_verified"] = False
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="reference audit"):
        _publish(root, validating="aligned-010")
    assert not (root / ".sink-summary.lock").exists()
    assert json.loads((root / "aligned-010-status.json").read_text())["status"] == "running"


def test_five_parallel_sink_workers_publish_complete_aggregate(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import subprocess
    import sys
    root, plan = completed_suite(tmp_path)
    for name in plan["variants"][1:]:
        (root / f"{name}-status.json").write_text('{"status":"running"}')
    script = (
        "import sys\nfrom pathlib import Path\n"
        "from tools.ttn_submit_sink import atomic_json, _publish\n"
        "root, name = Path(sys.argv[1]), sys.argv[2]\n"
        "_publish(root, validating=name)\n"
        "atomic_json(root / f'{name}-status.json', {'status': 'completed'})\n"
        "_publish(root)\n"
    )
    def complete(name):
        return subprocess.run([sys.executable, "-c", script, str(root), name],
                              capture_output=True, text=True, timeout=60)
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(complete, plan["variants"][1:]))
    for result in results: assert result.returncode == 0, result.stderr
    result = json.loads((root / "summary.json").read_text())
    assert result["status"] == "completed" and len(result["results"]) == 6
    assert all(item["status"] == "completed" for item in result["variants"].values())
    assert not (root / ".sink-summary.lock").exists()


@pytest.mark.parametrize("changed_prefix", [False, True])
def test_worker_reuses_custom_baseline_and_requires_validated_videos(tmp_path, monkeypatch, changed_prefix):
    from tools import ttn_submit_sink as tool
    from tools import ttn_custom_inference
    root, plan = completed_suite(tmp_path)
    (root / "aligned-010-status.json").write_text('{"status":"pending"}')
    baseline = torch.ones(1, 1, 61, 2, 2)
    baseline[:, :, :1] = 0
    full_row = json.loads((root / "full/summary.json").read_text())["episodes"][1]
    torch.save(dict(latents=baseline, method="ttn", case_id=full_row["case_id"], seed=full_row["seed"]),
               root / "full/case-000-ttn.pt")
    calls = []
    def infer(args):
        calls.append(args)
        output = Path(args[args.index("--output") + 1])
        current = baseline.clone()
        current[:, :, 13:] += .2  # After-activation differences are allowed.
        if changed_prefix: current[:, :, 8] += .05
        torch.save(dict(latents=current, method="ttn", case_id=full_row["case_id"], seed=full_row["seed"]),
                   output / "case-000-ttn.pt")
        for name in ("videos", "videos-vs-ttn-baseline"):
            folder = output / name
            folder.mkdir()
            (folder / "comparison.json").write_text(json.dumps({"status": "completed", "encoded_video_validation": {"comparison.mp4": {"frames": 481}}}))
    monkeypatch.setattr(ttn_custom_inference, "main", infer)
    if changed_prefix:
        with pytest.raises(AssertionError, match="first four chunks"):
            tool.run_variant(root, 3)
        assert json.loads((root / "aligned-010-status.json").read_text())["status"] == "failed"
        assert tool.collect(root)["status"] == "failed"
        return
    assert tool.run_variant(root, 3)["status"] == "completed"
    assert calls[0][calls[0].index("--reuse-baseline") + 1] == str(root / "full")
    assert calls[0][calls[0].index("--tla-sink") + 1] == "protected"
    assert calls[0][calls[0].index("--sink-position") + 1] == "temporal-realign"
    status = json.loads((root / "aligned-010-status.json").read_text())
    assert status["status"] == "completed" and status["phase"] == "completed"
    assert status["prefix_13_validation"]["max_abs_difference"] == 0
    result = tool.collect(root)["results"]["aligned-010"]
    assert result["prefix_13_validation"]["status"] == "passed" and result["gt_metrics"] is None


def test_failed_worker_preserves_first_trace_and_blocks_following_inference(tmp_path, monkeypatch):
    from tools import ttn_submit_sink as tool
    from tools import ttn_custom_inference
    root, _ = completed_suite(tmp_path)
    for name in ("zero-010", "absolute-010"):
        (root / f"{name}-status.json").write_text('{"status":"pending"}')
    calls = []
    def infer(args):
        calls.append(args)
        raise RuntimeError("first decoder failure")
    monkeypatch.setattr(ttn_custom_inference, "main", infer)
    with pytest.raises(RuntimeError, match="decoder failure"): tool.run_variant(root, 1)
    status = json.loads((root / "zero-010-status.json").read_text())
    assert "first decoder failure" in status["first_exception"]
    tool.mark_failed(root, 1, 137)
    assert json.loads((root / "zero-010-status.json").read_text())["first_exception"] == status["first_exception"]
    with pytest.raises(ValueError, match="prior sink task failed"): tool.run_variant(root, 2)
    assert len(calls) == 1 and tool.collect(root)["status"] == "failed"


def test_worker_cannot_bypass_baseline_decode_barrier(tmp_path, monkeypatch):
    from tools import ttn_submit_sink as tool
    from tools import ttn_custom_inference
    root, _ = completed_suite(tmp_path)
    (root / "full-status.json").write_text('{"status":"running", "phase":"decode"}')
    (root / "zero-010-status.json").write_text('{"status":"pending"}')
    def forbidden(args): raise AssertionError("inference ran before the baseline decode completed")
    monkeypatch.setattr(ttn_custom_inference, "main", forbidden)
    with pytest.raises(ValueError, match="baseline must complete"): tool.run_variant(root, 1)


def test_failed_baseline_submission_does_not_submit_interventions(tmp_path, monkeypatch):
    from tools import ttn_submit_sink as tool
    plan = tool.prepare(training_run=snapshot(tmp_path), output=tmp_path / "suite")
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        raise tool.subprocess.CalledProcessError(1, command, stderr="submission failed")
    monkeypatch.setattr(tool.subprocess, "run", run)
    with pytest.raises(tool.subprocess.CalledProcessError): tool.submit(plan, root=tmp_path)
    assert len(calls) == 1
    failure = json.loads((Path(plan["output"]) / "submission-status.json").read_text())
    assert failure["status"] == "failed" and "CalledProcessError" in failure["first_exception"]
    result = tool.collect(plan["output"])
    assert result["status"] == "failed"
    assert result["submission"]["first_exception"] == failure["first_exception"]
    assert result["submission"]["stderr"] == "submission failed"


def dataset_suite(tmp_path):
    from tools import ttn_submit_sink as tool
    from worldttn.evaluation import latent_metrics
    from worldttn.mechanism_evaluation import IDENTITY
    directory = snapshot(tmp_path)
    fixed = tmp_path / "fixed-cases.pt"
    fixed.write_bytes(b"pinned cases")
    source = tmp_path / "milestone"
    (source / "long").mkdir(parents=True)
    specs = tool.interventions()
    protocol = {key: "same" for key in IDENTITY}
    protocol.update(stage="C", step=100, frames=61, noise_frames=61, steps=20, cfg_scale=4.5,
                    cached_blocks=2, flow_shift=9.8, training_latent_frames=13,
                    training_run=str(directory), checkpoint_sha256=file_sha256(directory / "last.pt"),
                    fixed_cases_sha256=file_sha256(fixed), ttn_camera_attention="sana", cross_attn_backend="math",
                    state_diagnostics=True, history_source="generated", ttn_ablation="full",
                    tla_sink=specs["full"])
    gt = torch.zeros(1, 1, 61, 1, 1)
    def episode(method, value):
        generated = gt.clone()
        generated[:, :, 1:] = 1. if method == "sana" else 2.
        if method == "ttn": generated[:, :, 13:] = value ** .5
        return dict(method=method, case_id="case", seed=3407, input_sha256={"observation": "same"},
                    initial_noise_sha256="noise", base_sha256="base", timing={"seconds": 1},
                    metrics=latent_metrics(generated, gt, [], training_frames=13),
                    prefix_13_metrics=latent_metrics(generated[:, :, :13], gt[:, :, :13], [], training_frames=13))
    original = dict(protocol=protocol, episodes=[episode("sana", 1.), episode("ttn", 4.)])
    (source / "long/summary.json").write_text(json.dumps(original))
    (source / "long/manifest.json").write_text('{"cases":[{"seed":3407}]}')
    (source / "summary.json").write_text(json.dumps({"status": "completed", "results": {
        "long": {"metrics": {"mean_future_latent_mse": {"paired_count": 1}}}}}))
    plan = tool.prepare(evaluation=source, fixed_cases=fixed, output=tmp_path / "suite")
    root = Path(plan["output"])
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan))
    for name, spec in specs.items():
        folder = root / name
        folder.mkdir()
        value = 4. if name == "full" else 3.5 if name.startswith("zero") else 3.
        records = [episode("sana", 1.), episode("ttn", value)] if name == "full" else [episode("ttn", value)]
        if spec["mode"] == "protected":
            records[-1].update(sink_reference_sha256="a"*64, sink_reference_verified=True)
        summary = dict(protocol=dict(protocol, tla_sink=spec), episodes=records)
        (folder / "summary.json").write_text(json.dumps(summary))
        (root / f"{name}-status.json").write_text('{"status":"completed"}')
    return root, source


@pytest.mark.parametrize("corrupt", [None, "prefix", "source-reproduction"])
def test_dataset_collector_validates_source_reproduction_prefix_and_nullable_metrics(tmp_path, corrupt):
    from tools.ttn_submit_sink import collect
    root, source = dataset_suite(tmp_path)
    if corrupt:
        path = root / "aligned-010/summary.json" if corrupt == "prefix" else source / "long/summary.json"
        changed = json.loads(path.read_text())
        row = changed["episodes"][0 if corrupt == "prefix" else 1]
        row["metrics"]["per_frame_latent_mse"][1 if corrupt == "prefix" else 30] += .1
        if corrupt == "prefix": row["prefix_13_metrics"]["per_frame_latent_mse"][1] += .1
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="four chunks|reproduce"): collect(root)
        return
    result = collect(root)
    assert result["status"] == "completed" and result["full_c_reproduction"]["cases"][0]["within_tolerance"]
    assert result["contrasts"]["aligned-010_minus_zero-010"]["long"]["mean_future_latent_mse"] < 0
    assert result["contrasts"]["aligned-010_minus_full"]["long"]["revisit_return_gt_latent_mse"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("failed", [False, True])
def test_sink_batch_is_single_gpu_and_marks_nonzero_worker_exit(tmp_path, failed):
    import os
    import shutil
    import subprocess
    import sys
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash unavailable")
        bash = Path(candidate)
    repo = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    for name in ("ttn_cache_env.sh", "ttn_slurm_sink.sbatch"):
        shutil.copyfile(repo / "tools" / name, project / "tools" / name)
    root = tmp_path / "root"
    prefix_python = root / "envs/worldttn/bin/python"
    prefix_python.parent.mkdir(parents=True)
    writer = tmp_path / "writer.py"
    writer.write_text("import json,os,sys\nfrom pathlib import Path\n"
                     "with Path(os.environ['CAPTURE']).open('a') as stream: stream.write(json.dumps(sys.argv[1:])+'\\n')\n"
                     "sys.exit(7 if sys.argv[1]=='srun' and os.environ['FAILED']=='1' else 0)\n")
    wrapper = '#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$WRITER" "$@"\n'
    prefix_python.write_text(wrapper, newline="\n")
    prefix_python.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "srun").write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$WRITER" srun "$@"\n', newline="\n")
    (bins / "srun").chmod(0o755)
    output = tmp_path / "suite"
    output.mkdir()
    (output / "plan.json").write_text('{}')
    capture = tmp_path / "capture.jsonl"
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SLURM_", "SBATCH_"))}
    env.update(ROOT=root.as_posix(), PROJECT_ROOT=project.as_posix(), SINK_OUTPUT=output.as_posix(),
               SLURM_JOB_ID="12345", SLURM_ARRAY_TASK_ID="3", SLURM_NTASKS="1", SLURM_JOB_NUM_NODES="1",
               REAL_PYTHON=Path(sys.executable).as_posix(), WRITER=writer.as_posix(), CAPTURE=capture.as_posix(),
               FAILED=str(int(failed)), MOCK_BIN=bins.as_posix(), SCRIPT=(project / "tools/ttn_slurm_sink.sbatch").as_posix())
    command = 'export MSYS_NO_PATHCONV=1 MSYS2_ENV_CONV_EXCL="*"; export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash "$SCRIPT"'
    result = subprocess.run([str(bash), "-c", command], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == (7 if failed else 0), result.stderr
    calls = [json.loads(row) for row in capture.read_text().splitlines()]
    assert calls[0][0] == "srun" and "--nodes=1" in calls[0] and "--ntasks=1" in calls[0] and "--gres=gpu:1" in calls[0]
    assert calls[0][-2:] == ["--run-variant", "3"]
    if failed:
        assert "--mark-failed" in calls[1] and calls[1][-2:] == ["--failure-exit-code", "7"]
    else: assert len(calls) == 1
