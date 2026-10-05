"""Offline exports must retain history pairing, head detail and failure evidence."""
import csv
import json
import tarfile
from pathlib import Path

import pytest
import torch

from tools.ttn_export_analysis import export_analysis


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_run(tmp_path):
    run = tmp_path / "experiment"
    joint = run / "joint"
    lines(joint / "train.jsonl", [
        dict(step=step, loss=.4, stage="C", train_scope="dit", outer_grad_norm=.1,
             seconds=140, ranks=[dict(rank=0, peak_allocated_bytes=2**30, chunks=[
                 dict(chunk=0, start=0, end=4, anchors=[dict(block=19, prefill=False,
                     per_head=dict(innovation_relative=[[.9, None]], inner_grad_norm=[[1., 2.]]))])])])
        for step in (99, 100, 160, 170, 171)])
    # No stability.jsonl: exporter must recover it from the gathered train record.
    write(joint / "run_config.json", {"history_protocol": {"camera_cache": "full clip"}})
    lines(joint / "jobs.jsonl", [dict(job="42", start_step=160, stop_step=170)])
    lines(joint / "eval_jobs.jsonl", [dict(job="43", step=125)])
    write(joint / "evaluations/step-000125/summary.json", {
        "status": "completed", "identity": {"step": 125, "checkpoint_sha256": "checkpoint", "fixed_cases_sha256": "case"},
        "results": {"long": {"metrics": {"mean_future_latent_mse": {"sana": .5, "ttn": 1., "paired_count": 1}}}}})
    mechanism = joint / "evaluations/mechanism"
    metrics = lambda baseline: {"mean_future_latent_mse": {"sana": baseline, "ttn": 1., "paired_count": 1},
                               "revisit_return_gt_latent_mse": {"sana": None, "ttn": None, "paired_count": 0}}
    write(mechanism / "summary.json", {
        "status": "completed", "identity": {"step": 100}, "results": {
            "full": {"history_source": "generated", "long": {"metrics": metrics(.5)}},
            "gt-history": {"history_source": "gt", "long": {"metrics": metrics(.2)}}},
        "contrasts": {"online_ttt_minus_no_ttt": {"long": {"mean_future_latent_mse": -.1}}}})
    write(mechanism / "full/summary.json", {"protocol": {"step": 100}, "episodes": [
        {"method": "ttn", "case_id": "fixed", "seed": 3407, "metrics": {"per_frame_latent_mse": [None, 1.]},
         "chunks": [{"chunk": 0, "start": 0, "end": 4, "anchors": [{"block": 19,
             "state_dynamics": {"transport_relative": [[.01, None]]}}]}]}]})
    (mechanism / "full/case-000-ttn.pt").write_bytes(b"must not archive weights or latents")
    write(mechanism / "job.json", {"job": "44"})
    (joint / "slurm-42.out").write_text("[TTN progress] step 170/170\n")
    folder = joint / "last-resume-160"
    folder.mkdir()
    torch.save({"step": 160, "stage": "C", "train_scope": "dit", "distributed": {
        "world_size": 1, "mode": "fsdp2", "resume_dir": folder.name, "checkpoint_id": "id"}}, joint / "last.pt")
    torch.save({"step": 160, "rank": 0, "world_size": 1, "mode": "fsdp2", "checkpoint_id": "id"}, folder / "rank-00000.pt")
    return run, mechanism


def test_exports_recorded_metrics_without_cross_history_baseline_or_checkpoint_claim(tmp_path):
    run, mechanism = fixture_run(tmp_path)
    original = (run / "joint/train.jsonl").read_bytes()
    out = tmp_path / "export"
    audit = export_analysis(run, mechanism, out, 100, 170, slurm=False)
    assert audit["training"]["observed_last_step"] == 171
    assert audit["training"]["selected_steps"] == [100, 160, 170]
    assert audit["checkpoint"]["step"] == 160 and audit["checkpoint"]["shards_match"] is True
    assert audit["training"]["steps_after_published_checkpoint"] == [170]
    comparisons = list(csv.DictReader((out / "method_comparison.csv").open(encoding="utf-8-sig")))
    gt = next(r for r in comparisons if r["variant"] == "gt-history" and r["metric"] == "mean_future_latent_mse")
    assert gt["sana"] == "0.2" and gt["history"] == "gt"
    missing = next(r for r in comparisons if r["metric"] == "revisit_return_gt_latent_mse")
    assert missing["ttn"] == "" and missing["paired_count"] == "0"
    stats = list(csv.DictReader((out / "training_stability.csv").open(encoding="utf-8-sig")))
    assert any(r["metric"] == "per_head.inner_grad_norm.0.1" and r["value"] == "2.0" for r in stats)
    assert any(r["metric"] == "per_head.innovation_relative.0.1" and r["value"] == "" for r in stats)
    rollout = list(csv.DictReader((out / "rollout_stability.csv").open(encoding="utf-8-sig")))
    assert any(r["metric"] == "state_dynamics.transport_relative.0.0" for r in rollout)
    assert audit["selected_jobs"] == ["42", "43", "44"]
    report = (out / "summary.txt").read_text()
    assert "saved checkpoint=160" in report and "gt-history/sana: NA | 0.200000" in report
    with tarfile.open(out.with_suffix(".tar.gz")) as archive:
        names = archive.getnames()
        assert any(n.endswith("slurm-42.out") for n in names)
        assert not any(n.endswith(".pt") for n in names)
        assert any(n.endswith("method_comparison.csv") for n in names)
    assert (run / "joint/train.jsonl").read_bytes() == original


def test_export_requires_new_external_directory_and_reports_partial_json(tmp_path):
    run, mechanism = fixture_run(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        export_analysis(run, mechanism, run / "export", 100, 170, slurm=False)
    with (run / "joint/train.jsonl").open("a") as stream:
        stream.write('{"step":')
    out = tmp_path / "export"
    audit = export_analysis(run, mechanism, out, 100, 170, slurm=False)
    assert any("incomplete" in message for message in audit["warnings"])
    with pytest.raises(FileExistsError):
        export_analysis(run, mechanism, out, 100, 170, slurm=False)


def test_export_retains_both_teachers_and_collects_external_stderr(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools import ttn_export_analysis as module
    run, mechanism = fixture_run(tmp_path)
    episode = {"case_id": "fixed", "seed": 3407, "probes": [{"timestep": 500,
        "anchors": [{"block": 3, "stages": {"visual_raw": {"relative_l2": .9}}}],
        "representation_drift": [{"block": 3, "point": "after", "relative_l2": .1}]}],
        "gradient_health": {"anchors": [{"block": 3, "groups": {"qkv": {"grad_norm": .02}}}]},
        "parameter_changes": [{"block": 3, "groups": {"beta_proj": {"relative_update_ratio": None}}}]}
    episode["matched_backbone_softmax"] = {"probes": episode["probes"]}
    write(run / "joint/evaluations/step-000125/align/summary.json", {"protocol": {"step": 125}, "episodes": [episode]})
    error = tmp_path / "external-42.err"
    error.write_text("Disk quota exceeded during checkpoint save\n")
    def accounting(command, **kwargs):
        if "--noheader" in command:
            stdout = f"42|ttn-formal||{tmp_path}/external-%j.err|{tmp_path}|\n"
        else:
            stdout = "JobID|State|ExitCode\n42|FAILED|1:0\n"
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
    monkeypatch.setattr(module.subprocess, "run", accounting)
    out = tmp_path / "export"
    module.export_analysis(run, mechanism, out, 100, 170)
    teachers = list(csv.DictReader((out / "teacher_alignment.csv").open(encoding="utf-8-sig")))
    assert {r["teacher"] for r in teachers} == {"original_sana", "matched_backbone_softmax"}
    drift = list(csv.DictReader((out / "representation_drift.csv").open(encoding="utf-8-sig")))
    assert {r["point"] for r in drift} == {"after"}
    assert any("Disk quota exceeded" in p.read_text() for p in (out / "raw/slurm").glob("*.err"))
