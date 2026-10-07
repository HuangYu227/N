"""Pinned, single-GPU sink interventions with a verified baseline dependency."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

import torch

from tools.ttn_custom_inference import load_case
from tools.ttn_submit_mechanism import prepare as prepare_mechanisms
from worldttn.core import ANCHORS, TTNConfig
from worldttn.evaluation import file_sha256, paired_summary
from worldttn.mechanism_evaluation import atomic_json, IDENTITY, sink_identity
from worldttn.sink import SinkOptions

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CASE = REPO / "assets/worldttn/study_static/case.json"
SIGNATURES = ("input_sha256", "initial_noise_sha256", "base_sha256")


def check_smoke(root):
    """Step100 / five-chunk acceptance: prove noisy output influence, not quality."""
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    protocol = summary["protocol"]
    expected = dict(status="completed", step=100, stage="C", frames=16, raw_frames=121, steps=4,
                    camera_attention="sana", history_source="generated", ttn_ablation="full",
                    execution="reference/reference", state_diagnostics=True,
                    tla_sink=asdict(SinkOptions("protected")))
    mismatches = {key: {"actual": protocol.get(key, "<missing>"), "expected": value}
                  for key, value in expected.items() if protocol.get(key) != value}
    if mismatches:
        raise ValueError("smoke protocol mismatch: " + json.dumps(mismatches, ensure_ascii=False))
    cfg = protocol.get("cfg_scale")
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(cfg) or cfg < 1:
        raise ValueError("invalid smoke CFG")
    rows = [row for row in summary["episodes"] if row["method"] == "ttn"]
    if len(rows) != 1: raise ValueError("smoke requires exactly one TTN episode")
    row = rows[0]
    digest = row.get("sink_reference_sha256", "")
    if (row.get("commits") != 6 or row.get("sink_reference_verified") is not True
            or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("smoke commit count or protected reference SHA audit failed")
    ranges = ((0, 4), (4, 7), (7, 10), (10, 13), (13, 16))
    if len(row["chunks"]) != len(ranges): raise ValueError("smoke requires five predicted chunks")
    shifts, checked, noisy_norms = [12]*(2 if cfg > 1 else 1), [], {}

    def active_read(sink):
        if (sink.get("active") is not True or sink.get("mode") != "protected" or sink.get("gain") != .1
                or sink.get("position") != "temporal-realign" or sink.get("temporal_shift") != shifts
                or sink.get("reference_source") != "observed-prefill" or sink.get("reference_sha256") != digest):
            raise ValueError("fifth chunk sink activation/shift/settings/reference SHA differs")
        value = sink.get("effective_delta_norm")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("sink effective_delta_norm must be finite and nonnegative")
        return value

    for index, (chunk, bounds) in enumerate(zip(row["chunks"], ranges)):
        anchors = chunk["anchors"]
        if (chunk.get("chunk") != index or (chunk.get("start"), chunk.get("end")) != bounds
                or len(anchors) != len(ANCHORS) or {anchor["block"] for anchor in anchors} != set(ANCHORS)):
            raise ValueError("smoke chunk bounds or five anchor identities differ")
        for anchor in anchors:
            if index < 4:
                if anchor["sink"].get("active") is not False or anchor.get("sink_trajectory"):
                    raise ValueError("sink activated before the fifth predicted chunk")
            else:
                active_read(anchor["sink"])
                calls = anchor.get("sink_trajectory", [])
                if [call.get("call") for call in calls] != [0, 2, 3]:
                    raise ValueError("fifth chunk requires first/middle/last noisy sink measurements")
                noisy_norms[str(anchor["block"])] = max(active_read(call) for call in calls)
        checked.append(dict(predicted_chunk=index+1, latent_range=list(bounds), active=index == 4,
                            anchors=list(ANCHORS), temporal_shift=shifts if index == 4 else None))
    nonzero = [int(block) for block, value in noisy_norms.items() if value > 0]
    if not nonzero: raise ValueError("sink has no nonzero gate/projection contribution in noisy generation calls")
    video = json.loads((root / "videos/comparison.json").read_text(encoding="utf-8"))
    encoded = video.get("encoded_video_validation", {})
    if video.get("status") != "completed" or any(encoded.get(name, {}).get("frames") != 121
            for name in ("sana.mp4", "ttn.mp4", "comparison.mp4")):
        raise ValueError("smoke video encoding is incomplete or has incorrect frame counts")
    return dict(status="passed", step=100, checkpoint_sha256=protocol["checkpoint_sha256"], commits=6,
                sink=expected["tla_sink"], reference_sha256=digest, reference_verified=True, chunks=checked,
                noisy_effective_delta_norm=noisy_norms, nonzero_noisy_anchors=nonzero,
                measurement="FP32 gate/projection contribution before output quantization; no quality claim")


def interventions(profile="static"):
    if profile not in ("static", "orbit"): raise ValueError("unknown sink profile")
    specs = {"full": SinkOptions(), "zero-010": SinkOptions("zero"),
             "absolute-010": SinkOptions("protected", position="absolute"),
             "aligned-010": SinkOptions("protected"), "zero-025": SinkOptions("zero", .25),
             "aligned-025": SinkOptions("protected", .25)}
    if profile == "orbit": specs = {name: specs[name] for name in ("full", "aligned-010")}
    return {name: asdict(value) for name, value in specs.items()}


def prepare(*, training_run=None, evaluation=None, case=None, output=None, fixed_cases=None, profile="static"):
    if bool(training_run) == bool(evaluation):
        raise ValueError("provide exactly one immutable --training-run snapshot or completed --evaluation")
    specs = interventions(profile)
    if evaluation:
        if case is not None or profile != "static": raise ValueError("dataset suite uses its pinned cases")
        plan = dict(prepare_mechanisms(evaluation, output, fixed_cases))
        if output is not None: plan["output"] = str(Path(output).resolve())
        plan.update(kind="dataset", source_evaluation=str(Path(evaluation).resolve()), frames=61)
        source = Path(evaluation) / "long/summary.json"
        original = json.loads(source.read_text(encoding="utf-8"))["protocol"] if source.is_file() else {}
        plan["source_protocol"] = original
        plan["noise_frames"] = original.get("noise_frames", 61)
    else:
        snapshot = Path(training_run).resolve()
        marker = snapshot / "snapshot.json"
        if not marker.is_file(): raise ValueError("training-run must be an immutable evaluation snapshot")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("format") not in ("TTN-evaluation-snapshot-v1", "TTN-evaluation-snapshot-v2"):
            raise ValueError("unsupported immutable evaluation snapshot format")
        payload = torch.load(snapshot / "last.pt", map_location="cpu", weights_only=False, mmap=True)
        config = TTNConfig(**payload["config"])
        run = json.loads((snapshot / "run_config.json").read_text(encoding="utf-8"))
        last = json.loads((snapshot / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        if (payload["step"] != metadata["step"] or last["step"] != metadata["step"]
                or payload["stage"] != config.stage or last["stage"] != config.stage
                or run["base"]["sha256"] != payload["base_sha256"]):
            raise ValueError("snapshot checkpoint and completed training identity disagree")
        if config.stage != "C" or config.camera_attention != "sana":
            raise ValueError("sink suite requires Stage C with native SANA camera")
        case_path = Path(case or DEFAULT_CASE).resolve()
        case_data, _, prompt = load_case(case_path)
        if case_data["latent_frames"] != 61: raise ValueError("sink suite requires 61 latent frames")
        if profile == "orbit" and case_data["camera"]["trajectory"] != "closed_orbit":
            raise ValueError("orbit profile requires a closed_orbit case")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(output).resolve() if output else snapshot.parent / f"sink-{profile}-step-{metadata['step']:06d}-{stamp}"
        plan = dict(kind="custom", snapshot=str(snapshot), step=metadata["step"], output=str(output),
                    checkpoint_sha256=file_sha256(snapshot / "last.pt"), case=str(case_path),
                    case_sha256=file_sha256(case_path), case_data=case_data, prompt=prompt,
                    frames=61, steps=20, cfg_scale=4.5, cached_blocks=2, seed=case_data["seed"],
                    cross_attn_backend="math", compile={})
    if Path(plan["output"]).exists(): raise ValueError("use a new sink output directory")
    plan.update(format="TTN-sink-suite-v1", profile=profile, variants=list(specs), interventions=specs,
                array=f"1-{len(specs)-1}%1")
    return plan


def _plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text(encoding="utf-8"))
    expected = interventions(plan.get("profile", "static"))
    if (plan.get("format") != "TTN-sink-suite-v1" or plan["variants"] != list(expected)
            or plan["interventions"] != expected or plan["kind"] not in ("custom", "dataset")):
        raise ValueError("unexpected sink suite plan or interventions")
    return plan


def _rows(summary, method):
    selected = [r for r in summary["episodes"] if r["method"] == method]
    indexed = {(r["case_id"], r["seed"]): r for r in selected}
    if not selected or len(indexed) != len(selected): raise ValueError("missing/duplicate sink case identity")
    return indexed


def _paired_identity(reference, current):
    if current.keys() != reference.keys(): raise ValueError("sink case/seed identity differs")
    if any(current[key][field] != reference[key][field] for key in reference for field in SIGNATURES):
        raise ValueError("sink conditioning/noise/base identity differs")


def _frame_comparison(reference, current, *, prefix=False):
    _paired_identity(reference, current)
    checks = []
    for key, row in current.items():
        a = reference[key]["metrics"]["per_frame_latent_mse"]
        b = row["metrics"]["per_frame_latent_mse"]
        if prefix: a, b = a[:13], b[:13]
        if len(a) != len(b) or a[0] is not None or b[0] is not None:
            raise ValueError("sink metric horizons differ")
        valid = all(abs(x-y) <= 1e-5 + 1e-4 * abs(x) for x, y in zip(a[1:], b[1:]))
        checks.append(dict(case_id=key[0], seed=key[1], within_tolerance=valid,
                           max_abs_per_frame_mse_delta=max(abs(x-y) for x, y in zip(a[1:], b[1:]))))
    return checks


def collect(root, *, validating=None):
    root = Path(root)
    plan = _plan(root)
    submission_path = root / "submission-status.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8")) if submission_path.is_file() else None
    failed_submission = submission is not None and submission.get("status") == "failed"
    statuses, summaries, identity = {}, {}, None
    for name in plan["variants"]:
        status = root / f"{name}-status.json"
        statuses[name] = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {"status": "pending"}
        path = root / name / "summary.json"
        # Only this worker may inspect its finished output before marking completion.
        if statuses[name]["status"] != "completed" and name != validating: continue
        if not path.is_file():
            if statuses[name]["status"] == "completed": raise ValueError(f"{name}: completed result is missing")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        protocol = summary["protocol"]
        if protocol.get("tla_sink") != plan["interventions"][name]:
            raise ValueError(f"{name}: sink parameters differ from plan")
        for field in ("checkpoint_sha256", "step", "frames", "steps", "cfg_scale", "cached_blocks", "cross_attn_backend"):
            if protocol.get(field) != plan[field]: raise ValueError(f"{name}: sink protocol differs: {field}")
        if protocol.get("stage") != "C" or protocol.get("history_source") != "generated" or protocol.get("ttn_ablation") != "full":
            raise ValueError("sink runs require Full Stage C with generated history")
        if plan["kind"] == "custom":
            keys = ("stage", "checkpoint_sha256", "step", "frames", "steps", "seed", "cfg_scale", "cached_blocks",
                    "cross_attn_backend", "flow_shift", "camera_attention", "execution", "case", "prompt", "config",
                    "input_bundle_sha256", "state_diagnostics", "ttn_config", "provenance", "compile")
            if (protocol.get("status") != "completed" or protocol.get("camera_attention") != "sana"
                    or protocol.get("execution") != "reference/reference" or protocol.get("case") != plan["case_data"]
                    or protocol.get("prompt") != plan["prompt"] or protocol.get("seed") != plan["seed"]
                    or summary.get("metrics") is not None):
                raise ValueError("custom sink protocol differs or contains fabricated future GT metrics")
        else:
            keys = (*IDENTITY, "meta_ttt", "config", "state_diagnostics", "compile")
            if not protocol.get("state_diagnostics") or protocol.get("ttn_camera_attention") != "sana":
                raise ValueError("dataset sink runs require state diagnostics and native SANA camera")
            if protocol.get("fixed_cases_sha256") != plan["fixed_cases_sha256"]:
                raise ValueError("dataset sink fixed cases differ")
        current = {key: protocol.get(key) for key in keys}
        if identity is not None and current != identity: raise ValueError("sink sampling/source protocol identity differs")
        identity = current
        summaries[name] = summary
        if statuses[name]["status"] == "completed": statuses[name]["path"] = str(path)
    result = dict(status="failed" if failed_submission or any(s["status"] == "failed" for s in statuses.values()) else "running",
                  submission=submission,
                  kind=plan["kind"], identity=identity, variants=statuses, results={}, contrasts={},
                  metric_space="none: no future GT" if plan["kind"] == "custom" else "cached LTX latents",
                  interpretation="same-weight inference interventions; zero controls isolate output shrinkage; single-case results are diagnostic")
    if "full" not in summaries: return result
    baseline = _rows(summaries["full"], "ttn")
    sana = _rows(summaries["full"], "sana")
    _paired_identity(baseline, sana)
    if plan["kind"] == "dataset":
        original = json.loads((Path(plan["source_evaluation"]) / "long/summary.json").read_text(encoding="utf-8"))
        for field in ("checkpoint_sha256", "fixed_cases_sha256", "stage", "step", "frames", "noise_frames",
                      "steps", "cfg_scale", "cached_blocks", "flow_shift", "ttn_camera_attention", "cross_attn_backend"):
            if original["protocol"].get(field) != summaries["full"]["protocol"].get(field):
                raise ValueError(f"sink full differs from source milestone protocol: {field}")
        if sink_identity(original["protocol"]) != {"mode": "off"}: raise ValueError("source milestone sink must be off")
        reproduction = _frame_comparison(_rows(original, "ttn"), baseline)
        if not all(row["within_tolerance"] for row in reproduction): raise ValueError("sink full does not reproduce source milestone")
        result["full_c_reproduction"] = {"cases": reproduction, "metric_tolerance": "abs 1e-5 + rel 1e-4"}
    for name, summary in summaries.items():
        current = _rows(summary, "ttn")
        _paired_identity(baseline, current)
        reference_hashes = []
        if plan["interventions"][name]["mode"] == "protected":
            for row in current.values():
                digest = row.get("sink_reference_sha256", "")
                if (row.get("sink_reference_verified") is not True or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError(f"{name}: protected reference audit missing or failed")
                reference_hashes.append(dict(case_id=row["case_id"], seed=row["seed"], sha256=digest, verified=True))
        if any(r["method"] not in ("sana", "ttn") for r in summary["episodes"]):
            raise ValueError("unexpected sink method")
        if any(r["method"] == "sana" for r in summary["episodes"]):
            _paired_identity(sana, _rows(summary, "sana"))
        item = dict(path=str(root / name / "summary.json"), gt_metrics=None,
                    reference_audit=reference_hashes,
                    timing=[r["timing"] for r in current.values()],
                    return_view=[r.get("return_view") for r in current.values()],
                    prefix_13_validation=statuses[name].get("prefix_13_validation"))
        if plan["kind"] == "dataset":
            for row in current.values():
                if row["prefix_13_metrics"]["per_frame_latent_mse"] != row["metrics"]["per_frame_latent_mse"][:13]:
                    raise ValueError("13-frame prefix metrics differ from the causal rollout prefix")
            checks = _frame_comparison(baseline, current, prefix=True)
            if not all(row["within_tolerance"] for row in checks): raise ValueError("sink changed the first four chunks")
            item["prefix_13_matches_baseline"] = checks
            item["gt_metrics"] = {h: paired_summary([dict(r, metrics=r[field]) for table in (sana, current) for r in table.values()])
                                  for h, field in (("long", "metrics"), ("short", "prefix_13_metrics"))}
        result["results"][name] = item
    if plan["kind"] == "dataset":
        pairs = [(name, "full") for name in summaries if name != "full"]
        pairs += [(name, zero) for name, zero in (("absolute-010", "zero-010"), ("aligned-010", "zero-010"),
                                                ("aligned-025", "zero-025")) if name in summaries and zero in summaries]
        for a, b in pairs:
            result["contrasts"][f"{a}_minus_{b}"] = {h: {metric: values["ttn"] - result["results"][b]["gt_metrics"][h]["metrics"][metric]["ttn"]
                if values["ttn"] is not None and result["results"][b]["gt_metrics"][h]["metrics"][metric]["ttn"] is not None else None
                for metric, values in result["results"][a]["gt_metrics"][h]["metrics"].items()} for h in ("short", "long")}
    if not failed_submission and len(summaries) == len(plan["variants"]) and all(s["status"] == "completed" for s in statuses.values()):
        result["status"] = "completed"
    return result


def _publish(root, *, validating=None):
    from worldttn.checkpoint_integrity import acquire_checkpoint_lock, release_checkpoint_lock
    root = Path(root)
    lock, token = root / ".sink-summary.lock", uuid.uuid4().hex
    acquire_checkpoint_lock(lock, token)
    try:
        result = collect(root) if validating is None else collect(root, validating=validating)
        atomic_json(root / "summary.json", result)
        return result
    finally:
        release_checkpoint_lock(lock, token)


def mark_failed(root, index, exit_code):
    """Batch EXIT fallback covers TERM/OOM that cannot run Python's handler."""
    root = Path(root)
    plan = _plan(root)
    if not 0 <= index < len(plan["variants"]): raise ValueError("sink task index is outside the plan")
    name = plan["variants"][index]
    path = root / f"{name}-status.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else dict(variant=name, output=str(root/name))
    record.update(status="failed", exit_code=exit_code)
    record.setdefault("error", f"Slurm worker exited with status {exit_code}; inspect its registered log")
    record.setdefault("first_exception", record["error"])
    atomic_json(path, record)
    try: _publish(root)
    except Exception: atomic_json(root / "summary.json", dict(status="failed", variant=name, error=record["error"]))


def run_variant(root, index):
    root = Path(root)
    plan = _plan(root)
    if not 0 <= index < len(plan["variants"]): raise ValueError("sink task index is outside the plan")
    name = plan["variants"][index]
    status_path = root / f"{name}-status.json"
    prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if prior.get("status") in ("completed", "failed"):
        raise ValueError("sink task already finished; use a new suite output rather than overwrite its results")
    record = dict(status="running", variant=name, output=str(root / name), job=os.getenv("SLURM_JOB_ID"), phase="validate")
    atomic_json(status_path, record)
    try:
        if file_sha256(Path(plan["snapshot"]) / "last.pt") != plan["checkpoint_sha256"]:
            raise ValueError("pinned sink checkpoint changed")
        partial = collect(root)
        if partial["status"] == "failed": raise ValueError("a prior sink task failed; subsequent inference is stopped")
        if index and partial["variants"]["full"]["status"] != "completed":
            raise ValueError("sink baseline must complete and validate before interventions")
        spec = plan["interventions"][name]
        args = ["--training-run", plan["snapshot"], "--output", str(root / name), "--steps", str(plan["steps"]),
                "--cfg-scale", str(plan["cfg_scale"]), "--cached-blocks", str(plan["cached_blocks"]),
                "--cross-attn-backend", plan["cross_attn_backend"], "--state-diagnostics",
                "--tla-sink", spec["mode"], "--sink-gain", str(spec["gain"]),
                "--sink-position", spec["position"], "--sink-start-chunk", str(spec["start_chunk"])]
        record["phase"] = "inference"
        atomic_json(status_path, record)
        print(f"[TTN sink task] variant={name} phase=inference output={root/name}", flush=True)
        if plan["kind"] == "custom":
            if file_sha256(plan["case"]) != plan["case_sha256"]: raise ValueError("sink custom case changed")
            if index: args += ["--reuse-baseline", str(root / "full")]
            from tools.ttn_custom_inference import main as infer
            infer([*args, "--case", plan["case"]])
            for folder in (("videos", "videos-vs-ttn-baseline") if index else ("videos",)):
                video = json.loads((root / name / folder / "comparison.json").read_text(encoding="utf-8"))
                if video.get("status") != "completed" or not video.get("encoded_video_validation"):
                    raise ValueError(f"{name}: encoded video validation is missing or incomplete")
            if index:
                from tools.ttn_decode_comparison import load_pair
                _, _, (baseline, current) = load_pair(root / name, 0, 4, left_evaluation=root / "full", left_method="ttn")
                torch.testing.assert_close(current, baseline, atol=1e-5, rtol=1e-4,
                    msg=lambda message: f"sink changed the first four chunks: {message}")
                record["prefix_13_validation"] = dict(status="passed", latent_frames=13, atol=1e-5, rtol=1e-4,
                    max_abs_difference=float((current - baseline).abs().max()),
                    reference="saved TTN baseline prefix; no future GT")
        else:
            from worldttn.cli import main as infer
            infer(["evaluate", *args, "--parallel", "single", "--fixed-cases", plan["fixed_cases"],
                   "--frames", "61", "--noise-frames", str(plan["noise_frames"]),
                   "--eval-cases", str(plan["eval_cases"]), "--seed", str(plan["seed"]),
                   "--history-source", "generated", "--ttn-ablation", "full",
                   "--ttn-core-backend", "reference", "--ttn-psi-backend", "reference",
                   "--eval-methods", *(("sana", "ttn") if index == 0 else ("ttn",))])
        record["phase"] = "validate_results"
        atomic_json(status_path, record)
        _publish(root, validating=name)
        record.update(status="completed", phase="completed")
        atomic_json(status_path, record)
        return _publish(root)
    except BaseException as error:
        record.update(status="failed", error=str(error), first_exception=traceback.format_exc())
        atomic_json(status_path, record)
        try: _publish(root)
        except Exception: atomic_json(root / "summary.json", dict(status="failed", error=str(error), variant=name))
        array = os.getenv("SLURM_ARRAY_JOB_ID", "")
        if array.isdecimal():
            try: subprocess.run(["scancel", "--state=PENDING", array], capture_output=True, check=False)
            except OSError: pass  # Batch EXIT trap retries; preserve the original inference exception.
        raise


def submit(plan, *, partition="day", root=None):
    root = Path(root or os.environ.get("ROOT", "/data/group/zhaolab/home/z2zhang/huangyu"))
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    # The existing snapshot releaser recognizes this pin; no model is copied.
    (Path(plan["snapshot"]) / ".keep").touch(exist_ok=True)
    _publish(output)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("SLURM_", "SBATCH_"))
           and key not in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                           "CUDA_VISIBLE_DEVICES", "TTN_ENTRY_MODULE", "COMMAND", "ADAPTER", "CONFIG", "SANA_CONFIG", "BASE_WEIGHTS",
                           "TTN_CORE_BACKEND", "TTN_PSI_BACKEND", "CAMERA_ATTENTION", "CAMERA_ABLATION", "NOISE_FRAMES",
                           "TLA_SINK", "SINK_GAIN", "SINK_POSITION", "SINK_START_CHUNK", "HISTORY_SOURCE", "TTN_ABLATION")}
    env.update(ROOT=str(root), PROJECT_ROOT=str(REPO), PYTHON=str(root / "envs/worldttn/bin/python"), SINK_OUTPUT=str(output))
    for key in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE"):
        if plan.get("compile", {}).get(key) is not None: env[key] = plan["compile"][key]
    command = ["sbatch", "--parsable", "--export=ALL", f"--partition={partition}", f"--chdir={REPO}",
               f"--output={output}/slurm-%A_%a.out"]
    jobs = {}
    try:
        response = subprocess.run([*command, str(REPO / "tools/ttn_slurm_sink.sbatch")], env=env, capture_output=True, text=True, check=True)
        jobs["baseline"] = response.stdout.strip().split(";")[0]
        if not jobs["baseline"].isdecimal(): raise RuntimeError("unexpected sink baseline sbatch response")
        atomic_json(output / "jobs.json", jobs)
        response = subprocess.run([*command, f"--dependency=afterok:{jobs['baseline']}", "--kill-on-invalid-dep=yes",
                                   f"--array={plan['array']}", str(REPO / "tools/ttn_slurm_sink.sbatch")],
                                  env=env, capture_output=True, text=True, check=True)
        jobs["interventions"] = response.stdout.strip().split(";")[0]
        if not jobs["interventions"].isdecimal(): raise RuntimeError("unexpected sink array sbatch response")
        atomic_json(output / "jobs.json", jobs)
        return jobs
    except BaseException as error:
        atomic_json(output / "submission-status.json", dict(status="failed", jobs=jobs, error=str(error),
            first_exception=traceback.format_exc(), stderr=getattr(error, "stderr", None), stdout=getattr(error, "stdout", None)))
        _publish(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run")
    parser.add_argument("--evaluation")
    parser.add_argument("--case")
    parser.add_argument("--fixed-cases")
    parser.add_argument("--output")
    parser.add_argument("--profile", choices=("static", "orbit"), default="static")
    parser.add_argument("--partition", default="day")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect", type=Path)
    parser.add_argument("--check-smoke", type=Path, help="validate completed step100 five-chunk GPU smoke")
    parser.add_argument("--run-variant", type=int)
    parser.add_argument("--mark-failed", type=int)
    parser.add_argument("--failure-exit-code", type=int, default=1)
    args = parser.parse_args(argv)
    if args.check_smoke:
        report = args.check_smoke / "smoke-validation.json"
        try: result = check_smoke(args.check_smoke)
        except Exception as error:
            atomic_json(report, dict(status="failed", error=str(error), first_exception=traceback.format_exc()))
            raise
        atomic_json(report, result)
        print(json.dumps(result, indent=2, allow_nan=False)); return
    if args.collect:
        print(json.dumps(_publish(args.collect), indent=2, allow_nan=False)); return
    if args.mark_failed is not None:
        if not args.output: parser.error("--mark-failed requires --output")
        mark_failed(args.output, args.mark_failed, args.failure_exit_code); return
    if args.run_variant is not None:
        if not args.output: parser.error("--run-variant requires --output")
        run_variant(args.output, args.run_variant); return
    plan = prepare(training_run=args.training_run, evaluation=args.evaluation, case=args.case, output=args.output,
                   fixed_cases=args.fixed_cases, profile=args.profile)
    if args.dry_run:
        print(json.dumps(plan, indent=2, allow_nan=False)); return
    root = Path(os.environ.get("ROOT", "/data/group/zhaolab/home/z2zhang/huangyu"))
    if Path(sys.executable).resolve() != (root / "envs/worldttn/bin/python").resolve():
        raise ValueError("use the existing $ROOT/envs/worldttn/bin/python interpreter")
    jobs = submit(plan, partition=args.partition, root=root)
    print(f"SINK_BASELINE_JOB={jobs['baseline']}\nSINK_ARRAY_JOB={jobs['interventions']}\nSINK_OUTPUT={plan['output']}", flush=True)


if __name__ == "__main__": main()
