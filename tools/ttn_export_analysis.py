"""Read existing TTN experiment results and export CSVs plus original logs. No GPU work."""
import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def leaves(value, prefix=""):
    """Array indices remain in the path: per_head.<metric>.<branch>.<head>."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from leaves(child, f"{prefix}.{i}")
    else:
        yield prefix, value


def json_file(path, warnings):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        warnings.append(f"Cannot read {path}: {error}")
        return {}


def json_lines(path, warnings):
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError:
                warnings.append(f"Invalid/incomplete JSON line {path}:{number}; skipped")


def copy_logs(source, destination):
    """Only small text evidence; never copy weights, RNG/optimizer shards or video latents."""
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".out", ".err", ".log", ".csv", ".txt"}:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def checkpoint_audit(joint):
    result = {"path": str(joint / "last.pt"), "exists": (joint / "last.pt").is_file(),
              "verification": "CPU mmap deserialization and metadata only; tensor contents not checked"}
    if not result["exists"]:
        return result
    try:
        import torch  # Optional, delayed; does not initialize CUDA or import SANA.
        value = torch.load(joint / "last.pt", map_location="cpu", weights_only=False, mmap=True)
        result.update({key: value.get(key) for key in ("format", "step", "stage", "train_scope", "base_sha256")})
        meta = value.get("distributed", {})
        result["distributed"] = meta
        name = meta.get("resume_dir", "")
        if not name or Path(name).name != name:
            raise ValueError("invalid or missing resume directory")
        shards = []
        for rank in range(meta["world_size"]):
            path = joint / name / f"rank-{rank:05d}.pt"
            row = {"rank": rank, "path": str(path), "exists": path.is_file(), "matches": False}
            if path.is_file():
                shard = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                row["step"] = shard.get("step")
                row["matches"] = all(shard.get(key) == expected for key, expected in {
                    "step": value["step"], "rank": rank, "world_size": meta["world_size"],
                    "mode": meta["mode"], "checkpoint_id": meta["checkpoint_id"]}.items())
                del shard
            shards.append(row)
        result.update(shards=shards, shards_match=bool(shards) and all(s["matches"] for s in shards))
    except Exception as error:
        result["error"] = str(error)
    return result


def slurm_evidence(jobs, destination, warnings, registered=()):
    if not jobs:
        return
    destination.mkdir(parents=True, exist_ok=True)
    by_job = {str(row["job"]): row for row in registered if row.get("job")}
    def collect(row):
        job = str(row["job"])
        for pattern in {row.get("stdout"), row.get("stderr")}:
            if not pattern or pattern in {"Unknown", "None", "(null)"}: continue
            base, _, task = job.partition("_")
            name = pattern.replace("%j", job).replace("%A", base).replace("%x", row.get("job_name", ""))
            if task: name = name.replace("%a", task)
            path = Path(name)
            if not path.is_absolute(): path = Path(row.get("workdir") or ".") / path
            if path.is_file():
                digest = hashlib.sha256(str(path).encode()).hexdigest()[:10]
                try: shutil.copy2(path, destination / f"{job}-{digest}-{path.name}")
                except OSError as error: warnings.append(f"Cannot copy Slurm log {path}: {error}")
            else:
                warnings.append(f"Slurm log path missing or unresolved: {path}")
    # These accounting fields exist on LTU; log paths come from our registry or scontrol.
    command = ["sacct", "-j", ",".join(jobs), "--parsable2",
               "--format=JobID,JobName,State,ExitCode,Elapsed,ReqMem,MaxRSS"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        (destination / "accounting.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode: warnings.append(f"sacct unavailable/failed: {result.stderr.strip()}")
    except (OSError, subprocess.TimeoutExpired) as error:
        warnings.append(f"Slurm accounting unavailable: {error}")
    for job in jobs:
        row = by_job.get(job)
        if row and (row.get("stdout") or row.get("stderr")):
            collect(row)
            continue
        try:
            result = subprocess.run(["scontrol", "show", "job", job, "-o"], capture_output=True, text=True, timeout=30)
            (destination / f"scontrol-{job}.txt").write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                warnings.append(f"scontrol job {job} unavailable; registered/local logs retained: {result.stderr.strip()}")
                continue
            fields = dict(re.findall(r"(?:^| )([A-Za-z][A-Za-z0-9_]*)=(.*?)(?= [A-Za-z][A-Za-z0-9_]*=|$)", result.stdout.strip()))
            collect({"job": job, "job_name": fields.get("JobName", ""), "stdout": fields.get("StdOut"),
                     "stderr": fields.get("StdErr"), "workdir": fields.get("WorkDir")})
        except (OSError, subprocess.TimeoutExpired) as error:
            warnings.append(f"Slurm log lookup unavailable: {error}")


def export_analysis(run, mechanism, output, first=100, last=170, *, slurm=True):
    run, mechanism, output = (Path(p).resolve() for p in (run, mechanism, output))
    if first > last or first < 0:
        raise ValueError("invalid step range")
    if not run.is_dir() or not mechanism.is_dir():
        raise ValueError("training run and mechanism directory must exist")
    if output == run or run in output.parents or output == mechanism or mechanism in output.parents:
        raise ValueError("export directory must be outside the input experiments")
    if output.exists() or output.with_suffix(".tar.gz").exists():
        raise FileExistsError("use a new output directory/archive")
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "ttn_export_analysis.py")
    joint, warnings = run / "joint", []
    audit = {"format": "TTN-offline-analysis-v1", "run": str(run), "mechanism": str(mechanism),
             "requested_steps": [first, last], "warnings": warnings,
             "notes": ["No model calls/optimizer changes. Inputs are read-only; no GPU job is submitted.",
                 "CSV blanks are missing/undefined, never zero. Array indices retain branch/head detail.",
                 "GT history updates all caches and is a diagnostic protocol, not one-frame deployment.",
                 "Slope is descriptive; absent revisit pairs cannot establish revisit quality.",
                 "Historical FSDP gradient norms are after clipping LOCAL shard norms, not rank means.",
                 "Teacher gradient probes are unclipped fixed-case gradients, not historical training gradients.",
                 "Training transport spectra were not recorded unless diagnostics was enabled; no reconstruction.",
                 "LPIPS/decoded-video metrics are not computed by this exporter."]}
    tables = {
        "training": ["step", "stage", "train_scope", "loss", "outer_grad_norm", "seconds", "tbptt", "parallel", "world_size", "global_batch", "implementation_id"],
        "training_ranks": ["step", "rank", "metric", "value"],
        "training_stability": ["step", "rank", "chunk", "start", "end", "block", "prefill", "metric", "value"],
        "method_comparison": ["source", "status", "step", "horizon", "variant", "history", "metric", "sana", "ttn", "ttn_minus_sana", "paired_count", "checkpoint_sha256", "fixed_cases_sha256"],
        "mechanism_contrasts": ["contrast", "horizon", "metric", "delta"],
        "rollout_metrics": ["source", "step", "case_id", "seed", "method", "metric", "value"],
        "rollout_frames": ["source", "step", "case_id", "seed", "method", "frame", "latent_mse"],
        "rollout_chunks": ["source", "step", "case_id", "seed", "method", "chunk", "metric", "value"],
        "rollout_stability": ["source", "step", "case_id", "seed", "method", "chunk", "block", "metric", "value"],
        "teacher_alignment": ["source", "step", "case_id", "seed", "teacher", "timestep", "block", "metric", "value"],
        "representation_drift": ["source", "step", "case_id", "seed", "teacher", "timestep", "block", "point", "metric", "value"],
        "teacher_health": ["source", "step", "case_id", "seed", "teacher", "kind", "block", "metric", "value"],
    }
    with ExitStack() as stack:
        writers = {}
        for name, fields in tables.items():
            stream = stack.enter_context((output / f"{name}.csv").open("w", encoding="utf-8-sig", newline=""))
            writers[name] = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writers[name].writeheader()
        def metrics(table, context, value):
            for key, number in leaves(value):
                writers[table].writerow(dict(context, metric=key, value=number))
        def anchors(table, context, rows):
            for anchor in rows:
                metrics(table, dict(context, block=anchor.get("block"), prefill=anchor.get("prefill")), anchor)
        selected, observed, nonfinite = [], None, []
        raw = output / "raw/training"
        raw.mkdir(parents=True)
        for name in ("run_config.json", "first_update.json", "chain.json"):
            if (joint / name).is_file():
                shutil.copy2(joint / name, raw / name)
        for pattern in ("failure-job*.json", "chain-status-*.json"):
            for evidence in joint.glob(pattern):
                if evidence.is_file(): shutil.copy2(evidence, raw / evidence.name)
        with (raw / "train-steps.jsonl").open("w", encoding="utf-8") as stream:
            for record in json_lines(joint / "train.jsonl", warnings):
                observed = record.get("step", observed)
                if not first <= record.get("step", -1) <= last:
                    continue
                selected.append(record["step"])
                stream.write(json.dumps(record) + "\n")
                writers["training"].writerow(record)
                for key in ("loss", "outer_grad_norm"):
                    if isinstance(record.get(key), (int, float)) and not math.isfinite(record[key]):
                        nonfinite.append({"step": record["step"], "field": key})
                for rank in record.get("ranks", []):
                    context = {"step": record["step"], "rank": rank["rank"]}
                    metrics("training_ranks", context, {k: v for k, v in rank.items() if k not in {"chunks", "prefill"}})
        with (raw / "stability-steps.jsonl").open("w", encoding="utf-8") as stream:
            source = joint / "stability.jsonl"
            def stability():
                if source.is_file():
                    yield from json_lines(source, warnings)
                else:
                    warnings.append("stability.jsonl missing: recovered clean-anchor records from train.jsonl")
                    for record in json_lines(raw / "train-steps.jsonl", warnings):
                        for rank in record.get("ranks", []):
                            chunks = ([rank["prefill"]] if rank.get("prefill") else []) + rank.get("chunks", [])
                            for chunk in chunks:
                                for anchor in chunk.get("anchors", []):
                                    yield dict(anchor, step=record["step"], rank=rank["rank"], chunk=chunk["chunk"],
                                               start=chunk.get("start"), end=chunk.get("end"))
            for row in stability():
                if first <= row.get("step", -1) <= last:
                    stream.write(json.dumps(row) + "\n")
                    metrics("training_stability", row, row)
        audit["checkpoint"] = checkpoint_audit(joint)
        saved = audit["checkpoint"].get("step")
        duplicates = sorted({s for s in selected if selected.count(s) > 1})
        if duplicates:
            warnings.append(f"Duplicate training steps retained as separate observations: {duplicates}")
        audit["training"] = {"observed_last_step": observed, "selected_steps": selected, "duplicate_steps": duplicates,
            "missing_steps": sorted(set(range(first, last + 1)) - set(selected)),
            "nonfinite_loss_or_grad": nonfinite,
            "steps_after_published_checkpoint": [s for s in selected if saved is not None and s > saved]}
        def comparison(source, status, step, horizon, variant, history, values, identity):
            for key, value in values.items():
                if not isinstance(value, dict) or "sana" not in value:
                    continue
                a, b = value.get("sana"), value.get("ttn")
                writers["method_comparison"].writerow(dict(source=source, status=status, step=step,
                    horizon=horizon, variant=variant, history=history, metric=key, sana=a, ttn=b,
                    ttn_minus_sana=b-a if a is not None and b is not None else None,
                    paired_count=value.get("paired_count"), checkpoint_sha256=identity.get("checkpoint_sha256"),
                    fixed_cases_sha256=identity.get("fixed_cases_sha256")))
        def episode_tables(folder, label, default_step):
            summary = json_file(folder / "summary.json", warnings)
            protocol = summary.get("protocol", {})
            step = protocol.get("step", default_step)
            rows = json_lines(folder / "episodes.jsonl", warnings) if (folder / "episodes.jsonl").is_file() else summary.get("episodes", [])
            for row in rows:
                context = dict(source=label, step=step, case_id=row.get("case_id"), seed=row.get("seed"), method=row.get("method"))
                metrics("rollout_metrics", context, {k: row[k] for k in ("metrics", "timing", "commits", "predictions", "input_sha256", "initial_noise_sha256", "base_sha256") if k in row})
                for frame, value in enumerate(row.get("metrics", {}).get("per_frame_latent_mse", [])):
                    writers["rollout_frames"].writerow(dict(context, frame=frame, latent_mse=value))
                for chunk in row.get("chunks", []):
                    chunk_context = dict(context, chunk=chunk.get("chunk"))
                    metrics("rollout_chunks", chunk_context, {k: v for k, v in chunk.items() if k != "anchors"})
                    anchors("rollout_stability", chunk_context, chunk.get("anchors", []))
                for teacher, probe in (("original_sana", row), ("matched_backbone_softmax", row.get("matched_backbone_softmax", {}))):
                    for item in probe.get("probes", []):
                        base = dict(context, teacher=teacher, timestep=item.get("timestep"))
                        anchors("teacher_alignment", base, item.get("anchors", []))
                        metrics("teacher_alignment", dict(base, block="final_flow"), item.get("native_flow", {}))
                        for drift in item.get("representation_drift", []):
                            metrics("representation_drift", dict(base, block=drift.get("block"), point=drift.get("point")), drift)
                    anchors("teacher_health", dict(context, teacher=teacher, kind="parameter_change"), probe.get("parameter_changes", []))
                    anchors("teacher_health", dict(context, teacher=teacher, kind="gradient_probe"), (probe.get("gradient_health") or {}).get("anchors", []))
        milestones = []
        for folder in sorted((joint / "evaluations").glob("step-*")):
            match = re.fullmatch(r"step-(\d+)", folder.name)
            if not match or not first <= int(match[1]) <= last or not folder.is_dir():
                continue
            step = int(match[1])
            root = json_file(folder / "summary.json", warnings)
            status, identity = root.get("status", "missing"), root.get("identity", {})
            label = f"stage-{step:06d}"
            copy_logs(folder, output / "raw/evaluations" / folder.name)
            milestones.append({"step": step, "status": status, "completed_parts": sorted(root.get("results", {})), "identity": identity})
            for horizon in ("short", "long", "align"):
                part = root.get("results", {}).get(horizon, {})
                comparison(label, status, step, horizon, "full", "generated", part.get("metrics") or {}, identity)
                episode_tables(folder / horizon, f"{label}/{horizon}", step)
        audit["milestones"] = milestones
        mechanism_summary = json_file(mechanism / "summary.json", warnings)
        audit["mechanisms"] = {k: mechanism_summary.get(k) for k in ("status", "identity", "variants", "full_c_reproduction")}
        copy_logs(mechanism, output / "raw/mechanisms")
        step = (mechanism_summary.get("identity") or {}).get("step")
        for name, value in mechanism_summary.get("results", {}).items():
            for horizon in ("short", "long"):
                comparison("mechanisms", mechanism_summary.get("status"), step, horizon, name,
                           value.get("history_source"), value.get(horizon, {}).get("metrics", {}), mechanism_summary.get("identity") or {})
            episode_tables(mechanism / name, f"mechanisms/{name}", step)
        for name, value in mechanism_summary.get("contrasts", {}).items():
            for horizon, entries in value.items():
                for key, delta in entries.items():
                    writers["mechanism_contrasts"].writerow(dict(contrast=name, horizon=horizon, metric=key, delta=delta))
    jobs = set()
    registered = []
    for filename in ("jobs.jsonl", "eval_jobs.jsonl", "implementations.jsonl"):
        path = joint / filename
        if path.is_file():
            shutil.copy2(path, raw / filename)
        for row in json_lines(path, warnings):
            relevant = (row.get("stop_step", -1) >= first and row.get("start_step", last) < last) if filename == "jobs.jsonl" else first <= row.get("step", -1) <= last
            if relevant and str(row.get("job", "")).isdecimal():
                jobs.add(str(row["job"]))
                registered.append(row)
    job = json_file(mechanism / "job.json", warnings).get("job")
    if str(job).isdecimal():
        jobs.add(str(job))
    for job in jobs:
        path = joint / f"slurm-{job}.out"
        if path.is_file():
            shutil.copy2(path, raw / path.name)
    audit["selected_jobs"] = sorted(jobs)
    if slurm:
        slurm_evidence(sorted(jobs), output / "raw/slurm", warnings, registered)
    else:
        warnings.append("Slurm accounting explicitly skipped")
    def fmt(value):
        return f"{value:.6f}" if isinstance(value, (int, float)) else "NA"
    def value(part, key, method):
        return (part.get("metrics") or {}).get(key, {}).get(method)
    report = [f"Training requested {first}..{last}; observed last={observed}; saved checkpoint={saved}",
              f"Selected records={len(selected)}; nonfinite loss/grad={len(nonfinite)}; missing steps={audit['training']['missing_steps']}",
              f"Logged after published checkpoint={audit['training']['steps_after_published_checkpoint']}",
              "", f"Step{step} mechanism comparison: 13 mean | 61 mean | tail mean | final chunk | 61 slope"]
    for name, result in mechanism_summary.get("results", {}).items():
        short, long = result.get("short", {}), result.get("long", {})
        for method in ("sana", "ttn"):
            values = [value(short, "mean_future_latent_mse", method), value(long, "mean_future_latent_mse", method),
                      value(long, "after_training_horizon_mean_latent_mse", method),
                      value(long, "final_chunk_latent_mse", method), value(long, "error_slope_per_latent_frame", method)]
            report.append(f"{name}/{method}: " + " | ".join(fmt(v) for v in values))
    report += ["", "Milestones: step/status | 13 mean SANA/TTN | 61 mean SANA/TTN | 61 final SANA/TTN"]
    for milestone in milestones:
        root = json_file(joint / f"evaluations/step-{milestone['step']:06d}/summary.json", warnings)
        parts = root.get("results", {})
        fields = [f"{milestone['step']}/{milestone['status']}"]
        for horizon, metric in (("short", "mean_future_latent_mse"), ("long", "mean_future_latent_mse"), ("long", "final_chunk_latent_mse")):
            fields.append("/".join(fmt(value(parts.get(horizon, {}), metric, method)) for method in ("sana", "ttn")))
        report.append(" | ".join(fields))
    report += ["", "All metrics/head detail: CSVs. Undefined/missing values are NA or empty, not zero.",
               "Original evaluation protocols, cases/seeds/hashes and failure logs: raw/.",
               "No weights/optimizer shards/latent .pt files are included."]
    (output / "summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(output, arcname=output.name)
    print(f"[TTN export] requested={first}..{last} records={len(selected)} checkpoint={saved} observed_last={observed}")
    print(f"[TTN export] mechanism={audit['mechanisms']['status']} milestones=" + ", ".join(f"{m['step']}:{m['status']}" for m in milestones))
    print(f"[TTN export] warnings={len(warnings)}; details in audit.json")
    print(f"[TTN export] directory={output}\n[TTN export] archive={archive}")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Experiment root containing joint/")
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=100)
    parser.add_argument("--last-step", type=int, default=170)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-slurm", action="store_true", help="For locally downloaded logs only")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or args.run.parent / f"analysis-{args.first_step}-{args.last_step}-{stamp}"
    export_analysis(args.run, args.mechanism, output, args.first_step, args.last_step, slurm=not args.no_slurm)


if __name__ == "__main__":
    main()
