"""Isolated periodic evaluations of one immutable completed training snapshot."""
import json
from pathlib import Path
import subprocess
import sys


def stage_evaluate_command(args):
    from .performance import ExecutionOptions, validate_execution
    validate_execution(ExecutionOptions(getattr(args, "ttn_core_backend", "reference"),
                       getattr(args, "ttn_psi_backend", "reference")), "C", diagnostics=True)
    from .provenance import implementation_identity
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.frames != 61 or not args.fixed_cases:
        raise ValueError("stage-evaluate requires --frames 61 and a shared --fixed-cases path")
    common = ["--parallel", "single", "--training-run", args.training_run, "--seed", str(args.seed),
              "--eval-cases", str(args.eval_cases), "--fixed-cases", args.fixed_cases, "--noise-frames", "61",
              "--cross-attn-backend", args.cross_attn_backend, "--device", args.device]
    common += ["--ttn-core-backend", getattr(args, "ttn_core_backend", "reference"),
               "--ttn-psi-backend", getattr(args, "ttn_psi_backend", "reference")]
    for flag in ("ttn_profile", "ttn_layout_audit"):
        if getattr(args, flag, False): common += ["--" + flag.replace("_", "-")]
    for name in ("base_weights", "sana_config", "dataset_root", "config"):
        if getattr(args, name, None): common += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    commands = {
        # Long first creates the full immutable case bundle, then every diagnostic uses prefixes.
        "long": ["evaluate", "--frames", "61", "--steps", str(args.steps),
                 "--cfg-scale", str(args.cfg_scale), "--cached-blocks", str(args.cached_blocks)],
        "short": ["evaluate", "--frames", "13", "--steps", str(args.steps),
                  "--cfg-scale", str(args.cfg_scale), "--cached-blocks", str(args.cached_blocks)],
        "align": ["align-chunk", "--frames", "4", "--alignment-timesteps", "500", "--alignment-grad-timestep", "500"],
    }
    result = {"status": "running", "training_run": args.training_run, "provenance": implementation_identity(), "results": {}}
    status = output / "summary.json"
    def publish():
        tmp = output / "summary.tmp"
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(status)
    publish()
    try:
        for name, command in commands.items():
            if name != "align" and getattr(args, "ttn_compare_reference", False): command += ["--ttn-compare-reference"]
            subprocess.run([sys.executable, "-u", "-m", "worldttn.cli", *command, *common,
                            "--output", str(output / name)], check=True)
            summary = json.loads((output / name / "summary.json").read_text(encoding="utf-8"))
            protocol = summary["protocol"]
            identities = {k: protocol[k] for k in ("step", "stage", "checkpoint_sha256", "fixed_cases_sha256")}
            identities["evaluation_provenance"] = protocol.get("provenance")
            if "identity" in result and result["identity"] != identities:
                raise ValueError("periodic evaluations used different checkpoints/cases")
            result["identity"] = identities
            result["results"][name] = {"path": str(output / name / "summary.json"),
                "metrics": summary.get("metrics"), "ttn_minus_sana": summary.get("ttn_minus_sana"),
                "backend_comparison": summary.get("backend_comparison"),
                "teacher_diagnoses": [{"original": r["diagnostic_summary"],
                    "matched": r["matched_backbone_softmax"]["diagnostic_summary"]} for r in summary["episodes"]] if name == "align" else None}
            publish()
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=str(error))
        publish()
        raise
    publish()
    print("[TTN stage evaluation] " + json.dumps(result), flush=True)
