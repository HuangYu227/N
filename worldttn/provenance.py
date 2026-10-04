"""Implementation identity for interpreting training/evaluation logs."""
import hashlib
import json
from pathlib import Path
import subprocess


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    files = ("worldttn/cli.py", "worldttn/anchor.py", "worldttn/core.py", "worldttn/controller.py", "worldttn/runtime.py",
             "worldttn/geometry.py", "worldttn/session.py", "worldttn/training.py", "worldttn/sana.py",
             "worldttn/evaluation.py", "worldttn/alignment.py", "worldttn/alignment_detail.py", "worldttn/stability.py",
             "worldttn/stage_evaluation.py", "worldttn/mechanism_evaluation.py", "worldttn/checkpoint.py", "worldttn/distributed.py",
             "worldttn/parallel_checkpoint.py", "worldttn/parallel_data.py",
             "diffusion/model/nets/sana_blocks.py", "diffusion/model/nets/sana_gdn_blocks.py",
             "diffusion/model/nets/sana_camctrl_blocks.py", "diffusion/data/datasets/video/sana_wm_zip_latent_data.py",
             "diffusion/model/nets/sana_gdn_camctrl_blocks.py",
             "diffusion/model/nets/sana_multi_scale_video_camctrl.py",
             "diffusion/scheduler/self_forcing_flow_euler_sampler.py")
    identity = {"source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}}
    identity["source_fingerprint"] = hashlib.sha256(json.dumps(identity["source_sha256"], sort_keys=True).encode()).hexdigest()
    try:
        def git(*args):
            return subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", *args], cwd=root,
                                  capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        identity.update(git_commit=git("rev-parse", "HEAD"), tracked_dirty=bool(git("status", "--porcelain", "--untracked-files=no")))
    except (OSError, subprocess.SubprocessError):
        identity.update(git_commit=None, tracked_dirty=None)
    return identity


def camera_contract(checkpoint_mode, requested=None, *, ablation=False):
    effective = requested or checkpoint_mode
    changed = effective != checkpoint_mode
    if changed and not ablation:
        raise ValueError("camera attention differs from training; use --camera-ablation for an explicitly labelled operator/cache ablation")
    return {"checkpoint_camera_attention": checkpoint_mode, "ttn_camera_attention": effective,
            "camera_ablation": changed, "camera_weight_source": "loaded TTN checkpoint; no implicit base-weight reset"}
