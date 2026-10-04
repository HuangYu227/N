"""Exercise Slurm argument/environment wiring without allocating GPUs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["valid", "sana-camera", "align-chunk", "stage-evaluate", "base-python", "multi-node"])
def test_eval_launcher_uses_prefix_python_one_task_and_existing_local_cache_worker(tmp_path, mode):
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash not installed")
        bash = Path(candidate)
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "WorldTTN"
    (project / "worldttn").mkdir(parents=True)
    (project / "worldttn/evaluation.py").touch()
    (project / "tools").mkdir()
    shutil.copyfile(root / "tools/ttn_cache_env.sh", project / "tools/ttn_cache_env.sh")
    training = project / "output/completed"
    training.mkdir(parents=True)
    for name in ("run_config.json", "train.jsonl"): (training / name).touch()
    python = tmp_path / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nexit 99\n', newline="\n")
    python.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    capture = tmp_path / "capture.json"
    writer = tmp_path / "capture.py"
    writer.write_text("import os, sys, json\nfrom pathlib import Path\n"
                      "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'env': "
                      "{k: os.getenv(k) for k in ('PYTHON', 'ROOT', 'RANK', 'MASTER_ADDR', 'CUDA_VISIBLE_DEVICES', "
                      "'GDN_DISABLE_COMPILE', 'HF_HOME')}}))\n")
    srun = bins / "srun"
    srun.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$WRITER" "$@"\n', newline="\n")
    srun.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(ROOT=tmp_path.as_posix(), PROJECT_ROOT=project.as_posix(),
               PYTHON=python.as_posix() if mode != "base-python" else "/wrong/base/python",
               SLURM_JOB_ID="12345", SLURM_NTASKS="1" if mode != "multi-node" else "4",
               SLURM_JOB_NUM_NODES="1" if mode != "multi-node" else "4",
               TRAINING_RUN=training.as_posix(), RANK="3", MASTER_ADDR="old-master",
               CUDA_VISIBLE_DEVICES="GPU-slurm-mask", GDN_DISABLE_COMPILE="1",
               REAL_PYTHON=Path(sys.executable).as_posix(), WRITER=writer.as_posix(), CAPTURE=capture.as_posix(),
               SCRIPT=(root / "tools/ttn_slurm_eval.sbatch").as_posix(), MOCK_BIN=bins.as_posix())
    for name in ("OUTPUT", "FRAMES", "STEPS", "ADAPTER", "BASE_WEIGHTS", "SANA_CONFIG", "CONFIG", "CROSS_ATTN_BACKEND", "CAMERA_ATTENTION", "COMMAND"):
        env.pop(name, None)
    if mode == "align-chunk": env["COMMAND"] = "align-chunk"
    if mode == "stage-evaluate":
        env["COMMAND"] = "stage-evaluate"
        env["FIXED_CASES"] = str(training / "fixed-cases.pt")
    if mode == "sana-camera": env["CAMERA_ATTENTION"] = "sana"
    shell = 'export MSYS_NO_PATHCONV=1 MSYS2_ENV_CONV_EXCL="*"; export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash "$SCRIPT"'
    result = subprocess.run([str(bash), "-c", shell], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if mode not in ("valid", "sana-camera", "align-chunk", "stage-evaluate"):
        assert result.returncode != 0 and not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    args, exported = record["args"], record["env"]
    for flag in ("--nodes=1", "--ntasks=1", "--gres=gpu:1", "--mpi=none"): assert flag in args
    for flag, value in (("--parallel", "single"), ("--frames", "4" if mode == "align-chunk" else "61"), ("--cross-attn-backend", "math")):
        assert args[args.index(flag) + 1] == value
    if mode == "align-chunk":
        assert "align-chunk" in args and "--steps" not in args
        assert args[args.index("--output") + 1].endswith("/align-12345")
    else:
        assert args[args.index("--steps") + 1] == "20"
    if mode == "sana-camera":
        assert args[args.index("--camera-attention") + 1] == "sana"
    else:
        assert "--camera-attention" not in args
    assert args[args.index("bash") + 1].endswith("/tools/ttn_slurm_worker.sh")
    assert exported["PYTHON"] == env["PYTHON"] and exported["CUDA_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"]
    assert exported["RANK"] is None and exported["MASTER_ADDR"] is None
    assert exported["HF_HOME"] == tmp_path.as_posix() + "/.cache/huggingface"
    assert exported["GDN_DISABLE_COMPILE"] == "1"
