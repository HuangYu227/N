"""Check actual Bash array wiring without requesting cluster resources."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("task,fail,suite", [*( (task, False, "mechanisms") for task in range(6)),
    (0, True, "mechanisms"), *( (task, False, "histories") for task in range(4))])
def test_array_is_single_gpu_sequential_and_forwards_intervention(task, fail, suite, tmp_path):
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash unavailable")
        bash = Path(candidate)
    repo = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / "worldttn").mkdir()
    (project / "worldttn/evaluation.py").touch()
    for name in ("ttn_cache_env.sh", "ttn_slurm_eval.sbatch", "ttn_slurm_mechanism.sbatch"):
        shutil.copyfile(repo / "tools" / name, project / "tools" / name)
    training = tmp_path / "snapshot"
    training.mkdir()
    for name in ("run_config.json", "train.jsonl"): (training / name).touch()
    python = tmp_path / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    writer = tmp_path / "writer.py"
    writer.write_text("import json,os,sys\nfrom pathlib import Path\n"
                     "if sys.argv[1]=='-':\n"
                     "    sys.argv=sys.argv[1:]\n"
                     "    exec(compile(sys.stdin.read(),'<dispatch>','exec'))\n"
                     "    sys.exit(0)\n"
                     "p=Path(os.environ['CAPTURE'])\n"
                     "with p.open('a') as s: s.write(json.dumps(sys.argv[1:])+'\\n')\n"
                     "sys.exit(7 if sys.argv[1]=='srun' and os.environ['FAIL']=='1' else 0)\n")
    python.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$WRITER" "$@"\n', newline="\n")
    python.chmod(0o755)
    bins = tmp_path / "bin"
    bins.mkdir()
    (bins / "srun").write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$WRITER" srun "$@"\n', newline="\n")
    (bins / "srun").chmod(0o755)
    capture = tmp_path / "capture.jsonl"
    results = tmp_path / "results"
    if suite == "histories":
        results.mkdir()
        interventions = {"full": ["full", "generated", ["sana", "ttn"]],
                         "ttn-gt-history": ["full", "ttn-gt", ["ttn"]],
                         "native-gt-history": ["full", "native-gt", ["ttn"]],
                         "gt-history": ["full", "gt", ["sana", "ttn"]]}
        (results / "plan.json").write_text(json.dumps({"suite": suite, "variants": list(interventions),
                                                      "interventions": interventions}))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_"))}
    env.update(ROOT=tmp_path.as_posix(), PROJECT_ROOT=project.as_posix(), PYTHON=python.as_posix(),
               TRAINING_RUN=training.as_posix(), MECHANISM_OUTPUT=results.as_posix(),
               SLURM_JOB_ID="12345", SLURM_ARRAY_TASK_ID=str(task), SLURM_NTASKS="1", SLURM_JOB_NUM_NODES="1",
               REAL_PYTHON=Path(sys.executable).as_posix(), WRITER=writer.as_posix(), CAPTURE=capture.as_posix(),
               MOCK_BIN=bins.as_posix(), SCRIPT=(project / "tools/ttn_slurm_mechanism.sbatch").as_posix(), FAIL=str(int(fail)))
    for key in ("ADAPTER", "BASE_WEIGHTS", "SANA_CONFIG", "CONFIG", "CAMERA_ATTENTION", "CAMERA_ABLATION"):
        env.pop(key, None)
    command = 'export MSYS_NO_PATHCONV=1 MSYS2_ENV_CONV_EXCL="*"; export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash "$SCRIPT"'
    result = subprocess.run([str(bash), "-c", command], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == (7 if fail else 0), result.stderr
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert records[0][-1] == "running" and records[-1][-1] == ("failed" if fail else "completed")
    args = next(row for row in records if row[0] == "srun")
    assert "--nodes=1" in args and "--ntasks=1" in args and "--gres=gpu:1" in args
    assert args[args.index("--ttn-ablation") + 1] == ("full" if suite == "histories" else
        ("full", "no-ttt", "identity", "full", "no-local", "no-persistent")[task])
    assert args[args.index("--history-source") + 1] == (("generated", "ttn-gt", "native-gt", "gt")[task]
        if suite == "histories" else "gt" if task == 3 else "generated")
    assert "--state-diagnostics" in args and args[args.index("--frames") + 1] == "61"
    methods_index = args.index("--eval-methods")
    assert args[methods_index + 1] == ("ttn" if task in (1, 2, 4, 5) else "sana")
    assert "#SBATCH --array=0-3%1" in (repo / "tools/ttn_slurm_mechanism.sbatch").read_text()
