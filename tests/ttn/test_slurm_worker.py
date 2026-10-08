"""Exercise the actual shell worker without CUDA or a Slurm allocation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def worker_setup(tmp_path):
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash not installed")
        bash = Path(candidate)
    root = Path(__file__).resolve().parents[2]
    script = root / "tools/ttn_slurm_worker.sh"
    assert script.exists(), "per-node compile setup is missing"
    capture = tmp_path / "worker.json"
    writer = tmp_path / "capture.py"
    writer.write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "names = ('ROOT', 'HF_HOME', 'PIP_CACHE_DIR', 'TORCH_HOME', 'PYTHON', 'CUDA_VISIBLE_DEVICES', "
        "'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', "
        "'TORCHINDUCTOR_COMPILE_THREADS', 'TMPDIR', 'TMP', 'TEMP', 'PYTHONPYCACHEPREFIX', "
        "'GDN_DISABLE_COMPILE', 'GDN_DISABLE_COMPLEX_COMPILE', 'CUDA_LAUNCH_BLOCKING')\n"
        "record = {k: os.environ[k] for k in names}\n"
        "for k in ('TMPDIR', 'PYTHONPYCACHEPREFIX', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH'):\n"
        "    p = Path(record[k]); assert p.is_dir(); (p / 'generated.py').write_text('compile scratch')\n"
        "import tempfile\n"
        "assert Path(tempfile.gettempdir()) == Path(record['TMPDIR'])\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    p = Path(td); assert p.parent == Path(record['TMPDIR']); (p / '__pycache__').mkdir()\n"
        "    (p / '__pycache__/kernel.pyc').write_bytes(b'compile scratch')\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'env': record}))\n"
        "sys.exit(int(os.environ['EXIT_CODE']))\n", encoding="utf-8")
    python = tmp_path / "python"
    python.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$CAPTURE_WRITER" "$@"\n',
                      encoding="utf-8", newline="\n")
    python.chmod(0o755)
    personal = tmp_path / "shared personal root"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(ROOT=personal.as_posix(), HF_HOME=(personal / ".cache/huggingface").as_posix(),
               PIP_CACHE_DIR=(personal / ".cache/pip").as_posix(),
               TORCH_HOME=(personal / ".cache/torch").as_posix(), PYTHON=python.as_posix(),
               REAL_PYTHON=Path(sys.executable).as_posix(), CAPTURE_WRITER=writer.as_posix(),
               CAPTURE=capture.as_posix(), EXIT_CODE="0", CUDA_VISIBLE_DEVICES="GPU-slurm-mask",
               SLURM_JOB_ID="423009", SLURM_PROCID="0", SLURM_LOCALID="0",
               TORCHINDUCTOR_CACHE_DIR="/shared/old/inductor", TRITON_CACHE_DIR="/shared/old/triton",
               TORCH_EXTENSIONS_DIR="/shared/old/extensions", CUDA_CACHE_PATH="/shared/old/cuda",
               TMPDIR="/shared/old/tmp", TMP="/shared/old/tmp", TEMP="/shared/old/tmp",
               PYTHONPYCACHEPREFIX="/shared/old/pycache")
    env.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
    for name in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE", "CUDA_LAUNCH_BLOCKING"):
        env.pop(name, None)
    # Use Git Bash's native path conversion so Windows Python can inspect the
    # temporary directories created under its /tmp alias.
    env.pop("MSYS2_ENV_CONV_EXCL", None)
    env.pop("MSYS_NO_PATHCONV", None)
    if os.name == "nt":
        # Git Bash maps /tmp using Windows TEMP during startup. Keep that
        # bootstrap mapping valid; Python's preferred TMPDIR is still stale.
        env["TMP"] = os.environ["TEMP"]
        env["TEMP"] = os.environ["TEMP"]
    return bash, script, capture, writer, personal, env


@pytest.mark.parametrize("tasks", [1, 3, 5])
def test_proximal_meta_launcher_selects_tests_and_rejects_invalid_topology(worker_setup, tmp_path, tasks):
    bash, worker, capture, writer, personal, env = worker_setup
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    for name in ("ttn_slurm_meta_tests.sbatch", "ttn_cache_env.sh"):
        shutil.copyfile(worker.parent / name, project / "tools" / name)
    python = personal / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\n[[ "$1" == -c ]] || exit 99\nprintf "10.0.0.1\\n"\n',
                      encoding="utf-8", newline="\n")
    python.chmod(0o755)
    writer.write_text("import json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({'args':sys.argv[1:],"
        "'master':os.environ['MASTER_ADDR'],'module':os.environ['TTN_ENTRY_MODULE']}))\n", encoding="utf-8")
    mocks = tmp_path / "mocks"
    mocks.mkdir()
    for name, body in (
        ("scontrol", 'printf "ltu-hpc-1\\n"'),
        ("srun", 'exec "$REAL_PYTHON" "$CAPTURE_WRITER" "$@"'),
    ):
        path = mocks / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n", encoding="utf-8", newline="\n")
        path.chmod(0o755)
    env.update(PROJECT_ROOT=project.as_posix(), PROXIMAL_MEMORY_TESTS="1", FULL_MEMORY_TESTS="1",
               SLURM_NTASKS=str(tasks), SLURM_JOB_NODELIST="ltu-hpc-1")
    env["PATH"] = str(mocks) + os.pathsep + env["PATH"]
    result = subprocess.run([str(bash), str(project / "tools/ttn_slurm_meta_tests.sbatch")], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    if tasks == 5:
        assert result.returncode == 2 and "requires one, two, three or four" in result.stderr
        assert not capture.exists()
        return
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["master"] == "10.0.0.1" and record["module"] == "pytest"
    assert f"--ntasks={tasks}" in record["args"] and "--kill-on-bad-exit=1" in record["args"]
    tests = [arg for arg in record["args"] if arg.startswith("tests/")]
    if tasks == 3:
        assert tests == ["tests/ttn/test_proximal_cuda.py"]
    else:
        assert "tests/ttn/test_proximal_device.py" in tests and "tests/ttn/test_proximal_runtime.py" in tests
        assert "tests/ttn/test_meta_cuda.py" not in tests


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_uses_fresh_local_compile_files_and_preserves_exit_status(worker_setup, rank, exit_code):
    bash, script, capture, writer, personal, env = worker_setup
    env.update(SLURM_PROCID=str(rank), EXIT_CODE=str(exit_code))
    args = ["distributed-smoke", "--stages", "A", "--output", (personal / "output").as_posix()]
    result = subprocess.run([str(bash), str(script), *args], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == exit_code, result.stderr
    record = json.loads(capture.read_text())
    assert record["args"] == ["-u", "-m", "worldttn.cli", *args]
    exported = record["env"]
    for name in ("ROOT", "HF_HOME", "PIP_CACHE_DIR", "TORCH_HOME", "PYTHON", "CUDA_VISIBLE_DEVICES"):
        assert exported[name] == env[name]
    assert exported["TORCHINDUCTOR_COMPILE_THREADS"] == "1"
    for name in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE", "CUDA_LAUNCH_BLOCKING"):
        assert exported[name] == "0" and f"{name}=0" in result.stdout
    assert exported["TMP"] == exported["TEMP"] == exported["TMPDIR"]
    parents = {Path(exported[k]).parent for k in (
        "TMPDIR", "PYTHONPYCACHEPREFIX", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH")}
    assert len(parents) == 1
    scratch = parents.pop()
    assert scratch.name.startswith(f"worldttn-423009-rank{rank}.")
    assert not scratch.exists(), "compile scratch must be cleaned after success and failure"
    assert "[TTN compile]" in result.stdout and f"rank={rank}" in result.stdout


@pytest.mark.parametrize("mode", ["compiled", "eager", "complex-eager"])
def test_worker_passes_compile_and_synchronous_debug_policy_before_python(worker_setup, mode):
    bash, script, capture, writer, personal, env = worker_setup
    env.update(GDN_DISABLE_COMPILE="1" if mode == "eager" else "0",
               GDN_DISABLE_COMPLEX_COMPILE="1" if mode == "complex-eager" else "0",
               CUDA_LAUNCH_BLOCKING="1")
    result = subprocess.run([str(bash), str(script), "distributed-smoke", "--stages", "A"],
                            env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    exported = json.loads(capture.read_text())["env"]
    for name in ("GDN_DISABLE_COMPILE", "GDN_DISABLE_COMPLEX_COMPILE", "CUDA_LAUNCH_BLOCKING"):
        assert exported[name] == env[name] and f"{name}={env[name]}" in result.stdout


def test_concurrent_workers_have_separate_local_scratch(worker_setup):
    bash, script, capture, writer, personal, env = worker_setup
    writer.write_text(writer.read_text().replace("sys.exit(int", "import time; time.sleep(1)\nsys.exit(int"),
                      encoding="utf-8")
    processes = []
    for rank in range(3):
        rank_env = {**env, "SLURM_PROCID": str(rank), "CAPTURE": str(capture.with_name(f"rank-{rank}.json"))}
        processes.append(subprocess.Popen([str(bash), str(script), "distributed-check"], env=rank_env,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                          encoding="utf-8", errors="replace"))
    scratch_dirs = set()
    for rank, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        record = json.loads(capture.with_name(f"rank-{rank}.json").read_text())
        scratch = Path(record["env"]["TMPDIR"]).parent
        assert f"rank={rank}" in stdout and not scratch.exists()
        scratch_dirs.add(scratch)
    assert len(scratch_dirs) == 3


@pytest.mark.parametrize("module", ["pytest", "tools.ttn_custom_inference"])
def test_worker_uses_allowlisted_entry_and_cleans_scratch(worker_setup, module):
    bash, script, capture, writer, personal, env = worker_setup
    env["TTN_ENTRY_MODULE"] = module
    result = subprocess.run([str(bash), str(script), "-q", "tests/ttn/test_meta_core.py"], env=env,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    assert record["args"][:3] == ["-u", "-m", module]
    assert not Path(record["env"]["TMPDIR"]).parent.exists()


def test_worker_forwards_term_and_cleans_its_scratch(worker_setup):
    bash, script, capture, writer, personal, env = worker_setup
    writer.write_text(writer.read_text().replace("sys.exit(int", "import time; time.sleep(60)\nsys.exit(int"),
                      encoding="utf-8")
    env["WORKER"] = script.as_posix()
    shell = ('bash "$WORKER" distributed-check & worker=$!; '
             'for attempt in {1..100}; do [[ ! -s "$CAPTURE" ]] || break; sleep 0.1; done; '
             'if [[ ! -s "$CAPTURE" ]]; then kill -TERM "$worker"; wait "$worker"; exit 99; fi; '
             'kill -TERM "$worker"; wait "$worker"')
    result = subprocess.run([str(bash), "-c", shell], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    assert result.returncode == 143, result.stderr
    record = json.loads(capture.read_text())
    assert not Path(record["env"]["TMPDIR"]).parent.exists()
