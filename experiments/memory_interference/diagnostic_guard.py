"""Run the recorded experiment in one bounded systemd cgroup; no torch import."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command(args):
    return subprocess.check_output(args, text=True, timeout=10).strip()


def alive(pid):
    try:
        if int(pid) <= 0:
            return False
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def systemd_status(unit):
    result = command(["systemctl", "--user", "show", unit,
                      "-p", "MainPID", "-p", "ControlGroup", "-p", "Result",
                      "-p", "ExecMainStatus", "-p", "MemoryMax", "-p", "MemorySwapMax"])
    return dict(line.split("=", 1) for line in result.splitlines() if "=" in line)


def memory_sample(unit, gpu_ids):
    status = systemd_status(unit)
    group = Path("/sys/fs/cgroup") / status.get("ControlGroup", "").lstrip("/")
    result = {"time": now(), "unit": status}
    if status.get("ControlGroup"):
        for name in ("memory.current", "memory.peak", "memory.events", "memory.swap.current"):
            try:
                result[name] = (group / name).read_text().strip()
            except OSError:
                pass
    result["host_meminfo"] = {
        line.split(":", 1)[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith(("MemAvailable:", "SwapFree:"))}
    result["host_memory_pressure"] = Path("/proc/pressure/memory").read_text().strip()
    result["gpu"] = command(["nvidia-smi", "--id=" + gpu_ids,
                             "--query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw",
                             "--format=csv,noheader,nounits"]).splitlines()
    return result


def main():
    run = Path(__file__).resolve().parent
    spec = json.loads((run / "launch.json").read_text())
    state_path = Path(spec["state_file"])
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get("status") == "running" and (
                alive(previous.get("pid")) or alive(previous.get("wrapper_pid"))):
            raise RuntimeError("Recorded experiment is still running; refusing a duplicate")
        write_json(run / "previous-state.json", previous)
    for item in spec["synced_files"]:
        actual = hashlib.sha256((Path(spec["cwd"]) / item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise RuntimeError("Source changed after verification: " + item["path"])
    gpu_ids = spec["env"]["CUDA_VISIBLE_DEVICES"]
    if gpu_ids not in ("6", "7"):
        raise ValueError("Only the user-authorized diagnostic GPUs 6 and 7 are allowed")
    gpu_rows = command(["nvidia-smi", "--id=" + gpu_ids, "--query-gpu=index,uuid,memory.used",
                        "--format=csv,noheader,nounits"]).splitlines()
    if len(gpu_rows) != 1 or any(int(row.split(",")[-1]) > 1024 for row in gpu_rows):
        raise RuntimeError("A requested GPU is occupied; refusing to share a training GPU")
    available = next(int(line.split()[1]) * 1024 for line in
                     Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    memory_max = spec["host_memory_max_gib"]
    if type(memory_max) is not int or memory_max != 64:
        raise ValueError("This actual-model inference diagnostic is capped at exactly 64 GiB")
    required_available = max(96, memory_max + 80)
    if available < (required_available << 30):
        raise RuntimeError(f"Less than {required_available} GiB available host RAM; refusing to start")
    unit = spec["unit"]
    if not re.fullmatch(r"worldtla-[a-z0-9-]+", unit):
        raise ValueError("Unexpected systemd unit name")
    load = command(["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"])
    if load != "not-found":
        raise RuntimeError("Unit name already exists; refusing to reuse it")
    state = {**spec, "mode": "launch-only", "status": "running", "start_time": now(),
             "wrapper_pid": os.getpid(), "pid": None, "exit_code": None,
             "gpu": gpu_rows, "tmux_session": "wm", "tmux_pane_id": os.environ["TMUX_PANE"],
             "log": str(run / "train.log"), "output_dir": str(run)}
    write_json(state_path, state)
    write_json(run / "state.json", state)
    args = ["systemd-run", "--user", "--wait", "--pipe", "--quiet", "--unit=" + unit,
            "--working-directory=" + spec["cwd"], "-p", f"MemoryMax={memory_max}G", "-p", "MemorySwapMax=0",
            "-p", "OOMPolicy=stop", "-p", "KillMode=control-group", "-p", "TimeoutStopSec=30s",
            "-p", "TasksMax=512", "-p", "RuntimeMaxSec=14400"]
    args += ["--setenv=" + key + "=" + value for key, value in spec["env"].items()]
    args += ["--", *spec["command"]]
    stop = threading.Event()

    def monitor():
        with (run / "host-memory.jsonl").open("a", buffering=1) as handle:
            while not stop.is_set():
                try:
                    sample = memory_sample(unit, gpu_ids)
                    handle.write(json.dumps(sample) + "\n")
                    available = sample["host_meminfo"]["MemAvailable"]
                    pid = int(sample["unit"].get("MainPID", 0))
                    if available < (64 << 30) and pid:
                        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
                        cwd = Path(f"/proc/{pid}/cwd").resolve()
                        if argv == spec["command"] and str(cwd) == spec["cwd"] and Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                            handle.write(json.dumps({"time": now(), "guard": "host_MemAvailable_below_64GiB", "unit": unit}) + "\n")
                            state["guard_reason"] = "host_MemAvailable_below_64GiB"
                            write_json(state_path, state)
                            write_json(run / "state.json", state)
                            subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=45)
                            return
                    if pid and state["pid"] != pid:
                        state["pid"] = pid
                        write_json(state_path, state)
                        write_json(run / "state.json", state)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    handle.write(json.dumps({"time": now(), "monitor_error": str(error)}) + "\n")
                stop.wait(10)

    code = 1
    process = None
    thread = threading.Thread(target=monitor)
    def interrupt(signum, frame):
        raise KeyboardInterrupt("Wrapper received signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    try:
        print(f"[L20 guarded] experiment={spec['experiment_id']} GPUs={gpu_ids} "
              f"MemoryMax={memory_max}GiB swap=0", flush=True)
        with (run / "train.log").open("a", buffering=1) as handle:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            thread.start()
            for line in process.stdout:
                handle.write(line)
                print(line, end="", flush=True)
            code = process.wait()
    except BaseException:
        # This unique unit was absent before launch. Stop only its verified command.
        if process is not None:
            pid = int(systemd_status(unit).get("MainPID", 0))
            if pid:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
                cwd = Path(f"/proc/{pid}/cwd").resolve()
                if argv == spec["command"] and str(cwd) == spec["cwd"]:
                    subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=45)
            process.wait(timeout=45)
        raise
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join(timeout=25)
        try:
            state["final_unit_status"] = systemd_status(unit)
        except (OSError, subprocess.SubprocessError):
            pass
        final_pid = (state["final_unit_status"].get("MainPID") if "final_unit_status" in state
                     else state.get("pid") if code else None)
        live = alive(final_pid)
        state.update(status="running" if live else ("completed" if code == 0 else "failed"),
                     exit_code=None if live else code, end_time=None if live else now())
        if not live:
            (run / "exit-code.txt").write_text(str(code) + "\n")
        else:
            state["pid"] = int(final_pid)
            state["attention"] = "Wrapper ended but systemd unit remains alive; do not launch a duplicate"
        write_json(state_path, state)
        write_json(run / "state.json", state)
        print(f"[L20 guarded] status={state['status']} exit_code={state['exit_code']} output={run}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
