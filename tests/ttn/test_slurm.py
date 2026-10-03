import ast
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def slurm_env(rank=0, world=3):
    return {"SLURM_PROCID": str(rank), "SLURM_LOCALID": "0", "SLURM_NODEID": str(rank),
            "SLURM_NTASKS": str(world), "SLURM_NNODES": str(world), "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": f"ltu-hpc-[1-{world}]", "MASTER_ADDR": "ltu-hpc-1"}


@pytest.mark.parametrize("world,rank", [(2, 1), (3, 0), (3, 1), (3, 2), (5, 4)])
def test_slurm_maps_global_ranks_but_keeps_local_gpu_zero(world, rank):
    from worldttn import distributed
    assert hasattr(distributed, "resolve_launch_environment"), "Slurm launch normalization is missing"
    env = slurm_env(rank, world)
    env["CUDA_VISIBLE_DEVICES"] = "GPU-allocated-by-slurm"
    launch = distributed.resolve_launch_environment(env)
    assert launch["launcher"] == "slurm"
    assert (launch["rank"], launch["local_rank"], launch["world_size"], launch["node_rank"]) == (rank, 0, world, rank)
    assert env["RANK"] == str(rank) and env["LOCAL_RANK"] == "0" and env["WORLD_SIZE"] == str(world)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-by-slurm"
    assert int(env["MASTER_PORT"]) == 15000 + 12345 % 40000
    assert distributed.resolve_launch_environment(env) == launch, "normalization must be idempotent"


def test_slurm_step_nodes_and_explicit_master_port(monkeypatch):
    from worldttn import distributed
    assert hasattr(distributed, "resolve_launch_environment"), "Slurm launch normalization is missing"
    import subprocess
    calls = []
    def hosts(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="ltu-hpc-3\nltu-hpc-4\nltu-hpc-5\n")
    monkeypatch.setattr(subprocess, "run", hosts)
    monkeypatch.setattr(distributed.socket, "gethostbyname", lambda host: "10.0.0.3" if host == "ltu-hpc-3" else pytest.fail(host))
    env = slurm_env()
    env.pop("MASTER_ADDR")
    env.update(SLURM_STEP_NODELIST="ltu-hpc-[3-5]", SLURM_STEP_NUM_NODES="3",
               SLURM_NNODES="5", MASTER_PORT="23456")
    distributed.resolve_launch_environment(env)
    assert calls == [["scontrol", "show", "hostnames", "ltu-hpc-[3-5]"]]
    assert (env["MASTER_ADDR"], env["MASTER_PORT"]) == ("10.0.0.3", "23456")


def test_slurm_auto_master_reports_ipv4_resolution_failure(monkeypatch):
    from worldttn import distributed
    monkeypatch.setattr(distributed.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="ltu-hpc-1\nltu-hpc-2\nltu-hpc-3\n"))
    def no_ipv4(host): raise distributed.socket.gaierror("no IPv4 address")
    monkeypatch.setattr(distributed.socket, "gethostbyname", no_ipv4)
    env = slurm_env()
    env.pop("MASTER_ADDR")
    with pytest.raises(ValueError, match="IPv4"): distributed.resolve_launch_environment(env)


@pytest.mark.parametrize("change,match", [
    ({"RANK": "1"}, "conflict"), ({"LOCAL_RANK": "1"}, "conflict"),
    ({"WORLD_SIZE": "2"}, "conflict"), ({"SLURM_LOCALID": "1"}, "one task"),
    ({"SLURM_NNODES": "2"}, "one task"), ({"LOCAL_WORLD_SIZE": "2"}, "nested"),
    ({"SLURM_PROCID": "3"}, "rank"), ({"MASTER_ADDR": "127.0.0.1"}, "loopback"),
    ({"MASTER_PORT": "0"}, "MASTER_PORT"), ({"SLURM_NTASKS": "wrong"}, "SLURM_NTASKS"),
])
def test_invalid_slurm_topology_or_launcher_fails_early(change, match):
    from worldttn import distributed
    assert hasattr(distributed, "resolve_launch_environment"), "Slurm launch normalization is missing"
    env = slurm_env()
    env.update(change)
    with pytest.raises(ValueError, match=match): distributed.resolve_launch_environment(env)


def test_partial_slurm_environment_and_batch_shell(monkeypatch):
    from worldttn import distributed
    assert hasattr(distributed, "resolve_launch_environment"), "Slurm launch normalization is missing"
    env = slurm_env()
    env.pop("SLURM_LOCALID")
    with pytest.raises(ValueError, match="SLURM_LOCALID"): distributed.resolve_launch_environment(env)
    assert distributed.resolve_launch_environment({"SLURM_JOB_ID": "12345"})["world_size"] == 1
    env = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "localhost",
           "MASTER_PORT": "23456"}
    before = dict(env)
    launch = distributed.resolve_launch_environment(env)
    assert launch["launcher"] == "env" and launch["local_rank"] == 1 and env == before


def install_environment(monkeypatch, values):
    for name in list(os.environ):
        if name.startswith("SLURM_") or name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK",
                                                 "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            monkeypatch.delenv(name)
    for name, value in values.items(): monkeypatch.setenv(name, value)


def test_slurm_initialize_binds_gpu_zero_before_nccl(monkeypatch):
    from worldttn import distributed
    install_environment(monkeypatch, slurm_env(2))
    calls = []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: calls.append(("device", index)))
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda *args, **kwargs: calls.append(("init", args, kwargs)))
    mode, device = distributed.initialize("auto", "cuda", timeout_seconds=123)
    assert mode == "ddp" and device == torch.device("cuda:0")
    assert calls[0] == ("device", 0)
    options = calls[1][2]
    assert options["backend"] == "nccl" and options["init_method"] == "env://"
    assert (options["rank"], options["world_size"]) == (2, 3)
    assert options["timeout"].total_seconds() == 123
    assert options["device_id"] == torch.device("cuda:0")


@pytest.mark.parametrize("count", [0, 2])
def test_slurm_requires_exactly_one_visible_gpu(monkeypatch, count):
    from worldttn import distributed
    install_environment(monkeypatch, slurm_env())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    with pytest.raises(ValueError, match="one visible GPU"): distributed.initialize("ddp", "cuda")


@pytest.mark.parametrize("device,env,expected", [
    ("cuda:2", {}, 2),
    ("cuda", {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2", "MASTER_ADDR": "localhost", "MASTER_PORT": "23456"}, 1)
])
def test_non_slurm_device_binding_remains_compatible(monkeypatch, device, env, expected):
    from worldttn import distributed
    install_environment(monkeypatch, env)
    calls = []
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "init_process_group", lambda **kwargs: None)
    mode, selected = distributed.initialize("auto", device)
    assert mode == ("fsdp2" if env else "single")
    assert calls == [expected] and selected.index == expected


def test_cli_resolves_slurm_before_world_size_validation(monkeypatch):
    from worldttn import cli, distributed
    import sys
    install_environment(monkeypatch, slurm_env())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    class ReachedInitialize(RuntimeError): pass
    def initialize(*args, **kwargs):
        assert os.environ["WORLD_SIZE"] == "3" and os.environ["LOCAL_RANK"] == "0"
        assert kwargs["timeout_seconds"] == 123
        raise ReachedInitialize
    monkeypatch.setattr(distributed, "initialize", initialize)
    monkeypatch.setattr(sys, "argv", ["worldttn", "distributed-smoke", "--distributed-timeout", "123"])
    with pytest.raises(ReachedInitialize): cli.main()


@pytest.mark.parametrize("global_only,rank,level", [(True, 1, logging.ERROR), (True, 0, logging.INFO), (False, 1, logging.INFO)])
def test_sana_global_rank_logging_and_worker_environment_fallback(monkeypatch, global_only, rank, level):
    # Exercise the actual logger function without importing the CUDA/mmcv SANA dependency stack.
    source = Path(__file__).resolve().parents[2] / "diffusion/utils/logger.py"
    function = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "get_logger")
    namespace = {"logging": logging, "os": os, "logger_initialized": {}, "is_local_master": lambda: True,
                 "dist": SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False),
                 "TimezoneFormatter": lambda fmt, datefmt, tz: logging.Formatter(fmt, datefmt)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setenv("SANA_LOG_GLOBAL_RANK_ONLY", "1" if global_only else "0")
    monkeypatch.setenv("RANK", str(rank))
    logger = namespace["get_logger"]("ttn-slurm-worker-test")
    try:
        assert logger.level == level, "TTN uses global rank; original SANA entrypoints retain local-rank policy"
        assert logger.isEnabledFor(logging.ERROR)
    finally:
        for handler in list(logger.handlers): logger.removeHandler(handler); handler.close()


@pytest.mark.parametrize("global_only,rank", [(True, 1), (True, 0), (False, 1)])
def test_real_checkpoint_loading_messages_follow_ttn_rank_policy(tmp_path, monkeypatch, capsys, global_only, rank):
    # Execute the loader itself and perform real torch checkpoint I/O; only
    # isolate its optional torchvision/HF/color dependencies from CPU tests.
    source = Path(__file__).resolve().parents[2] / "tools/download.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "find_model")
    namespace = {"os": os, "torch": torch, "pretrained_models": {},
                 "colored": lambda text, **kwargs: text, "hf_download_or_fpath": lambda path: path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    path = tmp_path / "teacher.pt"
    torch.save({"value": torch.tensor([7])}, path)
    monkeypatch.setenv("SANA_LOG_GLOBAL_RANK_ONLY", "1" if global_only else "0")
    monkeypatch.setenv("RANK", str(rank))
    state = namespace["find_model"](str(path))
    assert state["value"].item() == 7
    output = capsys.readouterr().out
    assert bool(output) == (not global_only or rank == 0), "checkpoint loading must follow the same rank policy"


def test_slurm_launcher_contract_and_local_launcher_guard():
    root = Path(__file__).resolve().parents[2]
    script = root / "tools/ttn_slurm_train.sbatch"
    assert script.exists(), "Slurm sbatch entrypoint is missing"
    content = script.read_text()
    for token in ("--nodes=4", "--mem=128G", "--ntasks-per-node=1", "--gres=gpu:1", "--mpi=none", "--kill-on-bad-exit=1",
                  "SLURM_SUBMIT_DIR", "SLURM_JOB_ID", "distributed-check", "parallel=ddp", '--parallel "$parallel"'):
        assert token in content
    assert "CUDA_VISIBLE_DEVICES=" not in content
    assert "accelerate.commands.launch" not in content and "torchrun" not in content
    for name in ("ttn_l20_train.sh", "ttn_l20_smoke.sh"):
        local = (root / "tools" / name).read_text()
        assert local.index("SLURM_PROCID") < local.index("export CUDA_VISIBLE_DEVICES")


def _slurm_training_worker(rank, world, master, port, output, activation_offload):
    import contextlib
    import json
    from argparse import Namespace
    from unittest.mock import patch
    from test_training import TinyWorldModel
    from test_parallel import _rank_inputs
    from test_parallel_cli import CPUFlowConfig
    from worldttn import cli
    from worldttn.distributed import initialize, check_distributed
    from worldttn.training import train_clip, linear_flow_loss
    import torch.distributed as dist
    torch.set_num_threads(1)
    for name in list(os.environ):
        if name.startswith("SLURM_") or name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "NODE_RANK", "LOCAL_WORLD_SIZE"):
            del os.environ[name]
    os.environ.update(slurm_env(rank, world), MASTER_ADDR=master, MASTER_PORT=str(port), USE_LIBUV="0")
    mode, device = initialize("auto", "cpu", timeout_seconds=60)
    assert mode == "ddp" and device.type == "cpu"
    try:
        with patch("worldttn.distributed.socket.gethostname", return_value=f"ltu-hpc-{rank + 1}"):
            launch = check_distributed(device)
        captured = []
        def build(args):
            torch.manual_seed(17)
            model = TinyWorldModel("C")
            model.base_load_report = {"sha256": None}
            return model, Namespace(scheduler=CPUFlowConfig()), {"learning_rate": 1e-5, "tbptt": 2}
        def update(model, config, batch, optimizer, k, parallel):
            clean = batch["clean_latents"]
            noise = torch.randn_like(clean)
            result = train_clip(model, clean, batch["y"], batch["camera_conditions"], optimizer, linear_flow_loss,
                                torch.ones(1, 1, 13) * 500, noise, width=100, height=100, tbptt=k, parallel=parallel,
                                activation_offload=parallel.activation_offload)
            result["activation_offload"] = parallel.activation_offload
            captured.append({"state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                             "noise": noise, "loss": result["loss"], "norm": result["outer_grad_norm"],
                             "S": result["runtime"].world_state.detach().cpu(),
                             "psi": result["runtime"].transition_fast.detach().cpu(),
                             "commits": result["runtime"].commit_count})
            return result
        cli.build = build
        cli.SANAFlowLoss = lambda config: linear_flow_loss
        cli.train_update = update
        cli.timed_cuda = lambda call: (call(), {"seconds": 0., "peak_allocated_bytes": 0, "peak_reserved_bytes": 0})
        clean, noise, t, camera = _rank_inputs(rank)
        root = Path(output)
        torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera,
                    "width": 100, "height": 100}, root / f"batch-{rank}.pt")
        args = Namespace(parallel=mode, seed=3407, batch_file=str(root / "batch-{rank}.pt"), device="cpu",
                         adapter=None, resume=False, output=str(root / "resumed"), max_steps=1,
                         save_every=1, tbptt=2, launch=launch, activation_offload=activation_offload)
        with (root / f"stdout-{rank}.txt").open("w") as handle, contextlib.redirect_stdout(handle):
            cli.train_command(args)
            args.adapter, args.resume, args.max_steps = str(Path(args.output) / "last.pt"), True, 2
            cli.train_command(args)
            args.adapter, args.resume, args.output = None, False, str(root / "uninterrupted")
            cli.train_command(args)
        for name in captured[1]["state"]:
            torch.testing.assert_close(captured[1]["state"][name], captured[3]["state"][name], atol=0, rtol=0)
        torch.testing.assert_close(captured[1]["noise"], captured[3]["noise"], atol=0, rtol=0)
        torch.save({"first": captured[0], "launch": launch}, root / f"rank-{rank}.pt")
        if rank == 0:
            records = [json.loads(line) for line in (root / "resumed/train.jsonl").read_text().splitlines()]
            assert [row["step"] for row in records] == [1, 2]
            assert all(row["global_batch"] == world and len(row["ranks"]) == world for row in records)
            assert all(local["activation_offload"] == activation_offload for row in records for local in row["ranks"])
            payload = torch.load(root / "resumed/last.pt", weights_only=False)
            folder = root / "resumed" / payload["distributed"]["resume_dir"]
            assert len(list(folder.glob("rank-*.pt"))) == world
            assert payload["distributed"]["world_size"] == world
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,activation_offload", [(3, "none"), (4, "cpu")])
def test_slurm_ranks_initialize_train_save_resume_and_keep_state_local(tmp_path, world, activation_offload):
    import socket
    import json
    from test_training import TinyWorldModel
    from test_parallel import _init_uri, _rank_inputs
    from worldttn.training import train_clip, linear_flow_loss
    from worldttn.checkpoint import make_optimizer
    port = int(_init_uri().split(":")[-1].split("?")[0])
    torch.multiprocessing.spawn(_slurm_training_worker,
                                args=(world, socket.gethostname(), port, str(tmp_path), activation_offload), nprocs=world)
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=False) for rank in range(world)]
    torch.manual_seed(17)
    model = TinyWorldModel("C")
    batches = [_rank_inputs(rank) for rank in range(world)]
    clean = torch.cat([batch[0] for batch in batches])
    camera = torch.cat([batch[3] for batch in batches])
    noise = torch.cat([rank["first"]["noise"] for rank in ranks])
    result = train_clip(model, clean, torch.zeros(world, 1, 2, 8), camera, make_optimizer(model), linear_flow_loss,
                        torch.ones(world, 1, 13) * 500, noise, width=100, height=100, tbptt=2)
    checksum = world * (world + 1) // 2
    for rank in ranks:
        assert rank["launch"]["all_reduce_sum"] == checksum
        assert rank["launch"]["backend"] == "gloo"
        assert [row["local_rank"] for row in rank["launch"]["ranks"]] == [0] * world
        assert rank["first"]["commits"] == 5
        assert rank["first"]["norm"] == pytest.approx(result["outer_grad_norm"], rel=3e-5)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(rank["first"]["state"][name], value, atol=3e-7, rtol=3e-5)
    assert sum(rank["first"]["loss"] for rank in ranks) / world == pytest.approx(result["loss"], rel=3e-6)
    assert not torch.equal(ranks[0]["first"]["S"], ranks[1]["first"]["S"])
    assert not torch.equal(ranks[1]["first"]["psi"], ranks[2]["first"]["psi"])
    assert (tmp_path / "stdout-0.txt").read_text()
    assert all(not (tmp_path / f"stdout-{rank}.txt").read_text() for rank in range(1, world))
    run = json.loads((tmp_path / "resumed/run_config.json").read_text())
    assert run["launch"]["all_reduce_sum"] == checksum


def test_check_reports_actual_python_and_cache_environment(monkeypatch):
    import sys
    from worldttn import distributed
    install_environment(monkeypatch, {})
    monkeypatch.setenv("ROOT", "/shared/personal")
    monkeypatch.setenv("HF_HOME", "/shared/personal/.cache/huggingface")
    monkeypatch.setenv("TMPDIR", "/tmp/worldttn-123-rank0.example/tmp")
    monkeypatch.setenv("TORCHINDUCTOR_COMPILE_THREADS", "1")
    monkeypatch.setenv("GDN_DISABLE_COMPILE", "1")
    monkeypatch.setenv("GDN_DISABLE_COMPLEX_COMPILE", "0")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    report = distributed.check_distributed("cpu")
    record = report["ranks"][0]
    assert record["python"] == sys.executable
    assert record["python_version"] == sys.version.split()[0]
    assert record["torch"] == str(torch.__version__)
    assert record["cache"]["ROOT"] == "/shared/personal"
    assert record["cache"]["HF_HOME"] == "/shared/personal/.cache/huggingface"
    assert record["cache"]["TMPDIR"] == "/tmp/worldttn-123-rank0.example/tmp"
    assert record["compile_threads"] == "1"
    assert record["compile"] == {"gdn_disable_compile": "1", "gdn_disable_complex_compile": "0",
                                 "cuda_launch_blocking": "1"}


@pytest.mark.parametrize("command", ["distributed-check", "distributed-smoke", "train", "diagnose-update", "diagnose-single"])
@pytest.mark.parametrize("custom_root", [False, True])
@pytest.mark.parametrize("python_mode", ["valid", "missing", "base"])
def test_sbatch_executes_one_srun_with_shared_master_and_preserves_gpu_mask(tmp_path, command, custom_root, python_mode):
    import json
    import shutil
    import subprocess
    import sys
    tasks = 1 if command == "diagnose-single" else 3
    command = "diagnose-update" if command == "diagnose-single" else command
    bash = Path("D:/Git/bin/bash.exe")
    if not bash.exists():
        candidate = shutil.which("bash")
        if not candidate: pytest.skip("Bash not installed")
        bash = Path(candidate)
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "shared parent" / "WorldTTN"
    (project / "worldttn").mkdir(parents=True)
    (project / "worldttn/cli.py").touch()
    (project / "tools").mkdir()
    shutil.copyfile(root / "tools/ttn_cache_env.sh", project / "tools/ttn_cache_env.sh")
    shutil.copyfile(root / "tools/ttn_slurm_worker.sh", project / "tools/ttn_slurm_worker.sh")
    personal_root = tmp_path / "personal shared root" if custom_root else project.parent
    python = personal_root / "envs/worldttn/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/usr/bin/env bash\n[[ \"$1\" == -c ]] || exit 99\n"
                      "if [[ \"$3\" == ltu-hpc-1 ]]; then printf '10.0.0.1\\n'; else printf '10.0.0.2\\n'; fi\n",
                      encoding="utf-8", newline="\n")
    python.chmod(0o755)
    capture = tmp_path / "capture.json"
    cache_names = ("XDG_CACHE_HOME", "PIP_CACHE_DIR", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                   "HF_DATASETS_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE", "TRANSFORMERS_CACHE",
                   "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR",
                   "CUDA_CACHE_PATH")
    writer = tmp_path / "capture.py"
    writer.write_text("import json, os, sys\nfrom pathlib import Path\n"
                      "Path(os.environ['CAPTURE']).write_text(json.dumps({'args': sys.argv[1:], 'env': "
                      "{k: os.environ.get(k) for k in ('MASTER_ADDR', 'MASTER_PORT', 'CUDA_VISIBLE_DEVICES', "
                      f"'NCCL_SOCKET_IFNAME', 'NCCL_SOCKET_FAMILY', 'OUTPUT', 'ROOT', 'PYTHON', {', '.join(repr(k) for k in cache_names)})}}}}))\n")
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    for name, content in {
        "scontrol": "#!/usr/bin/env bash\nif [[ \"$3\" == ltu-hpc-2 ]]; then printf 'ltu-hpc-2\\n'; else printf 'ltu-hpc-1\\nltu-hpc-2\\nltu-hpc-3\\n'; fi\n",
        "srun": '#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$CAPTURE_WRITER" "$@"\n'
    }.items():
        stub = mock_bin / name
        stub.write_text(content, encoding="utf-8", newline="\n")
        stub.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SLURM_") and key not in ("MASTER_ADDR", "MASTER_PORT", "DATASET_ROOT", "BATCH_FILE",
                                                         "ADAPTER", "RESUME", "OUTPUT", "PYTHON", "CONFIG", "BASE_WEIGHTS",
                                                         "DISTRIBUTED_TIMEOUT", "NCCL_SOCKET_FAMILY", "ROOT")}
    env.update({name: "/home/ad/z2zhang/stale-cache" for name in cache_names})
    if custom_root: env["ROOT"] = personal_root.as_posix()
    env.update(SLURM_JOB_ID="12345", SLURM_NTASKS=str(tasks), SLURM_JOB_NODELIST="ltu-hpc-[1-3]",
               SLURM_SUBMIT_DIR=project.as_posix(), PROJECT_ROOT=project.as_posix(), COMMAND=command,
               SLURM_STEP_NODELIST="ltu-hpc-2", SLURM_STEP_NUM_NODES="1",
               REAL_PYTHON=Path(sys.executable).as_posix(),
               CAPTURE=capture.as_posix(), CAPTURE_WRITER=writer.as_posix(),
               MOCK_BIN=mock_bin.as_posix(),
               SCRIPT=(root / "tools/ttn_slurm_train.sbatch").as_posix(),
               CUDA_VISIBLE_DEVICES="GPU-slurm-mask", NCCL_SOCKET_IFNAME="=eth-test", STAGE="C")
    env["PYTHON_MODE"] = python_mode
    if python_mode == "base": env["PYTHON"] = "/data/group/zhaolab/project/miniconda/bin/python"
    if command == "train":
        env.update(DATASET_ROOT=(tmp_path / "shared data").as_posix(), ADAPTER=(tmp_path / "last.pt").as_posix(), RESUME="1")
        env.update(TRAIN_SCOPE="dit" if custom_root else "ttn", BACKBONE_LR="2e-6")
    env.pop("CROSS_ATTN_BACKEND", None)
    env.pop("DIAGNOSTIC_UNMASK_ALL_VALID", None)
    if command == "diagnose-update":
        env["CUDA_TRACE"] = "1"
        env["CROSS_ATTN_BACKEND"] = "flash" if custom_root else "math"
        env["DIAGNOSTIC_UNMASK_ALL_VALID"] = "1" if custom_root else "0"
    shell = ('export MSYS_NO_PATHCONV=1; export MSYS2_ENV_CONV_EXCL="*"; '
             'export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; '
             'if [[ "$PYTHON_MODE" == valid ]]; then '
             'export PYTHON="${ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)}/envs/worldttn/bin/python"; fi; bash "$SCRIPT"')
    result = subprocess.run([str(bash), "-c", shell], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if python_mode != "valid":
        assert result.returncode != 0
        assert "PYTHON" in result.stderr and "envs/worldttn/bin/python" in result.stderr
        assert not capture.exists(), "invalid environment must fail before srun"
        return
    assert result.returncode == 0, result.stderr
    record = json.loads(capture.read_text())
    args, exported = record["args"], record["env"]
    assert f"--ntasks={tasks}" in args and "--ntasks-per-node=1" in args and "--gres=gpu:1" in args
    assert "--mpi=none" in args and "--kill-on-bad-exit=1" in args
    worker_index = args.index("bash") + 1
    assert args[worker_index].endswith("/tools/ttn_slurm_worker.sh")
    assert args[worker_index + 1:worker_index + 3] == [command, "--parallel"]
    assert args[args.index("--parallel") + 1] == ("single" if tasks == 1 else "ddp")
    if command in ("train", "distributed-smoke", "diagnose-update"):
        assert args[args.index("--activation-offload") + 1] == "cpu" and "--memory-trace" in args
        assert args[args.index("--cross-attn-backend") + 1] == env.get("CROSS_ATTN_BACKEND", "auto")
    else:
        assert "--activation-offload" not in args and "--memory-trace" not in args
    if command == "train":
        assert args[args.index("--train-scope") + 1] == env["TRAIN_SCOPE"]
        assert args[args.index("--backbone-lr") + 1] == "2e-6"
    else:
        assert "--train-scope" not in args and "--backbone-lr" not in args
    if command == "diagnose-update":
        assert "--cuda-trace" in args and args[args.index("--frames") + 1] == "13"
        assert args[args.index("--tbptt") + 1] == "1" and args[args.index("--stage") + 1] == "C"
    assert ("--diagnostic-unmask-all-valid" in args) == (command == "diagnose-update" and custom_root)
    assert exported["MASTER_ADDR"] == "10.0.0.1" and exported["MASTER_PORT"] == "27345"
    assert "master_node=ltu-hpc-1" in result.stdout
    assert "--distribution=block" in args
    assert exported["NCCL_SOCKET_FAMILY"] == "AF_INET"
    assert args[args.index("--distributed-timeout") + 1] == ("120" if command == "distributed-check" else "600")
    assert exported["CUDA_VISIBLE_DEVICES"] == "GPU-slurm-mask" and exported["NCCL_SOCKET_IFNAME"] == "=eth-test"
    cache_root = exported["ROOT"]
    assert cache_root is not None
    if custom_root: assert cache_root == env["ROOT"]
    else: assert cache_root.endswith("/" + project.parent.name)
    assert exported["PYTHON"] == cache_root + "/envs/worldttn/bin/python"
    suffixes = ("", "/pip", "/huggingface", "/huggingface/hub", "/huggingface/hub",
                "/huggingface/datasets", "/huggingface/xet", "/huggingface/assets", "/huggingface/hub",
                "/torch", "/torch/extensions", "/torch/inductor", "/triton", "/cuda")
    for name, suffix in zip(cache_names, suffixes):
        assert exported[name] == cache_root + "/.cache" + suffix, name
    assert "[TTN cache]" in result.stdout
    assert exported["OUTPUT"].endswith("slurm-12345-C")
    assert ("--resume" in args) == (command == "train")
    assert ("--dataset-root" in args) == (command == "train")
    if command == "distributed-smoke": assert args[args.index("--frames") + 1] == "13"
