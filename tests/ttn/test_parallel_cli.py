import subprocess
import sys
from pathlib import Path
import json
import torch
import pytest
from dataclasses import dataclass


@dataclass
class CPUFlowConfig:
    noise_schedule: str = "linear_flow"


@dataclass
class CPUModelConfig:
    chunk_size: int = 3


@dataclass
class CPUTextConfig:
    text_encoder_name: str = "gemma"


@dataclass
class CPUTimestepConfig:
    noise_multiplier: float = 0.
    chunk_sampling_strategy: str = "incremental"


@pytest.mark.parametrize("field", ["timestep", "text", "model"])
def test_resume_identity_includes_noise_text_and_base_model_configuration(field):
    from argparse import Namespace
    from worldttn.cli import _training_identity
    args = Namespace(seed=3407, batch_file="batch-{rank}.pt", text_encoder_device="cpu")
    config = Namespace(task="df", scheduler=CPUFlowConfig(), model=CPUModelConfig(),
                       text_encoder=CPUTextConfig(), train=CPUTimestepConfig())
    before = _training_identity(args, config, {}, 2)
    if field == "timestep": config.train.noise_multiplier = .2
    elif field == "text": config.text_encoder.text_encoder_name = "changed"
    else: config.model.chunk_size = 4
    assert before != _training_identity(args, config, {}, 2), "resume must reject a changed training problem"


def _cli_worker(rank, size, init_uri, output, mode):
    from argparse import Namespace
    from test_training import TinyWorldModel
    from test_parallel import _rank_inputs
    from worldttn.training import train_clip, linear_flow_loss
    from worldttn import cli
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_uri, rank=rank, world_size=size)
    try:
        def build(args):
            torch.manual_seed(17)
            model = TinyWorldModel("C")
            model.base_load_report = {"sha256": None}
            return model, Namespace(scheduler=CPUFlowConfig()), {"learning_rate": 1e-5, "tbptt": 2}

        def update(model, config, batch, optimizer, k, parallel):
            clean = batch["clean_latents"]
            return train_clip(model, clean, batch["y"], batch["camera_conditions"], optimizer, linear_flow_loss,
                              torch.ones(1, 1, 13) * 500, torch.randn_like(clean), width=100, height=100,
                              tbptt=k, parallel=parallel)

        # Only replace the CUDA teacher builder/loss/sampler/timer. CLI,
        # distributed engine, train loop, logging and checkpoint paths run as-is.
        cli.build = build
        cli.SANAFlowLoss = lambda config: linear_flow_loss
        cli.train_update = update
        cli.timed_cuda = lambda call: (call(), {"seconds": 0., "peak_allocated_bytes": 0,
                                               "peak_reserved_bytes": 0})
        clean, noise, t, camera = _rank_inputs(rank)
        torch.save({"clean_latents": clean, "y": torch.zeros(1, 1, 2, 8), "camera_conditions": camera,
                    "width": 100, "height": 100}, Path(output) / f"batch-{rank}.pt")
        args = Namespace(parallel=mode, seed=3407, batch_file=str(Path(output) / "batch-{rank}.pt"),
                         device="cpu", adapter=None, resume=False, output=str(Path(output) / "train"),
                         max_steps=1, save_every=1, tbptt=2)
        cli.train_command(args)
        args.adapter = str(Path(args.output) / "last.pt")
        args.resume, args.max_steps = True, 2
        cli.train_command(args)
        if rank == 0:
            records = [json.loads(line) for line in (Path(args.output) / "train.jsonl").read_text().splitlines()]
            assert [record["step"] for record in records] == [1, 2]
            assert all(record["global_batch"] == 2 and len(record["ranks"]) == 2 for record in records)
            assert all(r["commits"] == 5 for record in records for r in record["ranks"])
            assert torch.load(args.adapter, weights_only=False)["step"] == 2
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ddp", "fsdp2"])
def test_cli_rank_bundles_train_log_export_and_resume(tmp_path, mode):
    from test_parallel import _init_uri
    if mode == "fsdp2" and not hasattr(torch.cpu, "Stream"): pytest.skip("CPU FSDP2 not available")
    torch.multiprocessing.spawn(_cli_worker, args=(2, _init_uri(), str(tmp_path), mode), nprocs=2)


def test_dataset_root_resolves_raw_and_cache_paths_without_cuda_imports(tmp_path):
    from argparse import Namespace
    from worldttn.sana import resolve_data_paths
    config = Namespace(data=Namespace(data_dir={"sekai_game": "data/raw"}, vae_cache_dir="data/cache"))
    resolve_data_paths(config, tmp_path)
    assert config.data.data_dir == {"sekai_game": str(tmp_path / "data/raw")}
    assert config.data.vae_cache_dir == str(tmp_path / "data/cache")


def test_parallel_cli_help_does_not_import_cuda_sana():
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ("--parallel", "--dataset-root", "--text-encoder-device", "distributed-smoke",
                   "distributed-check", "--distributed-timeout"):
        assert option in result.stdout


def test_resume_without_adapter_fails_before_cuda_or_teacher_initialization():
    result = subprocess.run([sys.executable, "-m", "worldttn.cli", "train", "--resume"],
                             capture_output=True, text=True)
    assert result.returncode == 2
    assert "--resume requires --adapter" in result.stderr
    assert "CUDA" not in result.stderr and "AttributeError" not in result.stderr


def test_l20_launcher_selects_two_processes_without_accelerate_double_wrapping():
    root = Path(__file__).resolve().parents[2]
    config = root / "configs/worldttn/l20_2gpu.yaml"
    script = root / "tools/ttn_l20_train.sh"
    assert config.exists() and script.exists(), "two-L20 launch configuration is missing"
    assert "distributed_type: MULTI_GPU" in config.read_text()
    assert "num_processes: 2" in config.read_text()
    content = script.read_text()
    assert "--parallel" in content and "fsdp2" in content
    assert "accelerate.commands.launch" in content
