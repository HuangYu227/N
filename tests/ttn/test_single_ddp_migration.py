"""Keep Adam moments when moving a complete single-card bundle to DDP."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import distributed as dist

from test_training import TinyWorldModel
from worldttn.checkpoint import load_checkpoint, make_optimizer
from worldttn.parallel_checkpoint import restore_training_checkpoint, save_training_checkpoint


def _migration_worker(rank, uri, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=2)
    try:
        model = TinyWorldModel("C")
        source = Path(directory) / "single.pt"
        load_checkpoint(source, model)
        optimizer = make_optimizer(model)
        parallel = SimpleNamespace(model=model, rank=rank, world=2, mode="ddp", reshard=lambda: None)
        step, cursor = restore_training_checkpoint(source, parallel, optimizer,
            {"tbptt": 0}, migrate_single_to_ddp=True)
        assert step == 325 and cursor["batch_in_epoch"] == 163 and cursor["rank"] == rank
        source_shard = next(Path(directory).glob("single-resume-*/rank-00000.pt"))
        expected = torch.load(source_shard, weights_only=False)["optimizer"]
        actual = optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        assert actual["state"].keys() == expected["state"].keys()
        for key, values in expected["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(actual["state"][key][name], value, rtol=0, atol=0)
        save_training_checkpoint(Path(directory) / "dual.pt", parallel, optimizer,
            step, cursor, {"tbptt": 0})
        assert restore_training_checkpoint(Path(directory) / "dual.pt", parallel,
            make_optimizer(model), {"tbptt": 0}) == (step, cursor)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo process group required")
def test_single_adam_migrates_to_both_ddp_ranks_and_resaves(tmp_path):
    torch.set_num_threads(1)
    model = TinyWorldModel("C")
    optimizer = make_optimizer(model)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parallel = SimpleNamespace(model=model, rank=0, world=1, mode="single", reshard=lambda: None)
    save_training_checkpoint(tmp_path / "single.pt", parallel, optimizer, 325,
        {"epoch": 0, "batch_in_epoch": 325, "seed": 3407, "rank": 0,
         "world": 1, "dataset_size": 11488, "batches_per_epoch": 11488}, {"tbptt": 0})
    torch.multiprocessing.spawn(_migration_worker,
        args=((tmp_path / "gloo-store").as_uri(), str(tmp_path)), nprocs=2, join=True)
