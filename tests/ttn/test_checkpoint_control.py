"""Checkpoint disk work must not enqueue a waiting NCCL collective."""
from datetime import timedelta
import os
from types import SimpleNamespace
import time

import pytest
import torch
from worldttn import parallel_checkpoint as saving


def test_checkpoint_phase_uses_cached_cpu_group(tmp_path, monkeypatch):
    group = object()
    engine = SimpleNamespace(rank=0, world=2, checkpoint_cpu_group=group)
    def gather(rows, result, *, group=None):
        assert group is engine.checkpoint_cpu_group, "CPU IO must not wait through the NCCL group"
        rows[:] = [result, {"rank": 1, "value": None, "error": None}]
    monkeypatch.setattr(saving.dist, "all_gather_object", gather)
    assert saving._phase(engine, tmp_path/'last.pt', "model-publish", lambda: "verified", 20) == ["verified", None]


def test_checkpoint_control_group_is_bounded_and_reused(tmp_path, monkeypatch):
    engine = SimpleNamespace(rank=0, world=2)
    calls = []
    group = object()
    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs["backend"] == "gloo"
        assert kwargs["timeout"] == timedelta(seconds=3600)
        return group
    def gather(rows, result, *, group=None):
        assert group is engine.checkpoint_cpu_group
        rows[:] = [result, {"rank": 1, "value": None, "error": None}]
    monkeypatch.setattr(saving.dist, "new_group", create)
    monkeypatch.setattr(saving.dist, "all_gather_object", gather)
    for _ in range(2): saving._phase(engine, tmp_path/'last.pt', "shard-write", lambda: 1, 20)
    assert len(calls) == 1


@pytest.mark.skipif(os.environ.get("CHECKPOINT_IO_CUDA_TEST") != "1",
                   reason="opt in to a torchrun NCCL/Gloo timing regression")
def test_slow_cpu_phase_survives_short_nccl_timeout(tmp_path):
    dist = saving.dist
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=2), device_id=device)
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        engine = SimpleNamespace(rank=rank, world=world)
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor)
        def slow_publish():
            if rank == 0:
                time.sleep(4)  # Longer than the data-plane timeout.
                return "verified"
        assert saving._phase(engine, tmp_path/'last.pt', "model-publish", slow_publish, 20)[0] == "verified"
        tensor.fill_(1)
        dist.all_reduce(tensor)
        assert tensor.item() == world
        def failed_io():
            if rank == world-1: raise OSError("injected disk failure")
        with pytest.raises(RuntimeError, match="injected disk failure"):
            saving._phase(engine, tmp_path/'last.pt', "shard-write", failed_io, 21)
        tensor.fill_(1)
        dist.all_reduce(tensor)
        assert tensor.item() == world
    finally:
        dist.destroy_process_group()
