from pathlib import Path
from datetime import timedelta
import torch
from test_parallel import _init_uri


def _worker(rank, uri, output):
    import torch.distributed as dist
    from types import SimpleNamespace
    from test_training import TinyWorldModel
    from worldttn.checkpoint import make_optimizer
    from worldttn import parallel_checkpoint as saving
    from worldttn.checkpoint_integrity import audit_checkpoint
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=uri, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(17)
        model = TinyWorldModel("C")
        engine = SimpleNamespace(model=model, rank=rank, world=2, mode="ddp", reshard=lambda: None)
        path = Path(output) / "last.pt"
        saving.save_training_checkpoint(path, engine, make_optimizer(model), 1)
        original = saving.atomic_save
        def fail(payload, target):
            if rank == 1 and target.name == "rank-00001.pt": raise OSError("injected rank1 disk quota")
            original(payload, target)
        saving.atomic_save = fail
        try:
            saving.save_training_checkpoint(path, engine, make_optimizer(model), 2)
            raise AssertionError("failure not propagated")
        except RuntimeError as error:
            assert "rank 1" in str(error) and "disk quota" in str(error)
        assert audit_checkpoint(path)["step"] == 1
    finally:
        dist.destroy_process_group()


def test_rank_io_failure_propagates_to_peers_without_publishing_or_hanging(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(_init_uri(), str(tmp_path)), nprocs=2)
    assert not (tmp_path / ".checkpoint-save.lock").exists()
