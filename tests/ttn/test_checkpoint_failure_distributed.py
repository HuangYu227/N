from pathlib import Path
from datetime import timedelta
import torch
import pytest
from test_parallel import _init_uri


def _worker(rank, uri, output, failure="shard"):
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
        if failure == "shard":
            original = saving.atomic_save
            def fail(payload, target):
                if rank == 1 and target.name == "rank-00001.pt": raise OSError("injected rank1 disk quota")
                original(payload, target)
            saving.atomic_save = fail
        else:
            original = saving.offline_state_dict
            def broken_state(model):
                state = original(model)
                first = next(iter(state))
                tensor = state[first]
                class Gathered:
                    def cpu(self):
                        if failure == "copy" and rank == 0: raise OSError("injected rank0 CPU copy")
                        return tensor
                class Shard:
                    shape, dtype = tensor.shape, tensor.dtype
                    def numel(self): return tensor.numel()
                    def element_size(self): return tensor.element_size()
                    def detach(self):
                        if failure == "prepare" and rank == 1: raise OSError("injected rank1 gather preparation")
                        return self
                    def full_tensor(self):
                        dist.barrier()  # A real collective before the injected local copy failure.
                        return Gathered()
                state = dict(state); state[first] = Shard()
                return state
            saving.offline_state_dict = broken_state
        try:
            saving.save_training_checkpoint(path, engine, make_optimizer(model), 2)
            raise AssertionError("failure not propagated")
        except RuntimeError as error:
            assert "rank " in str(error) and "injected" in str(error)
        assert audit_checkpoint(path)["step"] == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("failure", ["shard", "prepare", "copy"])
def test_rank_io_failure_propagates_to_peers_without_publishing_or_hanging(tmp_path, failure):
    torch.multiprocessing.spawn(_worker, args=(_init_uri(), str(tmp_path), failure), nprocs=2)
    assert not (tmp_path / ".checkpoint-save.lock").exists()
