import random
import torch
import pytest
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class RandomClipDataset(Dataset):
    def __len__(self): return 8
    def __getitem__(self, index): return index, random.random(), torch.rand(1)


def test_single_to_two_rank_cursor_skips_only_unpaired_source_sample():
    from worldttn.parallel_data import migrate_single_cursor_to_two_ranks, ResumableBatchStream
    source = {"epoch": 0, "batch_in_epoch": 5, "seed": 42, "rank": 0,
              "world": 1, "dataset_size": 8, "batches_per_epoch": 8}
    old = DataLoader(RandomClipDataset(), batch_size=1)
    old_stream = ResumableBatchStream(old, seed=42, state={**source, "batch_in_epoch": 0})
    seen = [int(next(old_stream)[0]) for _ in range(8)]
    migrated = []
    for rank in (0, 1):
        cursor, report = migrate_single_cursor_to_two_ranks(source, rank)
        assert report["skipped_source_samples"] == 1
        loader = DataLoader(RandomClipDataset(), batch_size=1)
        stream = ResumableBatchStream(loader, seed=42, rank=rank, world=2, state=cursor)
        migrated.append(int(next(stream)[0]))
    assert migrated == seen[6:8]
    with pytest.raises(ValueError):
        migrate_single_cursor_to_two_ranks({**source, "dataset_size": 7}, 0)


def test_single_to_ddp_resume_requires_explicit_flag_and_same_training_identity():
    from worldttn.parallel_checkpoint import validate_training_checkpoint
    payload = {"distributed": {"format": "TTN-parallel-resume-v1", "mode": "single",
        "world_size": 1, "resume_dir": "last-resume-00000325-abcd1234",
        "training_config": {"tbptt": 0, "execution": {"core_backend": "reference"}}}}
    identity = payload["distributed"]["training_config"]
    with pytest.raises(ValueError, match="same backend"):
        validate_training_checkpoint(payload, "ddp", 2, identity)
    assert validate_training_checkpoint(payload, "ddp", 2, identity,
        migrate_single_to_ddp=True)["single_to_ddp"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        validate_training_checkpoint(payload, "ddp", 2, {**identity, "tbptt": 4},
            migrate_single_to_ddp=True)
    with pytest.raises(ValueError, match="same backend"):
        validate_training_checkpoint(payload, "fsdp2", 2, identity,
            migrate_single_to_ddp=True)


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("world", [2, 3])
def test_rank_sampler_and_resume_reproduce_crops_without_consuming_training_rng(workers, world):
    import importlib.util
    assert importlib.util.find_spec("worldttn.parallel_data"), "resumable rank data stream is missing"
    from worldttn.parallel_data import ResumableBatchStream
    streams = []
    for rank in range(world):
        dataset = RandomClipDataset()
        loader = DataLoader(dataset, batch_size=1, sampler=DistributedSampler(dataset, world, rank, seed=42),
                            num_workers=workers, persistent_workers=workers > 0)
        streams.append(ResumableBatchStream(loader, seed=42, rank=rank, world=world))
    first = [next(stream) for stream in streams]
    assert len({int(batch[0]) for batch in first}) == world
    cursor = streams[0].state_dict()
    future = [next(streams[0]) for _ in range(6)]
    replay = ResumableBatchStream(streams[0].source_loader, seed=42, rank=0, world=world, state=cursor)
    torch.manual_seed(19)
    random.seed(20)
    before = (torch.get_rng_state(), random.getstate())
    again = [next(replay) for _ in range(6)]
    assert torch.equal(torch.get_rng_state(), before[0]) and random.getstate() == before[1]
    for expected, actual in zip(future, again):
        for a, b in zip(expected, actual): torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert replay.state_dict() == streams[0].state_dict()
    with pytest.raises(ValueError, match="seed mismatch"):
        ResumableBatchStream(streams[0].source_loader, seed=43, rank=0, world=world, state=cursor)
