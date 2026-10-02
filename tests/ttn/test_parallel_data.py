import random
import torch
import pytest
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class RandomClipDataset(Dataset):
    def __len__(self): return 8
    def __getitem__(self, index): return index, random.random(), torch.rand(1)


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
