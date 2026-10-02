"""Epoch-aware rank sampling and deterministic clip crops for exact cursor resume."""
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Sampler


class SeededClipDataset(Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self): return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        seed = (self.seed + epoch * 1000003 + index * 9176) % (2**32)
        py_state, torch_state = random.getstate(), torch.get_rng_state()
        try:
            import numpy as np
            np_state = np.random.get_state()
        except ImportError:
            np = None
        try:
            random.seed(seed)
            # Seed CPU only; torch.manual_seed would also change the trainer's
            # CUDA noise RNG when num_workers=0.
            torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            if np is not None: np.random.seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(py_state)
            torch.set_rng_state(torch_state)
            if np is not None: np.random.set_state(np_state)


class EpochSampler(Sampler):
    def __init__(self, sampler):
        self.sampler, self.epoch = sampler, 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        return iter((self.epoch, index) for index in self.sampler)

    def __len__(self): return len(self.sampler)


class ResumableBatchStream:
    def __init__(self, loader, *, seed, rank=0, world=1, state=None):
        self.source_loader = loader
        self.seed, self.rank, self.world = seed, rank, world
        source = loader.sampler
        shuffle = source.shuffle if isinstance(source, DistributedSampler) else isinstance(
            source, torch.utils.data.RandomSampler)
        source = DistributedSampler(loader.dataset, world, rank, shuffle=shuffle, seed=seed, drop_last=True)
        self.sampler = EpochSampler(source)
        self.generator = torch.Generator()
        kw = {}
        if loader.num_workers > 0:
            kw["prefetch_factor"] = loader.prefetch_factor
        self.loader = DataLoader(SeededClipDataset(loader.dataset, seed), batch_size=loader.batch_size,
                                 sampler=self.sampler, num_workers=loader.num_workers,
                                 pin_memory=loader.pin_memory, persistent_workers=loader.persistent_workers,
                                 drop_last=loader.drop_last, collate_fn=loader.collate_fn,
                                 generator=self.generator, **kw)
        if len(self.loader) == 0: raise ValueError("training dataset has no complete per-rank batches")
        self.epoch, self.batch_in_epoch = 0, 0
        self.iterator = None
        if state:
            for name in ("seed", "rank", "world", "dataset_size", "batches_per_epoch"):
                if state.get(name) != self.state_dict()[name]:
                    raise ValueError(f"resume data stream {name} mismatch")
            self.epoch, self.batch_in_epoch = state["epoch"], state["batch_in_epoch"]
            if self.epoch < 0 or not 0 <= self.batch_in_epoch <= len(self.loader):
                raise ValueError("invalid resume data cursor")

    def __iter__(self): return self

    def __next__(self):
        if self.batch_in_epoch == len(self.loader):
            self.epoch += 1
            self.batch_in_epoch = 0
            self.iterator = None
        if self.iterator is None:
            self.sampler.set_epoch(self.epoch)
            self.generator.manual_seed(self.seed + self.rank + self.epoch * 1000003)
            self.iterator = iter(self.loader)
            for _ in range(self.batch_in_epoch): next(self.iterator)
        result = next(self.iterator)
        self.batch_in_epoch += 1
        return result

    def state_dict(self):
        return {"epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch, "seed": self.seed,
                "rank": self.rank, "world": self.world, "dataset_size": len(self.source_loader.dataset),
                "batches_per_epoch": len(self.loader)}
