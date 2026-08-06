import math

import torch
from torch.utils.data import Sampler


class CurriculumDistributedSampler(Sampler):
    """Distributed sampler that re-reads `len(dataset)` each epoch.

    `SnapshotPairDataset.__len__` returns `_active_len`, which changes when
    `set_curriculum_fraction(...)` is called. A vanilla `DistributedSampler`
    caches `num_samples`/`total_size` at construction and would not see the
    curriculum change. This sampler recomputes them inside `__iter__` /
    `__len__`.

    Padding (`drop_last=False`) duplicates a few indices so each rank
    receives the same number of samples. Use `set_epoch(epoch)` to vary
    the shuffle across epochs.
    """

    def __init__(
        self,
        dataset,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas={num_replicas}")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _sizes(self):
        n = len(self.dataset)

        if self.drop_last and n % self.num_replicas != 0:
            num_samples = math.floor(n / self.num_replicas)
        else:
            num_samples = math.ceil(n / self.num_replicas)

        total_size = num_samples * self.num_replicas
        return n, num_samples, total_size

    def __iter__(self):
        n, num_samples, total_size = self._sizes()

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if not self.drop_last:
            padding_size = total_size - len(indices)
            if padding_size > 0:
                if padding_size <= len(indices):
                    indices += indices[:padding_size]
                else:
                    indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:total_size]

        assert len(indices) == total_size

        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples

        return iter(indices)

    def __len__(self):
        _, num_samples, _ = self._sizes()
        return num_samples
