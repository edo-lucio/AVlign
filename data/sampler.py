"""Class-stratified batch sampler.

Each batch contains `classes_per_batch` distinct classes with `per_class`
samples each. Critical for FGW: without this, intra-batch C_v / C_a degenerate.
"""
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler


class ClassStratifiedSampler(Sampler[List[int]]):
    def __init__(self, dataset, classes_per_batch: int = 8, per_class: int = 8,
                 num_batches: int | None = None, seed: int = 0):
        self.classes_per_batch = classes_per_batch
        self.per_class = per_class
        self.batch_size = classes_per_batch * per_class
        self.rng = random.Random(seed)

        # Group dataset indices by class.
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            cls = dataset[i]["class_id"]
            self.by_class[cls].append(i)
        self.classes = [c for c, idxs in self.by_class.items() if len(idxs) >= per_class]
        if len(self.classes) < classes_per_batch:
            raise ValueError(
                f"only {len(self.classes)} classes have >= {per_class} samples; "
                f"need >= {classes_per_batch}")

        if num_batches is None:
            num_batches = sum(len(v) for v in self.by_class.values()) // self.batch_size
        self.num_batches = num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            chosen = self.rng.sample(self.classes, self.classes_per_batch)
            batch: list[int] = []
            for c in chosen:
                batch.extend(self.rng.sample(self.by_class[c], self.per_class))
            yield batch
