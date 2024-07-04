import random
from collections import deque
from typing import Any, Collection, Deque, Iterable, Iterator, List, Sequence

Loader = Iterable[Any]


def _pooled_next(iterator: Iterator[Any], pool: Deque[Any], N):
    if not pool:
        pool.extend(next(iterator))
    ret = []
    for i in range(N):
        ret.append(pool.popleft())
    return ret

class CombinedDataLoader:
    """
    Combines data loaders using the provided sampling ratios
    """

    BATCH_COUNT = 100

    def __init__(self, loaders: Collection[Loader], batch_size: int, ratios: Sequence[float]):
        self.loaders = loaders
        self.batch_size = batch_size
        self.ratios = ratios

    def __iter__(self) -> Iterator[List[Any]]:
        iters = [iter(loader) for loader in self.loaders]
        indices = []
        pool = [deque()] * len(iters)
        # infinite iterator, as in D2
        while True:
            if not indices:
                # just a buffer of indices, its size doesn't matter
                # as long as it's a multiple of batch_size
                k = self.batch_size * self.BATCH_COUNT
                indices = random.choices(range(len(self.loaders)), self.ratios, k=k)
            try:
                idx = indices[0]
                batch = _pooled_next(iters[idx], pool[idx], self.batch_size)
            except StopIteration:
                break
            indices = indices[1:]
            yield batch
