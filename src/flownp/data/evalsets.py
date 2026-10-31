from __future__ import annotations

import torch

from .batch import Batch


def load_official(path: str) -> list[Batch]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return [Batch.from_mapping(b) for b in raw]


def make_evalset(sampler, num_batches=100, batch_size=128, seed=0, **kwargs) -> list[Batch]:
    g = torch.Generator().manual_seed(seed)
    return [sampler.sample(batch_size=batch_size, generator=g, **kwargs) for _ in range(num_batches)]


def save(batches: list[Batch], path: str) -> None:
    torch.save([dict(xc=b.xc, yc=b.yc, xt=b.xt, yt=b.yt) for b in batches], path)


def load(path: str) -> list[Batch]:
    return [Batch.from_mapping(b) for b in torch.load(path, map_location="cpu", weights_only=True)]
