from __future__ import annotations

import math

import torch

from .batch import Batch

SIGMA_EPS = 2e-2


def _uniform(low, high, shape, device, generator):
    return low + (high - low) * torch.rand(shape, device=device, generator=generator)


def kernel_matrix(kind: str, x: torch.Tensor, generator=None) -> torch.Tensor:
    """Matriz de covarianza [B, N, N] con hiperparámetros aleatorios por función."""
    B, N, _ = x.shape
    dev = x.device
    if kind in ("rbf", "matern"):
        length = _uniform(0.1, 0.6, (B, 1, 1), dev, generator)
        scale = _uniform(0.1, 1.0, (B, 1, 1), dev, generator)
        d = torch.cdist(x, x) / length
        if kind == "rbf":
            k = torch.exp(-0.5 * d**2)
        else:
            s5 = math.sqrt(5.0)
            k = (1 + s5 * d + 5.0 * d**2 / 3.0) * torch.exp(-s5 * d)
    elif kind == "periodic":
        period = _uniform(0.1, 0.5, (B, 1, 1), dev, generator)
        length = _uniform(0.1, 0.6, (B, 1, 1), dev, generator)
        scale = _uniform(0.1, 1.0, (B, 1, 1), dev, generator)
        d = torch.cdist(x, x, p=1)
        k = torch.exp(-2 * (torch.sin(math.pi * d / period) / length) ** 2)
    else:
        raise ValueError(f"núcleo desconocido: {kind}")
    return scale**2 * k + SIGMA_EPS**2 * torch.eye(N, device=dev)


class GPSampler:
    """Generador de lotes de tareas a partir de un proceso gaussiano."""

    def __init__(self, kind: str = "rbf", max_num_points: int = 50, x_range=(-2.0, 2.0)):
        self.kind = kind
        self.max_num_points = max_num_points
        self.x_range = x_range

    def sample_sizes(self, generator=None) -> tuple[int, int]:
        n = self.max_num_points
        num_ctx = int(torch.randint(3, n - 3, (1,), generator=generator))
        num_tar = int(torch.randint(3, n - num_ctx, (1,), generator=generator))
        return num_ctx, num_tar

    def sample(self, batch_size=16, num_ctx=None, num_tar=None, device="cpu", generator=None) -> Batch:
        if num_ctx is None or num_tar is None:
            c, t = self.sample_sizes(generator)
            num_ctx = c if num_ctx is None else num_ctx
            num_tar = t if num_tar is None else num_tar
        n = num_ctx + num_tar
        lo, hi = self.x_range
        x = _uniform(lo, hi, (batch_size, n, 1), device, generator)
        y = self.sample_functions(x, generator)
        return Batch(x[:, :num_ctx], y[:, :num_ctx], x[:, num_ctx:], y[:, num_ctx:])

    def sample_functions(self, x: torch.Tensor, generator=None) -> torch.Tensor:
        """Valores de funciones nuevas en las posiciones ``x`` [B, N, 1] -> [B, N, 1]."""
        cov = kernel_matrix(self.kind, x, generator)
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(x.shape[0], x.shape[1], 1, device=x.device, generator=generator)
        return L @ eps
