from __future__ import annotations

import torch


def coverage(samples: torch.Tensor, y: torch.Tensor, level: float) -> torch.Tensor:
    """``samples`` [S, ...], ``y`` [...] -> proporción de ``y`` dentro del intervalo central."""
    a = (1 - level) / 2
    q = torch.quantile(samples.float(), torch.tensor([a, 1 - a], device=samples.device), dim=0)
    return ((y >= q[0]) & (y <= q[1])).float().mean()


def crps(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Estimador sin sesgo de la puntuación de probabilidad ranqueada continua, media sobre puntos."""
    S = samples.shape[0]
    term1 = (samples - y[None]).abs().mean(0)
    s, _ = torch.sort(samples, dim=0)
    i = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype).view(-1, *([1] * (samples.dim() - 1)))
    term2 = (2 * (2 * i - S - 1) * s).sum(0) / (S * (S - 1))
    return (term1 - 0.5 * term2).mean()


def pit(samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (samples <= y[None]).float().mean(0).flatten()


def calibration_summary(samples: torch.Tensor, y: torch.Tensor) -> dict:
    p = pit(samples, y)
    hist = torch.histc(p, bins=10, min=0, max=1) / p.numel()
    return dict(
        coverage_50=coverage(samples, y, 0.5).item(),
        coverage_90=coverage(samples, y, 0.9).item(),
        crps=crps(samples, y).item(),
        pit_hist=hist.tolist(),
        pit_flatness=(hist - 0.1).abs().sum().item() / 2,
    )


def _knn_radius(X: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(X, X)
    return d.kthvalue(k + 1, dim=1).values


def precision_recall(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict:
    """``real`` [R, D], ``fake`` [F, D]."""
    rr, rf = _knn_radius(real, k), _knn_radius(fake, k)
    d = torch.cdist(fake, real)
    precision = (d <= rr[None, :]).any(1).float().mean()
    recall = (d.T <= rf[None, :]).any(1).float().mean()
    return dict(precision=precision.item(), recall=recall.item())


def density_coverage(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict:
    rr = _knn_radius(real, k)
    d = torch.cdist(fake, real)
    inside = d <= rr[None, :]
    density = inside.float().sum(1).mean() / k
    cov = (d.min(0).values <= rr).float().mean()
    return dict(density=density.item(), coverage=cov.item())


def mmd(real: torch.Tensor, fake: torch.Tensor) -> float:
    """Discrepancia media máxima al cuadrado (estimador sin sesgo), ancho por la mediana de distancias."""
    Z = torch.cat([real, fake])
    d2 = torch.cdist(Z, Z).pow(2)
    bw = d2[d2 > 0].median()
    K = torch.exp(-d2 / bw)
    R = real.shape[0]
    kxx, kyy, kxy = K[:R, :R], K[R:, R:], K[:R, R:]
    n, m = R, fake.shape[0]
    val = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (m * (m - 1)) - 2 * kxy.mean()
    return val.item()


def fidelity_diversity(real: torch.Tensor, fake: torch.Tensor, k: int = 5) -> dict:
    out = precision_recall(real, fake, k)
    out.update(density_coverage(real, fake, k))
    out["mmd"] = mmd(real, fake)
    return out
