from __future__ import annotations

import math
import time

import torch

from .data.batch import Batch
from .data.cosine import CosineSampler
from .eval import metrics
from .models.predictors import Predictor


def evaluate_likelihood(pred: Predictor, batches: list[Batch], device="cpu", seed=0, max_batches=None) -> dict:
    g = torch.Generator(device=device).manual_seed(seed)
    pred.model.eval()
    vals, t0 = [], time.time()
    for b in batches[:max_batches]:
        vals.append(pred.log_likelihood(b.to(device), generator=g).mean().item())
    return dict(tar_ll=sum(vals) / len(vals), num_batches=len(vals), seconds=time.time() - t0)


def evaluate_calibration(pred: Predictor, batches: list[Batch], num_samples=64, device="cpu", seed=0,
                         max_batches=10) -> dict:
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    pred.model.eval()
    S, Y, t0 = [], [], time.time()
    for b in batches[:max_batches]:
        b = b.to(device)
        S.append(pred.samples(b.xc, b.yc, b.xt, num_samples, generator=g).flatten(1).cpu())
        Y.append(b.yt.flatten().cpu())
    out = metrics.calibration_summary(torch.cat(S, 1), torch.cat(Y))
    out["seconds"] = time.time() - t0
    return out


def evaluate_cosine(pred: Predictor, sampler: CosineSampler, num_ctx_list=(1, 2, 3, 5, 10, 20), num_tasks=64,
                    num_samples=64, grid_points=100, device="cpu", seed=0) -> dict:
    """Para cada número de puntos de contexto, compara las muestras del modelo con el posterior exacto."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    pred.model.eval()
    lo, hi = sampler.x_range
    grid = torch.linspace(lo, hi, grid_points)[None, :, None]
    results = {}
    for M in num_ctx_list:
        b, k_true, _ = sampler.sample(num_tasks, num_ctx=M, num_tar=grid_points, generator=g, return_params=True)
        b = Batch(b.xc, b.yc, grid.expand(num_tasks, -1, -1).clone(), b.yt)
        log_w, _, _ = sampler.posterior(b)
        p_true = log_w.exp().float()
        real, _ = sampler.exact_samples(b, num_samples, generator=g)
        t0 = time.time()
        fake = pred.samples(b.xc.to(device), b.yc.to(device), b.xt.to(device), num_samples).float().cpu()
        secs = time.time() - t0
        kf, rmse, amp = sampler.classify(b.xt, fake)
        is_cos = sampler.is_cosine(rmse, amp)
        K = sampler.num_freqs
        p_model = torch.stack([((kf == j) & is_cos).float().mean(0) for j in range(K)], -1)
        p_invalid = 1 - p_model.sum(-1)
        tv = 0.5 * ((p_model - p_true).abs().sum(-1) + p_invalid)
        plausible = p_true > 0.1
        hit = (p_model > 0) & plausible
        modes = hit.sum(-1).float() / plausible.sum(-1).clamp(min=1)
        fd = [metrics.fidelity_diversity(real[:, i, :, 0].float(), fake[:, i, :, 0]) for i in range(num_tasks)]
        agg = {key: sum(d[key] for d in fd) / num_tasks for key in fd[0]}
        results[M] = dict(
            fidelity_is_cosine=is_cos.float().mean().item(),
            freq_tv_distance=tv.mean().item(),
            plausible_modes_found=modes.mean().item(),
            num_plausible_modes=plausible.sum(-1).float().mean().item(),
            seconds_per_task=secs / num_tasks,
            **agg,
        )
    return results


def exact_cosine_likelihood(sampler: CosineSampler, batches: list[Batch]) -> float:
    return sum(sampler.exact_log_likelihood(b).mean().item() for b in batches) / len(batches)
