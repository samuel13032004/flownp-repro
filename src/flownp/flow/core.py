from __future__ import annotations

import math

import torch

from ..data.batch import Batch


def loss(model, batch: Batch, time_per_point: bool = True, generator=None) -> torch.Tensor:
    """Pérdida de emparejamiento de flujo de un lote."""
    y1 = batch.yt
    y0 = torch.randn(y1.shape, device=y1.device, generator=generator)
    shape = y1.shape[:2] if time_per_point else y1.shape[:1]
    t = torch.rand(shape, device=y1.device, generator=generator)
    tt = t[..., None] if time_per_point else t[:, None, None]
    yt = tt * y1 + (1 - tt) * y0
    v = model(yt, t, batch.xt, batch.xc, batch.yc)
    return (v - (y1 - y0)).pow(2).mean()


def _step(f, y, t, h, method):
    if method == "euler":
        return y + h * f(y, t)
    if method == "midpoint":
        return y + h * f(y + 0.5 * h * f(y, t), t + 0.5 * h)
    if method == "rk4":
        k1 = f(y, t)
        k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(y + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(y + h * k3, t + h)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"resolvedor desconocido: {method}")


EVALS_PER_STEP = {"euler": 1, "midpoint": 2, "rk4": 4}


@torch.no_grad()
def sample(model, xc, yc, xt, num_samples=1, steps=100, method="euler", stochastic=False, generator=None):
    """Muestras de los objetivos dado el contexto: [S, B, N, dim_y]."""
    B, N = xt.shape[:2]
    S = num_samples
    rep = lambda z: z.repeat(S, *([1] * (z.dim() - 1)))
    xc_, yc_, xt_ = rep(xc), rep(yc), rep(xt)
    f = lambda y, t: model(y, t, xt_, xc_, yc_)
    y = torch.randn(S * B, N, model.dim_y, device=xt.device, generator=generator)
    h = 1.0 / steps
    for i in range(steps):
        t = i * h
        if stochastic:
            alpha = 1 + t * (1 - t)
            sigma = 0.2 * math.sqrt(t * (1 - t))
            noise = torch.randn(y.shape, device=y.device, generator=generator)
            y = y + h * (alpha * f(y, t) + sigma * noise)
        else:
            y = _step(f, y, t, h, method)
    return y.reshape(S, B, N, -1)


def _divergence(f, y, t, z=None):
    """Velocidad y su divergencia respecto a ``y`` por función: ([B,N,D], [B])."""
    with torch.enable_grad():
        y = y.detach().requires_grad_(True)
        v = f(y, t)
        if z is not None:
            (g,) = torch.autograd.grad((v * z).sum(), y)
            div = (g * z).flatten(1).sum(-1)
        else:
            flat = v.flatten(1)
            div = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
            for i in range(flat.shape[1]):
                (g,) = torch.autograd.grad(flat[:, i].sum(), y, retain_graph=i < flat.shape[1] - 1)
                div = div + g.flatten(1)[:, i]
    return v.detach(), div.detach()


@torch.no_grad()
def log_likelihood(model, batch: Batch, steps=100, method="midpoint", exact=False, generator=None):
    """log p(yt | contexto) por función, dividido por el número de objetivos [B]."""
    xc, yc, xt, y = batch.xc, batch.yc, batch.xt, batch.yt
    f = lambda yy, t: model(yy, t, xt, xc, yc)
    z = None
    if not exact:
        z = (torch.randint(0, 2, y.shape, device=y.device, generator=generator) * 2 - 1).to(y.dtype)
    h = -1.0 / steps
    t = 1.0
    acc = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
    for _ in range(steps):
        if method == "euler":
            v, d = _divergence(f, y, t, z)
            y, acc = y + h * v, acc + h * d
        elif method == "midpoint":
            ym = y + 0.5 * h * f(y, t)
            vm, dm = _divergence(f, ym, t + 0.5 * h, z)
            y, acc = y + h * vm, acc + h * dm
        elif method == "rk4":
            k1, d1 = _divergence(f, y, t, z)
            k2, d2 = _divergence(f, y + 0.5 * h * k1, t + 0.5 * h, z)
            k3, d3 = _divergence(f, y + 0.5 * h * k2, t + 0.5 * h, z)
            k4, d4 = _divergence(f, y + h * k3, t + h, z)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            acc = acc + h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
        else:
            raise ValueError(f"resolvedor desconocido: {method}")
        t = t + h
    log_p0 = -0.5 * (y.pow(2) + math.log(2 * math.pi)).flatten(1).sum(-1)
    return (log_p0 + acc) / batch.num_tar
