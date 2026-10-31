from __future__ import annotations

import math

import torch

from .batch import Batch


class CosineSampler:
    def __init__(
        self,
        freqs=(0.25, 0.5, 0.75, 1.0),
        sigma_amp: float = 1.0,
        sigma_noise: float = 0.05,
        max_num_points: int = 50,
        min_num_ctx: int = 1,
        max_num_ctx: int = 20,
        x_range=(-2.0, 2.0),
    ):
        self.freqs = torch.tensor(freqs, dtype=torch.float64)
        self.sigma_amp = sigma_amp
        self.sigma_noise = sigma_noise
        self.max_num_points = max_num_points
        self.min_num_ctx = min_num_ctx
        self.max_num_ctx = max_num_ctx
        self.x_range = x_range

    @property
    def num_freqs(self) -> int:
        return len(self.freqs)

    def sample_sizes(self, generator=None) -> tuple[int, int]:
        num_ctx = int(torch.randint(self.min_num_ctx, self.max_num_ctx + 1, (1,), generator=generator))
        num_tar = int(torch.randint(3, self.max_num_points - num_ctx + 1, (1,), generator=generator))
        return num_ctx, num_tar

    def sample_params(self, batch_size, device="cpu", generator=None):
        k = torch.randint(0, self.num_freqs, (batch_size,), device=device, generator=generator)
        ab = self.sigma_amp * torch.randn(batch_size, 2, device=device, generator=generator)
        return k, ab

    def evaluate(self, x, k, ab, noise=True, generator=None):
        """Valores en ``x`` [B, N, 1] de las funciones con frecuencia ``k`` [B] y coeficientes ``ab`` [B, 2]."""
        f = self.freqs.to(x.device, x.dtype)[k][:, None, None]
        w = 2 * math.pi * f * x
        y = ab[:, None, 0:1] * torch.cos(w) + ab[:, None, 1:2] * torch.sin(w)
        if noise:
            y = y + self.sigma_noise * torch.randn(y.shape, device=x.device, generator=generator, dtype=x.dtype)
        return y

    def sample(self, batch_size=16, num_ctx=None, num_tar=None, device="cpu", generator=None, return_params=False):
        if num_ctx is None or num_tar is None:
            c, t = self.sample_sizes(generator)
            num_ctx = c if num_ctx is None else num_ctx
            num_tar = t if num_tar is None else num_tar
        lo, hi = self.x_range
        x = lo + (hi - lo) * torch.rand(batch_size, num_ctx + num_tar, 1, device=device, generator=generator)
        k, ab = self.sample_params(batch_size, device, generator)
        y = self.evaluate(x, k, ab, generator=generator)
        b = Batch(x[:, :num_ctx], y[:, :num_ctx], x[:, num_ctx:], y[:, num_ctx:])
        return (b, k, ab) if return_params else b

    def _cov(self, xa, xb, k_index: int, noise: bool):
        """Covarianza [B, Na, Nb] dada la frecuencia ``k_index`` (común a todo el lote)."""
        f = float(self.freqs[k_index])
        da = xa[..., 0]
        db = xb[..., 0]
        c = self.sigma_amp**2 * torch.cos(2 * math.pi * f * (da[:, :, None] - db[:, None, :]))
        if noise:
            c = c + self.sigma_noise**2 * torch.eye(xa.shape[1], dtype=xa.dtype, device=xa.device)
        return c

    def posterior(self, batch: Batch):
        """Posterior exacto dado el contexto, en doble precisión."""
        xc, yc, xt = batch.xc.double(), batch.yc.double()[..., 0], batch.xt.double()
        B, M = yc.shape
        log_w, means, covs = [], [], []
        for j in range(self.num_freqs):
            Ktt = self._cov(xt, xt, j, noise=True)
            if M == 0:
                log_w.append(torch.zeros(B, dtype=torch.float64, device=yc.device))
                means.append(torch.zeros(B, xt.shape[1], dtype=torch.float64, device=yc.device))
                covs.append(Ktt)
                continue
            Kcc = self._cov(xc, xc, j, noise=True)
            Ktc = self._cov(xt, xc, j, noise=False)
            L = torch.linalg.cholesky(Kcc)
            alpha = torch.cholesky_solve(yc[..., None], L)
            logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
            quad = (yc[..., None] * alpha).sum((-2, -1))
            log_w.append(-0.5 * (quad + logdet + M * math.log(2 * math.pi)))
            V = torch.cholesky_solve(Ktc.transpose(-1, -2), L)
            means.append((Ktc @ alpha)[..., 0])
            covs.append(Ktt - Ktc @ V)
        log_w = torch.stack(log_w, -1)
        log_w = log_w - torch.logsumexp(log_w, -1, keepdim=True)
        return log_w, torch.stack(means, 1), torch.stack(covs, 1)

    def exact_log_likelihood(self, batch: Batch) -> torch.Tensor:
        """log p(yt | contexto) exacto por tarea, dividido por el número de objetivos [B]."""
        log_w, mu, cov = self.posterior(batch)
        yt = batch.yt.double()[..., 0]
        dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
        lp = dist.log_prob(yt[:, None, :].expand_as(mu))
        return torch.logsumexp(log_w + lp, -1) / yt.shape[1]

    def exact_samples(self, batch: Batch, num_samples: int, generator=None):
        """Muestras exactas de los objetivos [S, B, N, 1] y la frecuencia de cada una [S, B]."""
        log_w, mu, cov = self.posterior(batch)
        B, K, N = mu.shape
        probs = log_w.exp()
        k = torch.multinomial(probs, num_samples, replacement=True, generator=generator).T
        L = torch.linalg.cholesky(cov + 1e-9 * torch.eye(N, dtype=cov.dtype, device=cov.device))
        bi = torch.arange(B, device=mu.device)[None, :].expand(num_samples, B)
        eps = torch.randn(num_samples, B, N, 1, dtype=mu.dtype, device=mu.device, generator=generator)
        y = mu[bi, k][..., None] + L[bi, k] @ eps
        return y.to(batch.yt.dtype), k

    def classify(self, x, y):
        """Ajusta un coseno de cada frecuencia por mínimos cuadrados."""
        if x.dim() == 2:
            x = x[None].expand(y.shape[1], -1, -1)
        rmse, amp = [], []
        yy = y[..., 0].double()
        for j in range(self.num_freqs):
            w = 2 * math.pi * float(self.freqs[j]) * x[..., 0].double()
            Phi = torch.stack([torch.cos(w), torch.sin(w)], -1)
            coef = (torch.linalg.pinv(Phi)[None] @ yy[..., None])[..., 0]
            res = yy - (Phi[None] @ coef[..., None])[..., 0]
            rmse.append(res.pow(2).mean(-1).sqrt())
            amp.append(coef.norm(dim=-1))
        rmse, amp = torch.stack(rmse, -1), torch.stack(amp, -1)
        best = rmse.min(-1)
        return best.indices, best.values, amp.gather(-1, best.indices[..., None])[..., 0]

    def is_cosine(self, rmse, amp):
        """Muestra válida: el error del ajuste es del orden del ruido y la amplitud no es despreciable."""
        return (rmse < 2 * self.sigma_noise) & (amp > 2 * self.sigma_noise)
