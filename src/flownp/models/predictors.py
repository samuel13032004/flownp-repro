from __future__ import annotations

import torch

from ..data.batch import Batch
from ..flow import core


class Predictor:
    def __init__(self, model, kind: str, steps: int = 100, method: str = "euler", ll_steps: int = 100,
                 ll_method: str = "midpoint", anp_eval_samples: int = 200, train_num_samples: int = 4):
        self.model, self.kind = model, kind
        self.steps, self.method = steps, method
        self.ll_steps, self.ll_method = ll_steps, ll_method
        self.anp_eval_samples, self.train_num_samples = anp_eval_samples, train_num_samples

    @torch.no_grad()
    def samples(self, xc, yc, xt, num_samples: int, generator=None) -> torch.Tensor:
        m, k = self.model, self.kind
        if k == "flownp":
            return core.sample(m, xc, yc, xt, num_samples, self.steps, self.method, generator=generator)
        if k == "fnp":
            return m.predict(xc, yc, xt, num_samples=num_samples).loc
        if k == "ndp":
            with torch.enable_grad():
                return m.predict(xc, yc, xt, num_samples=num_samples).loc
        if k == "cnp":
            return m.predict(xc, yc, xt).sample([num_samples])
        if k == "anp":
            return m.predict(xc, yc, xt, num_samples=num_samples).sample()
        if k == "tnpa":
            S, B, N = num_samples, xt.shape[0], xt.shape[1]
            rep = lambda z: z.repeat(S, 1, 1)
            p = torch.argsort(torch.rand(S * B, N, device=xt.device, generator=generator), dim=1)
            xt_p = torch.gather(rep(xt), 1, p[..., None].expand(-1, -1, xt.shape[-1]))
            perm_flag, m.permute = m.permute, False
            try:
                s = m.sample(rep(xc), rep(yc), xt_p, num_samples=1)[0]
            finally:
                m.permute = perm_flag
            s = torch.gather(s, 1, torch.argsort(p, dim=1)[..., None].expand(-1, -1, s.shape[-1]))
            return s.reshape(S, B, N, -1)
        raise ValueError(f"modelo desconocido: {k}")

    @torch.no_grad()
    def log_likelihood(self, batch: Batch, generator=None) -> torch.Tensor:
        if self.kind == "flownp":
            return core.log_likelihood(self.model, batch, self.ll_steps, self.ll_method, generator=generator)
        B = batch.xc.shape[0]
        if self.kind == "anp":
            ll = self.model(batch.as_attrdict(), num_samples=self.anp_eval_samples, reduce_ll=False).tar_ll
            S = ll.shape[0]
            return (torch.logsumexp(ll, 0) - torch.log(torch.tensor(float(S)))).mean(-1)
        if self.kind == "ndp":
            with torch.enable_grad():
                return self.model(batch.as_attrdict()).tar_ll.expand(B)
        return self.model(batch.as_attrdict(), reduce_ll=False).tar_ll.flatten(1).mean(-1)

    def loss(self, batch: Batch, time_per_point: bool = True) -> torch.Tensor:
        if self.kind == "flownp":
            return core.loss(self.model, batch, time_per_point)
        return self.model(batch.as_attrdict(), num_samples=self.train_num_samples).loss
