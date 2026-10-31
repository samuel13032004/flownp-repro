from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_features(v: torch.Tensor, num_features: int) -> torch.Tensor:
    """[..., D] -> [..., D * num_features] con frecuencias π·2^(j-2), j = 0..num_features/2-1."""
    half = num_features // 2
    omega = math.pi * 2.0 ** (torch.arange(half, device=v.device, dtype=v.dtype) - 2)
    a = v[..., None] * omega
    return torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(-2)


def mlp(dim_in, dim_hid, dim_out, depth):
    layers = [nn.Linear(dim_in, dim_hid), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(dim_hid, dim_hid), nn.ReLU()]
    layers.append(nn.Linear(dim_hid, dim_out))
    return nn.Sequential(*layers)


class VelocityTransformer(nn.Module):
    def __init__(
        self,
        dim_x: int = 1,
        dim_y: int = 1,
        num_features: int = 20,
        d_model: int = 64,
        emb_depth: int = 4,
        dim_feedforward: int = 128,
        nhead: int = 4,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_x, self.dim_y, self.num_features = dim_x, dim_y, num_features
        self.embed = mlp((dim_x + 1) * num_features + dim_y, d_model, d_model, emb_depth)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, dim_y))

    def _tokens(self, x, t, y):
        return self.embed(torch.cat([sinusoidal_features(torch.cat([x, t], -1), self.num_features), y], -1))

    def forward(self, yt: torch.Tensor, t: torch.Tensor, xt: torch.Tensor, xc: torch.Tensor, yc: torch.Tensor):
        """Velocidad de los objetivos."""
        B, N, _ = yt.shape
        t = torch.as_tensor(t, dtype=yt.dtype, device=yt.device)
        if t.dim() == 0:
            t = t.expand(B, N)
        elif t.dim() == 1:
            t = t[:, None].expand(B, N)
        tok_t = self._tokens(xt, t[..., None], yt)
        ones = torch.ones(xc.shape[0], xc.shape[1], 1, dtype=yt.dtype, device=yt.device)
        tok_c = self._tokens(xc, ones, yc)
        h = self.encoder(torch.cat([tok_c, tok_t], 1))
        return self.head(h[:, xc.shape[1]:])
