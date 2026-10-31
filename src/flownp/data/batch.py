from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    xc: torch.Tensor
    yc: torch.Tensor
    xt: torch.Tensor
    yt: torch.Tensor

    @property
    def x(self) -> torch.Tensor:
        return torch.cat([self.xc, self.xt], dim=1)

    @property
    def y(self) -> torch.Tensor:
        return torch.cat([self.yc, self.yt], dim=1)

    @property
    def num_ctx(self) -> int:
        return self.xc.shape[1]

    @property
    def num_tar(self) -> int:
        return self.xt.shape[1]

    def to(self, device) -> "Batch":
        return Batch(self.xc.to(device), self.yc.to(device), self.xt.to(device), self.yt.to(device))

    def as_attrdict(self):
        """Versión con atributos ``x, y, xc, yc, xt, yt``, la que esperan los modelos oficiales."""

        class _AttrDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        return _AttrDict(x=self.x, y=self.y, xc=self.xc, yc=self.yc, xt=self.xt, yt=self.yt)

    @staticmethod
    def from_mapping(m) -> "Batch":
        """Construye un ``Batch`` a partir de un lote del código oficial (diccionario con ``xc, yc, xt, yt``)."""
        return Batch(m["xc"], m["yc"], m["xt"], m["yt"])
