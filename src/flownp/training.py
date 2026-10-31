from __future__ import annotations

import json
import os
import platform
import subprocess
import time

import torch

from .data.cosine import CosineSampler
from .data.gp import GPSampler
from .models.flownp import VelocityTransformer
from .models.official import flownp_from_official_checkpoint, load_official_model
from .models.predictors import Predictor


def make_sampler(data: str):
    """``gp:rbf``, ``gp:matern``, ``gp:periodic`` o ``cosine``."""
    if data.startswith("gp:"):
        return GPSampler(data.split(":", 1)[1])
    if data == "cosine":
        return CosineSampler()
    raise ValueError(f"datos desconocidos: {data}")


def make_predictor(model: str, official_code: str | None = None, checkpoint: str | None = None) -> Predictor:
    """``flownp`` (nuestro) u ``official:<nombre>`` (``fnp``, ``cnp``, ``anp``, ``tnpa``, ``ndp``)."""
    if model == "flownp":
        m = VelocityTransformer()
        if checkpoint:
            m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
        return Predictor(m, "flownp")
    if model.startswith("official:"):
        name = model.split(":", 1)[1]
        assert official_code, "hace falta la ruta del código oficial clonado"
        m = load_official_model(official_code, name, checkpoint=None)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
            m.load_state_dict(state)
        p = Predictor(m, name, train_num_samples=8 if name == "anp" else 4)
        return p
    raise ValueError(f"modelo desconocido: {model}")


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except Exception:
        return None


def _rng_state():
    s = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train(run_dir: str, model: str = "flownp", data: str = "gp:rbf", seed: int = 0, num_steps: int = 100_000,
          batch_size: int = 16, lr: float = 5e-4, save_every: int = 1000, log_every: int = 1000,
          time_per_point: bool = True, official_code: str | None = None, device: str | None = None,
          stop_after: int | None = None, init: str | None = None) -> str:
    """Entrena (o reanuda) y devuelve la ruta del punto de control final."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    cfg = dict(model=model, data=data, seed=seed, num_steps=num_steps, batch_size=batch_size, lr=lr,
               time_per_point=time_per_point, init=init)

    torch.manual_seed(seed)
    pred = make_predictor(model, official_code)
    net = pred.model.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)
    sampler = make_sampler(data)
    step = 0

    if os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck["config"] != cfg:
            raise RuntimeError(f"{run_dir} ya tiene un punto de control con otra configuración: {ck['config']}")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        _set_rng_state(ck["rng"])
        step = ck["step"]
        if step >= num_steps:
            return ckpt_path
        print(f"reanudando {run_dir} desde el paso {step}", flush=True)
    else:
        if init:
            state = torch.load(init, map_location=device, weights_only=False)["model"]
            if model == "flownp" and any(k.startswith("embedder.") for k in state):
                state = flownp_from_official_checkpoint(init).state_dict()
            net.load_state_dict(state)
            print(f"pesos iniciales de {init}", flush=True)
        meta = dict(cfg, git_commit=_git_commit(), torch=torch.__version__, python=platform.python_version(),
                    device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                    num_params=sum(p.numel() for p in net.parameters()), started=time.strftime("%Y-%m-%d %H:%M"))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    log = open(os.path.join(run_dir, "train.log"), "a")
    net.train()
    t0, acc = time.time(), 0.0
    while step < num_steps:
        batch = sampler.sample(batch_size, device=device)
        opt.zero_grad(set_to_none=True)
        loss = pred.loss(batch, time_per_point)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        acc += loss.item()
        if step % log_every == 0:
            line = f"step {step} loss {acc / log_every:.4f} lr {sched.get_last_lr()[0]:.3e} ({time.time() - t0:.1f} s)"
            print(line, file=log, flush=True)
            acc, t0 = 0.0, time.time()
        if step % save_every == 0 or step == num_steps:
            tmp = ckpt_path + ".tmp"
            torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                            rng=_rng_state(), step=step, config=cfg), tmp)
            os.replace(tmp, ckpt_path)
        if stop_after is not None and step >= stop_after:
            break
    return ckpt_path
