from __future__ import annotations

import importlib.util
import os
import sys

import torch
import yaml

from .flownp import VelocityTransformer


def _import(code_dir: str, name: str):
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    spec = importlib.util.spec_from_file_location(name, os.path.join(code_dir, "models", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_official_model(code_dir: str, name: str, config: str = "gp", checkpoint: str | None = None, **overrides):
    with open(os.path.join(code_dir, "configs", config, f"{name}.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    model = getattr(_import(code_dir, name), name.upper())(**cfg)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    return model


def flownp_from_official_checkpoint(path: str, **kwargs) -> VelocityTransformer:
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    rename = {"embedder.": "embed.", "predictor.": "head.", "encoder.": "encoder."}
    new = {}
    for k, v in state.items():
        for a, b in rename.items():
            if k.startswith(a):
                new[b + k[len(a):]] = v
    model = VelocityTransformer(**kwargs)
    model.load_state_dict(new)
    return model
