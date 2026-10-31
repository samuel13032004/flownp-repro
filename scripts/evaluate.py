import argparse
import json
import os

import torch

from flownp.data import evalsets
from flownp.data.cosine import CosineSampler
from flownp.evaluation import evaluate_calibration, evaluate_cosine, evaluate_likelihood
from flownp.models.official import flownp_from_official_checkpoint, load_official_model
from flownp.models.predictors import Predictor
from flownp.training import make_predictor

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--model", default="flownp")
p.add_argument("--evalset", default=None, help=".tar oficial o .pt nuestro")
p.add_argument("--tasks", default="likelihood,calibration")
p.add_argument("--out", required=True, help="fichero JSON de salida (se completa si ya existe)")
p.add_argument("--official-code", default=None)
p.add_argument("--max-batches", type=int, default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default=None)
a = p.parse_args()
QUICK = os.environ.get("FLOWNP_QUICK") == "1"
dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

if a.model == "official-fnp":
    pred = Predictor(flownp_from_official_checkpoint(a.ckpt), "flownp")
elif a.model.startswith("official:"):
    name = a.model.split(":", 1)[1]
    pred = Predictor(load_official_model(a.official_code, name, checkpoint=a.ckpt), name)
else:
    pred = make_predictor("flownp", checkpoint=a.ckpt)
pred.model.to(dev).eval()

batches = None
if a.evalset:
    batches = evalsets.load_official(a.evalset) if a.evalset.endswith(".tar") else evalsets.load(a.evalset)

res = json.load(open(a.out)) if os.path.isfile(a.out) else {}
res.update(ckpt=a.ckpt, model=a.model, evalset=a.evalset)
tasks = a.tasks.split(",")
if "likelihood" in tasks:
    if QUICK and pred.kind == "flownp":
        pred.ll_steps = 5
    if QUICK and pred.kind in ("ndp", "fnp"):
        pred.model.timesteps = 5
    res["likelihood"] = evaluate_likelihood(pred, batches, dev, a.seed, 1 if QUICK else a.max_batches)
    print("likelihood", res["likelihood"], flush=True)
if "calibration" in tasks:
    res["calibration"] = evaluate_calibration(pred, batches, device=dev, seed=a.seed,
                                              **(dict(num_samples=8, max_batches=1) if QUICK else {}))
    print("calibration", {k: v for k, v in res["calibration"].items() if k != "pit_hist"}, flush=True)
if "solvers" in tasks:
    assert pred.kind == "flownp"
    sweep = []
    for method in ("euler", "midpoint", "rk4"):
        for steps in ((2, 4) if QUICK else (5, 10, 20, 50, 100)):
            pred.method, pred.steps, pred.ll_method, pred.ll_steps = method, steps, method, steps
            ll = evaluate_likelihood(pred, batches, dev, a.seed, max_batches=1 if QUICK else (a.max_batches or 20))
            cal = evaluate_calibration(pred, batches, device=dev, seed=a.seed, max_batches=1 if QUICK else 5,
                                       num_samples=8 if QUICK else 64)
            sweep.append(dict(method=method, steps=steps, tar_ll=ll["tar_ll"], ll_seconds=ll["seconds"],
                              crps=cal["crps"], coverage_90=cal["coverage_90"], sample_seconds=cal["seconds"]))
            print(sweep[-1], flush=True)
    res["solvers"] = sweep
if "cosine" in tasks:
    kw = dict(num_ctx_list=(2,), num_tasks=2, num_samples=8, grid_points=30) if QUICK else {}
    if QUICK and pred.kind == "flownp":
        pred.steps = 5
    if QUICK and pred.kind == "ndp":
        pred.model.timesteps = 5
    res["cosine"] = evaluate_cosine(pred, CosineSampler(), device=dev, seed=a.seed, **kw)
    for M, r in res["cosine"].items():
        print("cosine M =", M, {k: round(v, 3) for k, v in r.items()}, flush=True)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=2)
print("guardado en", a.out)
