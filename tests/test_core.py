import math

import pytest
import torch

from flownp.data.batch import Batch
from flownp.data.cosine import CosineSampler
from flownp.data.gp import GPSampler
from flownp.flow import core
from flownp.models.flownp import VelocityTransformer


class GaussianVelocity(torch.nn.Module):
    """Velocidad exacta del flujo lineal cuando cada objetivo es N(mu, s²) independiente."""

    dim_y = 1

    def __init__(self, mu=0.7, s=0.3):
        super().__init__()
        self.mu, self.s = mu, s

    def forward(self, yt, t, xt, xc, yc):
        t = torch.as_tensor(t, dtype=yt.dtype)
        if t.dim() == 2:
            t = t[..., None]
        var = t**2 * self.s**2 + (1 - t) ** 2
        cov = t * self.s**2 - (1 - t)
        return self.mu + cov / var * (yt - t * self.mu)


def _toy_batch(B=8, M=3, N=5):
    g = torch.Generator().manual_seed(0)
    return Batch(torch.randn(B, M, 1, generator=g), torch.randn(B, M, 1, generator=g),
                 torch.randn(B, N, 1, generator=g), 0.7 + 0.3 * torch.randn(B, N, 1, generator=g))


def _true_ll(batch, mu=0.7, s=0.3):
    lp = -0.5 * (((batch.yt - mu) / s) ** 2 + math.log(2 * math.pi * s**2))
    return lp.flatten(1).sum(-1) / batch.num_tar


@pytest.mark.parametrize("method,steps", [("midpoint", 100), ("rk4", 20)])
def test_likelihood_exact_trace(method, steps):
    b = _toy_batch()
    ll = core.log_likelihood(GaussianVelocity(), b, steps=steps, method=method, exact=True)
    assert torch.allclose(ll, _true_ll(b), atol=2e-3), (ll - _true_ll(b)).abs().max()


def test_euler_overestimates_but_converges():
    b = _toy_batch()
    err = {n: (core.log_likelihood(GaussianVelocity(), b, steps=n, method="euler", exact=True)
               - _true_ll(b)).mean().item() for n in (100, 400)}
    assert err[100] > 0 and err[400] > 0 and err[400] < err[100] / 3


def test_likelihood_hutchinson_is_unbiased():
    b = _toy_batch()
    ll = core.log_likelihood(GaussianVelocity(), b, steps=100, method="midpoint", exact=False,
                             generator=torch.Generator().manual_seed(1))
    assert torch.allclose(ll, _true_ll(b), atol=2e-3)


def test_sampling_matches_target_distribution():
    b = _toy_batch(B=4, N=10)
    y = core.sample(GaussianVelocity(), b.xc, b.yc, b.xt, num_samples=500, steps=50, method="rk4",
                    generator=torch.Generator().manual_seed(2))
    assert abs(y.mean().item() - 0.7) < 0.01 and abs(y.std().item() - 0.3) < 0.01


def test_transformer_permutation_symmetry():
    torch.manual_seed(0)
    m = VelocityTransformer().eval()
    b = _toy_batch(B=2, M=6, N=7)
    t = torch.rand(2, 7)
    v = m(b.yt, t, b.xt, b.xc, b.yc)
    pc, pt = torch.randperm(6), torch.randperm(7)
    v2 = m(b.yt[:, pt], t[:, pt], b.xt[:, pt], b.xc[:, pc], b.yc[:, pc])
    assert torch.allclose(v[:, pt], v2, atol=1e-5)


def test_transformer_hutchinson_vs_exact():
    torch.manual_seed(0)
    m = VelocityTransformer(d_model=32, dim_feedforward=64, num_layers=2).eval()
    b = _toy_batch(B=2, M=4, N=6)
    exact = core.log_likelihood(m, b, steps=10, exact=True)
    est = torch.stack([core.log_likelihood(m, b, steps=10, generator=torch.Generator().manual_seed(s))
                       for s in range(200)]).mean(0)
    assert torch.allclose(exact, est, atol=0.02), (exact, est)


def test_training_loss_decreases():
    torch.manual_seed(0)
    m = VelocityTransformer(d_model=32, dim_feedforward=64, num_layers=2)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    s = GPSampler("rbf")
    losses = []
    for _ in range(150):
        opt.zero_grad()
        l = core.loss(m, s.sample(16))
        l.backward()
        opt.step()
        losses.append(l.item())
    assert sum(losses[-20:]) / 20 < 0.8 * sum(losses[:20]) / 20


def test_cosine_posterior_concentrates_and_is_exact_without_context():
    s = CosineSampler()
    g = torch.Generator().manual_seed(0)
    b, k, _ = s.sample(64, num_ctx=30, num_tar=10, generator=g, return_params=True)
    log_w, _, _ = s.posterior(b)
    assert (log_w.argmax(-1) == k).float().mean() > 0.95
    b0 = Batch(b.xc[:, :0], b.yc[:, :0], b.xt, b.yt)
    assert torch.allclose(s.posterior(b0)[0].exp(), torch.full((64, 4), 0.25, dtype=torch.float64))


def test_cosine_exact_samples_and_classifier():
    s = CosineSampler()
    g = torch.Generator().manual_seed(0)
    b = s.sample(16, num_ctx=2, num_tar=40, generator=g)
    y, k = s.exact_samples(b, 200, generator=g)
    kk, rmse, amp = s.classify(b.xt, y)
    assert (kk == k).float().mean() > 0.9
    assert rmse.median() < 0.07
    assert s.is_cosine(rmse, amp).float().mean() > 0.95
    flat = torch.zeros_like(y) + 0.01 * torch.randn_like(y)
    _, r2, a2 = s.classify(b.xt, flat)
    assert s.is_cosine(r2, a2).float().mean() < 0.05


def test_resume_is_identical_to_uninterrupted(tmp_path):
    from flownp.training import train
    full = torch.load(train(str(tmp_path / "a"), num_steps=30, save_every=10, log_every=10, device="cpu"),
                      weights_only=False)
    train(str(tmp_path / "b"), num_steps=30, save_every=10, log_every=10, device="cpu", stop_after=20)
    resumed = torch.load(train(str(tmp_path / "b"), num_steps=30, save_every=10, log_every=10, device="cpu"),
                         weights_only=False)
    assert resumed["step"] == full["step"] == 30
    for k in full["model"]:
        assert torch.equal(full["model"][k], resumed["model"][k]), k


def test_cosine_metrics_reward_exact_posterior_and_punish_blur():
    from flownp.evaluation import evaluate_cosine

    s = CosineSampler()

    class Exact:
        model = torch.nn.Linear(1, 1)
        kind = "exact"

        def samples(self, xc, yc, xt, n, generator=None):
            return s.exact_samples(Batch(xc, yc, xt, xt), n)[0]

    class Blur(Exact):
        def samples(self, xc, yc, xt, n, generator=None):
            y = s.exact_samples(Batch(xc, yc, xt, xt), n)[0]
            return y.mean(0, keepdim=True) + y.std(0, keepdim=True) * torch.randn_like(y)

    good = evaluate_cosine(Exact(), s, num_ctx_list=(2,), num_tasks=8, num_samples=64, grid_points=40)[2]
    bad = evaluate_cosine(Blur(), s, num_ctx_list=(2,), num_tasks=8, num_samples=64, grid_points=40)[2]
    assert good["fidelity_is_cosine"] > 0.95 and good["freq_tv_distance"] < 0.2
    assert bad["fidelity_is_cosine"] < 0.2 and bad["freq_tv_distance"] > 0.7
    assert good["coverage"] > bad["coverage"]


def test_init_from_checkpoint(tmp_path):
    from flownp.training import train
    a = train(str(tmp_path / "a"), num_steps=5, save_every=5, log_every=5, device="cpu")
    b = train(str(tmp_path / "b"), data="cosine", num_steps=1, save_every=1, log_every=1, device="cpu", init=a)
    wa = torch.load(a, weights_only=False)["model"]
    wb = torch.load(b, weights_only=False)
    assert wb["config"]["init"] == a
    diff = max((wa[k] - wb["model"][k]).abs().max().item() for k in wa)
    assert 0 < diff < 1e-2
