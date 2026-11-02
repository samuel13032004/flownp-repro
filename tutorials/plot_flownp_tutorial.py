"""
Procesos neuronales con emparejamiento de flujo
===============================================

Este tutorial muestra, paso a paso, cómo funciona **FlowNP** (Abu Hamad y
Rosenbaum, NeurIPS 2025) con nuestra reimplementación ``flownp``:

1. qué es un proceso neuronal y qué datos usa;
2. cómo se construye el modelo;
3. cómo se entrena con emparejamiento de flujo;
4. cómo genera funciones condicionadas a unos puntos observados;
5. cómo se calcula la verosimilitud;
6. cómo influye el resolvedor de la ecuación diferencial en la calidad y el coste;
7. por qué el conjunto de cosenos sirve para medir fidelidad y diversidad.

Todo se ejecuta en procesador en pocos minutos. El modelo que se entrena aquí
es una versión corta (unos miles de pasos); los resultados del artículo usan
100 000 pasos en una tarjeta gráfica.
"""

# %%
# 1. Procesos neuronales
# ----------------------
#
# Un proceso neuronal aprende una **distribución sobre funciones**. Se le dan
# unos puntos observados de una función, el *contexto*
# :math:`\mathcal{C} = \{(x_i, y_i)\}_{i=1}^{M}`, y devuelve la distribución de
# los valores de la función en otros puntos, los *objetivos*:
#
# .. math::
#
#    p_\theta\left(y^{\mathrm{obj}} \mid x^{\mathrm{obj}}, \mathcal{C}\right).
#
# Para entrenarlo se usan muchas funciones de ejemplo. Aquí, funciones
# aleatorias de un proceso gaussiano con núcleo exponencial cuadrático, con
# longitud de escala y amplitud distintas en cada función (el protocolo del
# artículo).

import math
import time

import matplotlib.pyplot as plt
import torch

from flownp.data.batch import Batch
from flownp.data.cosine import CosineSampler
from flownp.data.gp import GPSampler
from flownp.flow import core
from flownp.models.flownp import VelocityTransformer

torch.manual_seed(0)
sampler = GPSampler("rbf")
batch = sampler.sample(batch_size=4, num_ctx=10, num_tar=40)

fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for i, ax in enumerate(axes):
    ax.scatter(batch.xt[i, :, 0], batch.yt[i, :, 0], s=12, c="tab:gray", label="objetivo")
    ax.scatter(batch.xc[i, :, 0], batch.yc[i, :, 0], s=30, c="k", label="contexto")
    ax.set_title(f"función {i + 1}")
axes[0].legend(loc="lower left")
fig.suptitle("Cuatro funciones de entrenamiento: contexto (negro) y objetivos (gris)")
plt.tight_layout()

# %%
# 2. El modelo: un transformador que predice velocidades
# ------------------------------------------------------
#
# FlowNP no predice directamente los valores objetivo. Aprende un **flujo**:
# una ecuación diferencial que transforma ruido gaussiano en valores
# plausibles de la función. Con la trayectoria lineal del emparejamiento de
# flujo,
#
# .. math::
#
#    y_t = t\, y_1 + (1 - t)\, y_0, \qquad y_0 \sim \mathcal{N}(0, I),
#
# la velocidad objetivo es :math:`y_1 - y_0`. Un transformador recibe dos tipos
# de *fichas*:
#
# * **contexto**: :math:`[x_i, t = 1, y_i]`, con el valor observado;
# * **objetivo**: :math:`[x_j, t, y_{t,j}]`, con el valor intermedio del flujo.
#
# y devuelve la velocidad de cada objetivo. Como no usa codificación
# posicional de orden, el resultado no depende del orden en que se den los
# puntos.

model = VelocityTransformer()
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

t = torch.rand(4, 40)
v = model(batch.yt, t, batch.xt, batch.xc, batch.yc)
perm = torch.randperm(10)
v_perm = model(batch.yt, t, batch.xt, batch.xc[:, perm], batch.yc[:, perm])
print("Velocidad:", tuple(v.shape), "| cambio al reordenar el contexto:", f"{(v - v_perm).abs().max():.1e}")

# %%
# 3. Entrenamiento con emparejamiento de flujo
# --------------------------------------------
#
# En cada paso se toma un lote de funciones nuevas, se elige un tiempo
# :math:`t` al azar para cada punto objetivo, se interpola entre ruido y dato y
# se minimiza el error cuadrático entre la velocidad predicha y
# :math:`y_1 - y_0`:
#
# .. math::
#
#    \mathcal{L}(\theta) = \mathbb{E}\,\bigl\|v_\theta(y_t, t, x^{\mathrm{obj}}, \mathcal{C}) - (y_1 - y_0)\bigr\|^2 .
#
# Aquí entrenamos 3000 pasos; el artículo usa 100 000. Al principio de
# entrenar, FlowNP suele pasar por una *meseta* en la que ignora el contexto;
# cuánto tarda en salir depende de la inicialización.

torch.manual_seed(1)
model = VelocityTransformer()
steps = 3000
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
losses, t0 = [], time.time()
for step in range(steps):
    opt.zero_grad()
    loss = core.loss(model, sampler.sample(16))
    loss.backward()
    opt.step()
    sched.step()
    losses.append(loss.item())
print(f"Entrenado en {time.time() - t0:.0f} s")

smooth = torch.tensor(losses).unfold(0, 100, 100).mean(1)
plt.figure(figsize=(6, 3))
plt.plot(torch.arange(len(smooth)) * 100, smooth)
plt.xlabel("paso")
plt.ylabel("pérdida")
plt.title("Pérdida de emparejamiento de flujo")
plt.tight_layout()

# %%
# Con tan pocos pasos el modelo puede no haber salido todavía de la meseta.
# Para el resto del tutorial cargamos el modelo que entrenamos en Colab con la
# configuración completa (100 000 pasos, semilla 0), cuyos pesos están en
# ``tutorials/flownp_rbf_100k.pth``. Si el fichero no está, se sigue con el
# modelo corto.

import os

weights = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ".",
                       "flownp_rbf_100k.pth")
if os.path.isfile(weights):
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    print("Cargado el modelo entrenado 100 000 pasos")
else:
    print("No se encontró el modelo entrenado; se usa el de 3000 pasos")

# %%
# 4. Muestreo condicional
# -----------------------
#
# Para generar, se parte de ruido en los puntos objetivo y se integra la
# ecuación diferencial de :math:`t = 0` a :math:`t = 1`. Todos los objetivos se
# generan **a la vez**, y se pueden pedir en cualquier posición: aquí, en una
# rejilla de 200 puntos, aunque el modelo se entrenó con 50 puntos como máximo.

model.eval()
torch.manual_seed(2)
test = sampler.sample(batch_size=3, num_ctx=8, num_tar=40)
grid = torch.linspace(-2, 2, 200)[None, :, None].repeat(3, 1, 1)
samples = core.sample(model, test.xc, test.yc, grid, num_samples=20, steps=50, method="midpoint")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.2), sharey=True)
for i, ax in enumerate(axes):
    for s in range(samples.shape[0]):
        ax.plot(grid[0, :, 0], samples[s, i, :, 0], c="tab:blue", alpha=0.25, lw=1)
    ax.scatter(test.xt[i, :, 0], test.yt[i, :, 0], s=10, c="tab:gray", label="función real")
    ax.scatter(test.xc[i, :, 0], test.yc[i, :, 0], s=30, c="k", zorder=3, label="contexto")
axes[0].legend(loc="lower left")
fig.suptitle("20 muestras condicionadas a 8 puntos (en azul)")
plt.tight_layout()

# %%
# El flujo en acción: así se transforma el ruido (t = 0) en funciones
# compatibles con el contexto (t = 1).

def trajectory(model, xc, yc, xt, steps=50, keep=(0, 10, 25, 50)):
    y = torch.randn(xt.shape[0], xt.shape[1], 1, generator=torch.Generator().manual_seed(3))
    out, h = {0: y.clone()}, 1.0 / steps
    with torch.no_grad():
        for i in range(steps):
            y = y + h * model(y + 0.5 * h * model(y, i * h, xt, xc, yc), (i + 0.5) * h, xt, xc, yc)
            if i + 1 in keep:
                out[i + 1] = y.clone()
    return out

xc, yc = test.xc[:1].repeat(6, 1, 1), test.yc[:1].repeat(6, 1, 1)
traj = trajectory(model, xc, yc, grid[:1].repeat(6, 1, 1))
fig, axes = plt.subplots(1, len(traj), figsize=(14, 3), sharey=True)
for ax, (k, y) in zip(axes, traj.items()):
    for s in range(6):
        ax.plot(grid[0, :, 0], y[s, :, 0], lw=1)
    ax.scatter(test.xc[0, :, 0], test.yc[0, :, 0], s=25, c="k", zorder=3)
    ax.set_title(f"t = {k / 50:.1f}")
fig.suptitle("Trayectoria del flujo para 6 muestras")
plt.tight_layout()

# %%
# 5. Verosimilitud
# ----------------
#
# Integrando hacia atrás (de los datos al ruido) y acumulando la divergencia
# de la velocidad se obtiene la log-verosimilitud exacta del modelo:
#
# .. math::
#
#    \log p_1(y_1) = \log p_0(y_0) - \int_0^1 \nabla \cdot v_\theta(y_t, t)\, \mathrm{d}t .
#
# La divergencia se puede calcular de forma exacta (una pasada hacia atrás por
# coordenada) o estimar con el truco de Hutchinson (una sola pasada), que es lo
# que hace el código oficial.

ev = sampler.sample(batch_size=32, num_ctx=20, num_tar=20, generator=torch.Generator().manual_seed(4))
t0 = time.time()
ll_exact = core.log_likelihood(model, ev, steps=50, exact=True).mean()
t_exact = time.time() - t0
t0 = time.time()
ll_hutch = core.log_likelihood(model, ev, steps=50, generator=torch.Generator().manual_seed(5)).mean()
t_hutch = time.time() - t0
print(f"Traza exacta: {ll_exact:.3f} ({t_exact:.1f} s) | Hutchinson: {ll_hutch:.3f} ({t_hutch:.1f} s)")

# %%
# 6. Resolvedor y número de pasos
# -------------------------------
#
# La calidad de las muestras depende de cómo se integre la ecuación
# diferencial. Con Euler y pocos pasos las muestras se dispersan poco: los
# intervalos de confianza son demasiado estrechos. Lo medimos con la
# **cobertura**: la proporción de valores reales que caen dentro del intervalo
# central del 90 % de las muestras (lo ideal es 0,90).

from flownp.eval.metrics import coverage

ev = sampler.sample(batch_size=64, num_ctx=10, num_tar=30, generator=torch.Generator().manual_seed(6))
rows = []
for method, evals in [("euler", 1), ("midpoint", 2), ("rk4", 4)]:
    for steps in (2, 5, 20):
        t0 = time.time()
        s = core.sample(model, ev.xc, ev.yc, ev.xt, num_samples=64, steps=steps, method=method,
                        generator=torch.Generator().manual_seed(7))
        rows.append((method, steps, steps * evals, coverage(s, ev.yt, 0.9).item(), time.time() - t0))

for method in ("euler", "midpoint", "rk4"):
    r = [x for x in rows if x[0] == method]
    plt.plot([x[2] for x in r], [x[3] for x in r], "o-", label=method)
plt.axhline(0.9, c="k", ls="--", lw=1)
plt.xscale("log")
plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())
plt.xticks([2, 5, 10, 20, 40, 80], ["2", "5", "10", "20", "40", "80"])
plt.xlabel("evaluaciones de la red por muestra")
plt.ylabel("cobertura del intervalo del 90 %")
plt.legend()
plt.title("Calibración frente a coste")
plt.tight_layout()

# %%
# En los experimentos completos (100 000 pasos de entrenamiento) el mejor
# compromiso es Runge-Kutta 4 con 5 a 10 pasos: la misma calidad que Euler
# con 100 pasos con unas cinco veces menos tiempo.
#
# 7. Cosenos: una distribución con varias respuestas
# --------------------------------------------------
#
# Para medir **fidelidad** (¿las muestras son funciones reales?) y
# **diversidad** (¿salen todas las respuestas posibles?) usamos cosenos
#
# .. math::
#
#    y(x) = a\cos(2\pi f x) + b\sin(2\pi f x) + \varepsilon,
#
# con :math:`f` elegida entre cuatro frecuencias y :math:`(a, b)` gaussianos.
# Su distribución condicional **exacta** se calcula en forma cerrada. Con
# pocos puntos de contexto varias frecuencias explican igual de bien los datos:
# la respuesta correcta es multimodal.

cs = CosineSampler()
b, k_true, _ = cs.sample(1, num_ctx=2, num_tar=200, generator=torch.Generator().manual_seed(8), return_params=True)
b = Batch(b.xc, b.yc, grid[:1], b.yt)
log_w, _, _ = cs.posterior(b)
exact, k = cs.exact_samples(b, 24, generator=torch.Generator().manual_seed(9))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"width_ratios": [3, 1]})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for s in range(exact.shape[0]):
    ax1.plot(grid[0, :, 0], exact[s, 0, :, 0], c=colors[int(k[s, 0])], alpha=0.6, lw=1)
ax1.scatter(b.xc[0, :, 0], b.yc[0, :, 0], s=40, c="k", zorder=3)
ax1.set_title("Muestras del posterior exacto con 2 puntos (color = frecuencia)")
ax2.bar(range(4), log_w.exp()[0].tolist(), color=colors)
ax2.set_xticks(range(4), [f"{f:g}" for f in cs.freqs.tolist()])
ax2.set_xlabel("frecuencia (ciclos por unidad)")
ax2.set_title("Probabilidad exacta")
plt.tight_layout()

# %%
# Un buen modelo debe generar ondas de las frecuencias plausibles y en esa
# proporción. Las medidas del paquete (``flownp.evaluation.evaluate_cosine``)
# clasifican cada muestra ajustando un coseno de cada frecuencia:
#
# * **fidelidad**: proporción de muestras que son de verdad un coseno;
# * **diversidad**: distancia entre la proporción de muestras de cada
#   frecuencia y la probabilidad exacta.
#
# En nuestros experimentos, FlowNP entrenado desde cero con estos datos se
# queda atascado en una meseta en la que ignora el contexto; partiendo de un
# modelo entrenado con procesos gaussianos aprende bien, y con 10 o 20 puntos
# de contexto el 94 % y el 98 % de sus muestras son ondas de la frecuencia
# correcta. El artículo del proyecto recoge todos los resultados.
