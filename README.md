# flownp-repro

Reimplementación y estudio de **Flow Matching Neural Processes** (FlowNP; Abu Hamad y Rosenbaum, NeurIPS 2025,
[arXiv:2512.23853](https://arxiv.org/abs/2512.23853)). Proyecto de la asignatura de Inteligencia Artificial
Generativa del Máster Universitario en Inteligencia Artificial (Universidad Autónoma de Madrid).

**Tutorial:** https://samuel13032004.github.io/flownp-repro/tutoriales/plot_flownp_tutorial.html
(también como cuaderno en `notebooks/tutorial_flownp.ipynb`).

Memoria: [paper/flownp_report.pdf](paper/flownp_report.pdf)

Autores: Samuel Déniz, Carlos Enrique Pereda y Yaser Jafar.

## Qué hay aquí

1. **Reproducción** de la tabla 1 del artículo con el código oficial (`danrsm/flowNP`, commit `f3403b6`).
2. **Reimplementación** propia de FlowNP en `src/flownp`, equivalente a la oficial (mismas velocidades con los
   mismos pesos y misma verosimilitud: 1,712 ± 0,003 frente a 1,716 ± 0,004).
3. **Estudio nuevo**: calibración de la incertidumbre, coste del muestreo según el resolvedor, y un conjunto de
   cosenos con distribución exacta conocida para medir fidelidad y diversidad, comparando con un modelo de difusión.

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/flownp/` | datos, modelo, flujo (pérdida, resolvedores, verosimilitud), medidas, entrenamiento y evaluación |
| `scripts/` | `train.py` y `evaluate.py` |
| `tests/` | 13 pruebas (`pytest -q`, menos de un minuto en procesador) |
| `notebooks/` | cuadernos de Colab para lanzar todos los experimentos y el tutorial |
| `tutorials/`, `docs/` | tutorial con Sphinx-Gallery, publicado con GitHub Actions |
| `flownp_runs/` | resultados de todas las ejecuciones: registros, métricas y puntos de control |

## Uso

```bash
pip install -e . pytest attrdictionary pyyaml
pytest -q tests
python scripts/train.py --run runs/rbf/flownp/s0 --model flownp --data gp:rbf --seed 0
```

Los modelos oficiales necesitan el código oficial clonado aparte, porque no declara licencia:
`git clone https://github.com/danrsm/flowNP third_party/flowNP` y la opción `--official-code third_party/flowNP`.
