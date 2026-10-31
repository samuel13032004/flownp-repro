import argparse

from flownp.training import train

p = argparse.ArgumentParser()
p.add_argument("--run", required=True, help="carpeta de la ejecución")
p.add_argument("--model", default="flownp", help="flownp u official:<fnp|cnp|anp|tnpa|ndp>")
p.add_argument("--data", default="gp:rbf", help="gp:rbf, gp:matern, gp:periodic o cosine")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--time-per-function", action="store_true", help="un tiempo de flujo por función (por defecto, por punto)")
p.add_argument("--official-code", default=None)
p.add_argument("--device", default=None)
p.add_argument("--init", default=None, help="punto de control del que tomar los pesos iniciales")
a = p.parse_args()
path = train(a.run, a.model, a.data, a.seed, a.steps, time_per_point=not a.time_per_function,
             official_code=a.official_code, device=a.device, init=a.init)
print("terminado:", path)
