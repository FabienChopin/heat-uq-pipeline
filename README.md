# heat-uq-pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 2D transient heat-diffusion solver, differentiable with respect to its physical
parameters. This is the first phase of a larger pipeline: future phases will use
this solver to generate Monte Carlo datasets and train an MLP surrogate for
Bayesian calibration of the physical parameters. **This repo currently ships only
the solver.**

## Equation

```
dT/dt = alpha * laplacian(T) + S(x, y)      on [0, lx] x [0, ly]
T = 0 on the boundary, T(x, y, 0) = 0
S(x, y) = amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
```

Five physical parameters are exposed for later calibration: `alpha` (thermal
diffusivity), `amplitude`, `x0`, `y0`, and `sigma` (the Gaussian source term).

## Method

Explicit forward-Euler time-stepping with a 5-point Laplacian stencil, written in
`jax.numpy` so the output is differentiable w.r.t. the 5 physical parameters via
`jax.jacfwd`. Use forward-mode (`jacfwd`) specifically: there are only 5 inputs
but `nx * ny` outputs, and reverse-mode (`jax.jacobian`/`jax.grad`) would be far
slower here since its cost scales with the number of outputs, not inputs.

`dt` must be passed explicitly — it can't be auto-selected from `alpha` inside the
function, because `alpha` may itself be a traced (differentiated) value, and the
number of time steps needs to be fixed before running. Use `compute_stable_dt`
with a concrete `alpha` beforehand to pick a safe `dt`.

## Limits for UQ

Fixing `dt` once (rather than per-`alpha`) keeps the solver traceable, but it
also means discretization error isn't uniform across an `alpha` sweep — it
shows up as numerical noise in results and derivatives, worst near the top of
the explored range. Not addressed properly yet. In practice, `dt` should be
fixed once per full study — dataset generation, surrogate training, and
calibration for a given `alpha` prior — rather than re-picked per batch.

## Install

```bash
git clone https://github.com/FabienChopin/heat-uq-pipeline.git
cd heat-uq-pipeline
uv sync
```

## Usage

```python
import jax
from heatuq import compute_stable_dt, solve_heat_equation_2d

dx = dy = 1.0 / 100  # nx=ny=101 by default
dt = compute_stable_dt(alpha=0.01, dx=dx, dy=dy)

T = solve_heat_equation_2d(alpha=0.01, amplitude=5.0, x0=0.5, y0=0.5, sigma=0.05, t_end=0.5, dt=dt)

# Derivatives of the whole field w.r.t. the 5 physical parameters
# (jacfwd, not jacobian/jacrev: 5 inputs but nx*ny outputs):
jac = jax.jacfwd(solve_heat_equation_2d, argnums=(0, 1, 2, 3, 4))
dT_dalpha, dT_damplitude, dT_dx0, dT_dy0, dT_dsigma = jac(
    0.01, 5.0, 0.5, 0.5, 0.05, t_end=0.5, dt=dt
)
```

## Roadmap

- Tests, CI, and the rest of the "professional" tooling, once the API has settled
- Monte Carlo dataset generation over the 5 physical parameters
- MLP surrogate model training
- Bayesian calibration of the physical parameters
- Heterogeneous (spatially-varying) thermal conductivity

## License

[MIT](LICENSE)
