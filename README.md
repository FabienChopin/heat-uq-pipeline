# heat-uq-pipeline

[![CI](https://github.com/FabienChopin/heat-uq-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/FabienChopin/heat-uq-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 2D transient heat-diffusion finite-difference solver. This is the first phase of a
larger pipeline: future phases will use this solver to generate Monte Carlo datasets
and train an MLP surrogate for Bayesian calibration of the physical parameters. **This
repo currently ships only the solver.**

## Equation

```
dT/dt = alpha * laplacian(T) + S(x, y)      on [0, lx] x [0, ly]
T = 0 on the boundary, T(x, y, 0) = 0
S(x, y) = amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
```

Five physical parameters are exposed for later calibration: `alpha` (thermal
diffusivity), `amplitude`, `x0`, `y0`, and `sigma` (the Gaussian source term).

## Method

Explicit forward-Euler time-stepping with a 5-point Laplacian stencil, fully
vectorized with numpy. The timestep is auto-selected from the CFL stability limit
unless you supply one explicitly (an unstable explicit `dt` raises `ValueError`).

## Install

```bash
git clone https://github.com/FabienChopin/heat-uq-pipeline.git
cd heat-uq-pipeline
uv sync --all-groups
```

## Usage

```python
from heatuq import solve_heat_equation_2d

times, T = solve_heat_equation_2d(
    alpha=0.01,
    amplitude=5.0,
    x0=0.5,
    y0=0.5,
    sigma=0.05,
    nx=101,
    ny=101,
    t_end=0.5,
    save_times=(0.1, 0.25, 0.5),
)
print(times.shape, T.shape)  # (3,) (3, 101, 101)
```

`T[k]` is the temperature field at `times[k]`, shape `(ny, nx)`, with `T[k][j, i]`
corresponding to grid point `(x[i], y[j])`.

## Development

```bash
uv run pytest        # tests, including a manufactured-solution convergence check
uv run ruff check .  # lint
uv run mypy src tests  # type check
```

## Roadmap

- Monte Carlo dataset generation over the 5 physical parameters
- MLP surrogate model training
- Bayesian calibration of the physical parameters
- Heterogeneous (spatially-varying) thermal conductivity

## License

[MIT](LICENSE)
