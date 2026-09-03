"""2D transient heat-diffusion solver, differentiable w.r.t. its physical parameters.

Solves dT/dt = alpha * laplacian(T) + S(x, y) on a rectangular domain with
fixed Dirichlet boundaries (T=0) and a time-constant Gaussian source, using
an explicit forward-Euler, 5-point-stencil finite-difference scheme written
in jax.numpy so the output is differentiable (via jax.jacobian/jax.grad)
with respect to alpha, amplitude, x0, y0, sigma.
"""

from __future__ import annotations

import jax.numpy as jnp


def compute_stable_dt(alpha: float, dx: float, dy: float, safety_factor: float = 0.9) -> float:
    """Largest explicit-Euler timestep for which the 2D FTCS scheme is stable.

    Call this with a concrete alpha (representative of the range you plan to
    explore) to pick a safe `dt` before calling solve_heat_equation_2d.
    """
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    return safety_factor * dt_max


def solve_heat_equation_2d(
    alpha: float,
    amplitude: float,
    x0: float,
    y0: float,
    sigma: float,
    *,
    nx: int = 101,
    ny: int = 101,
    lx: float = 1.0,
    ly: float = 1.0,
    t_end: float = 1.0,
    dt: float,
) -> jnp.ndarray:
    """Return T(x, y, t_end), shape (ny, nx).

    Differentiable w.r.t. alpha, amplitude, x0, y0, sigma via jax.jacfwd
    (the output is a full field, not a scalar, so use a jacobian rather
    than grad; use jacfwd specifically — forward-mode — since there are
    only 5 inputs but ~nx*ny outputs, and reverse-mode jax.jacobian would
    be far slower here). `nx`, `ny`, `lx`, `ly`, `t_end`, and `dt` are
    plain Python values fixed at call time, not part of the differentiated
    computation.
    """
    x = jnp.linspace(0.0, lx, nx)
    y = jnp.linspace(0.0, ly, ny)
    dx, dy = lx / (nx - 1), ly / (ny - 1)
    xx, yy = jnp.meshgrid(x, y)
    source = amplitude * jnp.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))

    T = jnp.zeros((ny, nx))
    n_steps = int(t_end / dt)
    for _ in range(n_steps):
        laplacian = (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy**2
            + (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2
        )
        T = T.at[1:-1, 1:-1].set(T[1:-1, 1:-1] + dt * (alpha * laplacian + source[1:-1, 1:-1]))
    return T
