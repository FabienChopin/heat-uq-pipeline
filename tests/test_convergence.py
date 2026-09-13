"""Order-of-accuracy verification via self-convergence (Richardson).

The solver's real problem (Gaussian source, Dirichlet boundary) has no closed-form
exact solution, so instead of comparing to a known reference we compare successive
solutions on nested grids / nested timesteps to each other:

    T_h - T_exact ~ C * h**p  =>  T_h - T_(h/2) ~ C * h**p * (1 - 2**-p)

so the observed order is p = log2(||T_h - T_(h/2)|| / ||T_(h/2) - T_(h/4)||),
independent of the (unknown) exact solution. This exercises the solver exactly as
published (no changes to solver.py, no synthetic source).

Two separate studies are needed because the scheme is 2nd order in space (5-point
central-difference Laplacian) but only 1st order in time (explicit forward Euler):
holding dt ~ dx**2 (as the deleted pre-JAX-rewrite convergence test did, by
re-picking dt per resolution from compute_stable_dt) would hide the 1st-order time
error behind the 2nd-order space error and make the combined scheme look 2nd order
overall. So here dt is held fixed while nx/ny vary (spatial study), and nx/ny are
held fixed while dt varies (temporal study).
"""

from __future__ import annotations

import math

import numpy as np

from heatuq import compute_stable_dt, solve_heat_equation_2d

ALPHA = 0.05
AMPLITUDE = 5.0
X0 = Y0 = 0.5
SIGMA = 0.15
T_END = 0.01


def _l2_error(a: np.ndarray, b: np.ndarray) -> float:
    interior = slice(1, -1)
    diff = np.asarray(a)[interior, interior] - np.asarray(b)[interior, interior]
    return float(np.sqrt(np.mean(diff**2)))


def _observed_order(errors: list[float]) -> list[float]:
    return [math.log2(e1 / e2) for e1, e2 in zip(errors, errors[1:])]


def test_spatial_order() -> None:
    # Nested grids: nx_{k+1} = 2 * (nx_k - 1) + 1, so every point of a coarser
    # grid coincides exactly with a point of the next finer one (T[::2, ::2]).
    resolutions = (21, 41, 81, 161)
    # dt must be stable at the *finest* grid (the stability limit tightens as dx
    # shrinks) and is then reused unchanged across all resolutions, so temporal
    # error stays ~constant and doesn't contaminate the measured spatial order.
    dx_finest = 1.0 / (resolutions[-1] - 1)
    dt = 0.1 * compute_stable_dt(ALPHA, dx_finest, dx_finest)
    n_steps = round(T_END / dt)
    dt = T_END / n_steps  # exact division, avoids n_steps=int(t_end/dt) truncation drift

    fields = [
        solve_heat_equation_2d(
            ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=n, ny=n, t_end=T_END, dt=dt
        )
        for n in resolutions
    ]
    errors = [
        _l2_error(fields[i], np.asarray(fields[i + 1])[::2, ::2])
        for i in range(len(resolutions) - 1)
    ]

    orders = _observed_order(errors)
    # Only the finest pair is checked tightly: at nx=21 the Gaussian source is
    # barely resolved (~3 points per sigma), so higher-order error terms are
    # still non-negligible there and the observed order undershoots 2 (empirically
    # ~1.77) before settling in as the grid refines. The coarse-pair estimate is
    # still asserted, loosely, as a regression net against a badly broken scheme.
    assert orders[0] > 1.5, f"spatial orders={orders}, errors={errors}"
    assert 1.8 <= orders[-1] <= 2.2, f"spatial orders={orders}, errors={errors}"


def test_temporal_order() -> None:
    nx = ny = 81
    dx = 1.0 / (nx - 1)
    dt_max = compute_stable_dt(ALPHA, dx, dx, safety_factor=1.0)

    # dt sequence by successive halving, each an exact divisor of T_END, all below
    # the stability limit at this grid.
    n_steps0 = 20
    dt0 = T_END / n_steps0
    assert dt0 < dt_max, "base dt violates the FTCS stability limit at this resolution"
    dts = [dt0 / 2**k for k in range(4)]

    fields = [
        solve_heat_equation_2d(ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=nx, ny=ny, t_end=T_END, dt=dt)
        for dt in dts
    ]
    errors = [_l2_error(fields[i], fields[i + 1]) for i in range(len(dts) - 1)]

    orders = _observed_order(errors)
    assert all(0.8 <= p <= 1.2 for p in orders[-2:]), f"temporal orders={orders}, errors={errors}"
