"""Cheap physical-invariant checks, independent of the convergence-order tests.

These don't need an exact solution: they check properties the solver must satisfy
by construction of the PDE/BCs, so they catch a different class of bugs (sign
errors, axis swaps, instability) than order-of-accuracy verification does. All
exercise the published API as-is; solver.py is untouched.
"""

from __future__ import annotations

import numpy as np

from heatuq import compute_stable_dt, solve_heat_equation_2d

ALPHA = 0.05
AMPLITUDE = 5.0
X0 = Y0 = 0.5
SIGMA = 0.15
T_END = 0.02


def test_positivity() -> None:
    # Maximum principle: source >= 0, boundary = 0, T(., 0) = 0 => T(x, y, t) >= 0
    # everywhere for all t. The FTCS update is also a convex combination of
    # non-negative neighbour values plus a non-negative source term whenever dt is
    # within the stability limit (compute_stable_dt's safety_factor < 1 guarantees
    # this), so the discrete solution should honour this exactly, not just
    # approximately.
    nx = ny = 41
    dx = 1.0 / (nx - 1)
    dt = compute_stable_dt(ALPHA, dx, dx)  # default safety_factor=0.9, as in the README

    T = solve_heat_equation_2d(ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=nx, ny=ny, t_end=T_END, dt=dt)

    assert float(np.min(T)) >= -1e-9, f"negative temperature found: min={float(np.min(T))}"


def test_zero_source_stays_zero() -> None:
    # amplitude=0 => source is identically 0 everywhere. With T(., 0) = 0 and a
    # zero Dirichlet boundary, T=0 is an exact fixed point of the update
    # (laplacian(0) = 0), independent of alpha, dt, resolution, or t_end.
    nx = ny = 31
    dx = 1.0 / (nx - 1)
    dt = compute_stable_dt(ALPHA, dx, dx)

    T = solve_heat_equation_2d(ALPHA, amplitude=0.0, x0=X0, y0=Y0, sigma=SIGMA, nx=nx, ny=ny, t_end=T_END, dt=dt)

    assert np.array_equal(np.asarray(T), np.zeros((ny, nx)))


def test_symmetry() -> None:
    # Square domain (lx=ly), source centered (x0=y0=lx/2), square grid (nx=ny):
    # the whole problem (Laplacian, BCs, source) is invariant under mirroring
    # either axis about the centerline, and under swapping x and y. A correct
    # solver must reproduce all three symmetries on its own output.
    nx = ny = 41
    dx = 1.0 / (nx - 1)
    dt = compute_stable_dt(ALPHA, dx, dx)

    T = np.asarray(
        solve_heat_equation_2d(ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=nx, ny=ny, t_end=T_END, dt=dt)
    )

    assert np.allclose(T, T[:, ::-1], atol=1e-6), "not symmetric under x-mirror"
    assert np.allclose(T, T[::-1, :], atol=1e-6), "not symmetric under y-mirror"
    assert np.allclose(T, T.T, atol=1e-6), "not symmetric under x/y swap (transpose)"
