"""2D transient heat-diffusion finite-difference solver.

Solves the PDE

    dT/dt = alpha * laplacian(T) + S(x, y)

on a rectangular domain with fixed Dirichlet boundaries (T=0) and a
time-constant Gaussian source term, using an explicit forward-Euler,
5-point-stencil finite-difference scheme, fully vectorized with numpy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = ["compute_stable_dt", "solve_heat_equation_2d"]


def compute_stable_dt(alpha: float, dx: float, dy: float, safety_factor: float = 1.0) -> float:
    """Largest explicit-Euler timestep for which the 2D FTCS scheme is stable.

    dt_max = 1 / (2 * alpha * (1/dx**2 + 1/dy**2)), scaled by ``safety_factor``.

    Parameters
    ----------
    alpha : thermal diffusivity, must be positive.
    dx, dy : grid spacing along x and y, each must be positive.
    safety_factor : fraction of the raw stability limit to return, in (0, 1].

    Returns
    -------
    The stable timestep.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"dx and dy must be positive, got dx={dx}, dy={dy}")
    if not (0 < safety_factor <= 1):
        raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
    dt_max = 1.0 / (2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2))
    return safety_factor * dt_max


def _gaussian_source(
    x: FloatArray, y: FloatArray, amplitude: float, x0: float, y0: float, sigma: float
) -> FloatArray:
    """Build S(x, y) = amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))."""
    xx, yy = np.meshgrid(x, y)
    return amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


def _step(
    field: FloatArray, alpha: float, source: FloatArray, dx: float, dy: float, dt: float
) -> FloatArray:
    """One explicit-Euler update of the interior, then re-apply T=0 boundaries."""
    new_field = field.copy()
    laplacian = (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dy**2 + (
        field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]
    ) / dx**2
    new_field[1:-1, 1:-1] = field[1:-1, 1:-1] + dt * (alpha * laplacian + source[1:-1, 1:-1])
    new_field[0, :] = 0.0
    new_field[-1, :] = 0.0
    new_field[:, 0] = 0.0
    new_field[:, -1] = 0.0
    return new_field


def _solve_with_source(
    dx: float,
    dy: float,
    alpha: float,
    source: FloatArray,
    dt: float,
    t_end: float,
    save_times: Sequence[float],
) -> tuple[FloatArray, FloatArray]:
    """Time-march the explicit scheme from T=0, snapshotting exactly at ``save_times``.

    ``save_times`` must be sorted and lie within [0, t_end]. ``source`` sets the
    grid shape (ny, nx) implicitly and may be any time-constant array, not
    necessarily Gaussian.
    """
    ny, nx = source.shape
    field = np.zeros((ny, nx), dtype=np.float64)

    times_out = np.asarray(save_times, dtype=np.float64)
    fields_out = np.empty((len(save_times), ny, nx), dtype=np.float64)

    t = 0.0
    save_idx = 0
    while save_idx < len(save_times) and save_times[save_idx] <= t:
        fields_out[save_idx] = field
        save_idx += 1

    while save_idx < len(save_times):
        target = save_times[save_idx]
        while t < target:
            step_dt = min(dt, target - t)
            field = _step(field, alpha, source, dx, dy, step_dt)
            t += step_dt
        fields_out[save_idx] = field
        save_idx += 1

    return times_out, fields_out


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
    dt: float | None = None,
    safety_factor: float = 0.9,
    save_times: Sequence[float] | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Solve dT/dt = alpha * laplacian(T) + S(x, y) on [0, lx] x [0, ly].

    Fixed Dirichlet boundaries (T=0 on all edges) and initial condition T=0
    everywhere. S is a time-constant Gaussian bump:

        S(x, y) = amplitude * exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

    Parameters
    ----------
    alpha : thermal diffusivity, must be positive.
    amplitude : source peak amplitude.
    x0, y0 : source center, must lie strictly inside the domain.
    sigma : source width, must be positive.
    nx, ny : number of grid points along x and y, each >= 3.
    lx, ly : domain size along x and y, each > 0.
    t_end : final simulation time, must be positive.
    dt : fixed timestep. If None, the largest CFL-stable timestep (scaled by
        ``safety_factor``) is selected automatically. An explicit ``dt`` that
        exceeds the stability limit raises ``ValueError``.
    safety_factor : fraction of the CFL-stable limit to use when ``dt`` is
        None, in (0, 1].
    save_times : times at which to snapshot the field, each in [0, t_end]. If
        None, only t=0 and t=t_end are returned.

    Returns
    -------
    times : shape (n_saved,).
    fields : shape (n_saved, ny, nx); ``fields[k]`` is T at ``times[k]``, with
        ``fields[k][j, i]`` corresponding to grid point ``(x[i], y[j])``.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if nx < 3 or ny < 3:
        raise ValueError(f"nx and ny must be >= 3, got nx={nx}, ny={ny}")
    if lx <= 0 or ly <= 0:
        raise ValueError(f"lx and ly must be positive, got lx={lx}, ly={ly}")
    if not (0 < x0 < lx) or not (0 < y0 < ly):
        raise ValueError(
            f"source center ({x0}, {y0}) must lie strictly inside the domain [0, {lx}] x [0, {ly}]"
        )
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")

    resolved_save_times = (0.0, t_end) if save_times is None else tuple(save_times)
    save_times_sorted = tuple(sorted(resolved_save_times))
    if save_times_sorted[0] < 0 or save_times_sorted[-1] > t_end:
        raise ValueError(f"save_times must all lie in [0, t_end={t_end}], got {save_times_sorted}")

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    dt_max = compute_stable_dt(alpha, dx, dy, safety_factor=1.0)
    if dt is None:
        if not (0 < safety_factor <= 1):
            raise ValueError(f"safety_factor must be in (0, 1], got {safety_factor}")
        dt_used = safety_factor * dt_max
    else:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if dt > dt_max * (1 + 1e-9):
            raise ValueError(
                f"dt={dt} exceeds the stability limit dt_max={dt_max} for "
                f"alpha={alpha}, dx={dx}, dy={dy}; leave dt=None to auto-select "
                "a stable timestep."
            )
        dt_used = dt

    source = _gaussian_source(x, y, amplitude, x0, y0, sigma)
    return _solve_with_source(dx, dy, alpha, source, dt_used, t_end, save_times_sorted)
