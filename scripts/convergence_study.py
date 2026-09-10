"""Order-of-accuracy study for the 2D heat solver, with a log-log convergence plot.

Companion to tests/test_convergence.py (same self-convergence / Richardson approach —
see that file's docstring for the method), but with more refinement levels for a
clearer picture, and it produces a figure instead of just asserting a tolerance band.
Not run in CI: the extra levels make it too slow for a fast test suite, and its
point is to *show* the convergence behaviour, not just gate it.

Usage: uv run python scripts/convergence_study.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from heatuq import compute_stable_dt, solve_heat_equation_2d

ALPHA = 0.05
AMPLITUDE = 5.0
X0 = Y0 = 0.5
SIGMA = 0.15
T_END = 0.01

OUTPUT_DIR = Path(__file__).parent / "output"


def _l2_error(a: np.ndarray, b: np.ndarray) -> float:
    interior = slice(1, -1)
    diff = np.asarray(a)[interior, interior] - np.asarray(b)[interior, interior]
    return float(np.sqrt(np.mean(diff**2)))


def _linf_error(a: np.ndarray, b: np.ndarray) -> float:
    interior = slice(1, -1)
    diff = np.asarray(a)[interior, interior] - np.asarray(b)[interior, interior]
    return float(np.max(np.abs(diff)))


def _observed_orders(errors: list[float]) -> list[float]:
    return [math.log2(e1 / e2) for e1, e2 in zip(errors, errors[1:])]


def _print_table(header: str, steps: list[float], errors_l2: list[float], errors_linf: list[float]) -> None:
    orders = _observed_orders(errors_l2)
    print(f"\n{header}")
    print(f"{'step':>12}  {'L2 error':>12}  {'Linf error':>12}  {'order (L2)':>12}")
    for i, (step, e2, einf) in enumerate(zip(steps, errors_l2, errors_linf)):
        order_str = f"{orders[i - 1]:12.3f}" if i > 0 else " " * 12
        print(f"{step:12.6g}  {e2:12.6g}  {einf:12.6g}  {order_str}")


def spatial_study(resolutions: tuple[int, ...]) -> tuple[list[float], list[float], list[float]]:
    # Nested grids: nx_{k+1} = 2 * (nx_k - 1) + 1, so every point of a coarser grid
    # coincides exactly with a point of the next finer one (T[::2, ::2]).
    dx_finest = 1.0 / (resolutions[-1] - 1)
    dt = 0.1 * compute_stable_dt(ALPHA, dx_finest, dx_finest)
    n_steps = round(T_END / dt)
    dt = T_END / n_steps  # exact division, avoids n_steps=int(t_end/dt) truncation drift

    fields = [
        solve_heat_equation_2d(ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=n, ny=n, t_end=T_END, dt=dt)
        for n in resolutions
    ]
    dxs = [1.0 / (n - 1) for n in resolutions[:-1]]
    errors_l2 = [
        _l2_error(fields[i], np.asarray(fields[i + 1])[::2, ::2]) for i in range(len(resolutions) - 1)
    ]
    errors_linf = [
        _linf_error(fields[i], np.asarray(fields[i + 1])[::2, ::2]) for i in range(len(resolutions) - 1)
    ]
    _print_table("Spatial convergence (fixed dt, varying nx=ny)", dxs, errors_l2, errors_linf)
    return dxs, errors_l2, errors_linf


def temporal_study(nx: int, n_levels: int) -> tuple[list[float], list[float], list[float]]:
    dx = 1.0 / (nx - 1)
    dt_max = compute_stable_dt(ALPHA, dx, dx, safety_factor=1.0)
    n_steps0 = 20
    dt0 = T_END / n_steps0
    assert dt0 < dt_max, "base dt violates the FTCS stability limit at this resolution"
    dts = [dt0 / 2**k for k in range(n_levels)]

    fields = [
        solve_heat_equation_2d(ALPHA, AMPLITUDE, X0, Y0, SIGMA, nx=nx, ny=nx, t_end=T_END, dt=dt)
        for dt in dts
    ]
    errors_l2 = [_l2_error(fields[i], fields[i + 1]) for i in range(len(dts) - 1)]
    errors_linf = [_linf_error(fields[i], fields[i + 1]) for i in range(len(dts) - 1)]
    _print_table(f"Temporal convergence (fixed {nx}x{nx} grid, varying dt)", dts[:-1], errors_l2, errors_linf)
    return dts[:-1], errors_l2, errors_linf


def _plot_panel(ax: plt.Axes, steps: list[float], errors: list[float], expected_order: int, title: str) -> None:
    ax.loglog(steps, errors, "o-", label="observed L2 error", color="#2f6f9f")

    ref = [errors[0] * (s / steps[0]) ** expected_order for s in steps]
    ax.loglog(steps, ref, "--", label=f"order {expected_order} reference", color="#999999")

    ax.set_xlabel("dx" if "spatial" in title.lower() else "dt")
    ax.set_ylabel("L2 error (successive-refinement)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Label only the actual data points instead of matplotlib's dense default
    # log-scale ticks, which overlap and become unreadable at these magnitudes.
    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s:.2g}" for s in steps], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(plt.NullLocator())


def main() -> None:
    dxs, sp_l2, _ = spatial_study(resolutions=(11, 21, 41, 81, 161, 321))
    dts, tmp_l2, _ = temporal_study(nx=81, n_levels=6)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, (ax_spatial, ax_temporal) = plt.subplots(1, 2, figsize=(11, 4.5))
    _plot_panel(ax_spatial, dxs, sp_l2, expected_order=2, title="Spatial convergence")
    _plot_panel(ax_temporal, dts, tmp_l2, expected_order=1, title="Temporal convergence")
    fig.suptitle("heat-uq-pipeline solver: order-of-accuracy verification (self-convergence)")
    fig.tight_layout()

    out_path = OUTPUT_DIR / "convergence_study.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
