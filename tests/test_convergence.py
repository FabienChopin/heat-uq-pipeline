"""Numerical-accuracy test via the method of manufactured solutions (MMS).

Verifies the explicit finite-difference scheme achieves its expected ~2nd
order spatial convergence, using a manufactured solution that satisfies the
PDE, the Dirichlet boundary condition, and the T=0 initial condition exactly.

Manufactured field:  T*(x, y) = sin(pi x) sin(pi y)          (zero on all edges)
Steady source:        S(x, y) = 2 alpha pi^2 sin(pi x) sin(pi y)
                                 (from alpha * laplacian(T*) + S = 0)
Exact transient solution with T(x, y, 0) = 0 and this constant S:
    T_exact(x, y, t) = T*(x, y) * (1 - exp(-2 alpha pi^2 t))
(verified by direct substitution into dT/dt = alpha * laplacian(T) + S)
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from heatuq.solver import _solve_with_source, compute_stable_dt

ALPHA = 1.0


def _manufactured_exact(
    x: NDArray[np.float64], y: NDArray[np.float64], t: float
) -> NDArray[np.float64]:
    xx, yy = np.meshgrid(x, y)
    t_star = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    result: NDArray[np.float64] = t_star * (1.0 - np.exp(-2.0 * ALPHA * np.pi**2 * t))
    return result


def _manufactured_source(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    xx, yy = np.meshgrid(x, y)
    return 2.0 * ALPHA * np.pi**2 * np.sin(np.pi * xx) * np.sin(np.pi * yy)


def _l2_error(n: int, t_end: float) -> float:
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    source = _manufactured_source(x, y)
    dt = compute_stable_dt(ALPHA, dx, dy, safety_factor=0.5)

    _, fields = _solve_with_source(dx, dy, ALPHA, source, dt, t_end, (t_end,))
    numerical = fields[-1]
    exact = _manufactured_exact(x, y, t_end)

    interior = slice(1, -1)
    error = numerical[interior, interior] - exact[interior, interior]
    return float(np.sqrt(np.mean(error**2)))


def test_scheme_converges_at_second_order() -> None:
    t_end = 0.05
    resolutions = (21, 41, 81, 161)
    errors = [_l2_error(n, t_end) for n in resolutions]

    assert all(e2 < e1 for e1, e2 in pairwise(errors))
    assert errors[-1] < 1e-3

    orders = [math.log(e1 / e2) / math.log(2.0) for e1, e2 in pairwise(errors)]
    for order in orders[-2:]:
        assert 1.8 <= order <= 2.2
