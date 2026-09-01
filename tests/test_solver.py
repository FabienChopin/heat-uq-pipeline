"""Tests for heatuq.solver: input validation and physical sanity checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from heatuq import compute_stable_dt, solve_heat_equation_2d


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"alpha": -1.0}, "alpha must be positive"),
        ({"sigma": 0.0}, "sigma must be positive"),
        ({"nx": 2}, "nx and ny must be"),
        ({"ny": 2}, "nx and ny must be"),
        ({"lx": 0.0}, "lx and ly must be positive"),
        ({"ly": -1.0}, "lx and ly must be positive"),
        ({"x0": 0.0}, "source center"),
        ({"x0": 1.0}, "source center"),
        ({"y0": -0.1}, "source center"),
        ({"t_end": 0.0}, "t_end must be positive"),
    ],
)
def test_invalid_inputs_raise(
    default_kwargs: dict[str, Any], overrides: dict[str, float], match: str
) -> None:
    kwargs = {**default_kwargs, **overrides}
    with pytest.raises(ValueError, match=match):
        solve_heat_equation_2d(**kwargs)


def test_explicit_dt_too_large_raises(default_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="exceeds the stability limit"):
        solve_heat_equation_2d(**default_kwargs, dt=1.0)


def test_negative_dt_raises(default_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        solve_heat_equation_2d(**default_kwargs, dt=-0.001)


def test_valid_explicit_dt_is_used(default_kwargs: dict[str, Any]) -> None:
    dx = default_kwargs["lx"] / (default_kwargs["nx"] - 1)
    dy = default_kwargs["ly"] / (default_kwargs["ny"] - 1)
    dt_max = compute_stable_dt(default_kwargs["alpha"], dx, dy, safety_factor=1.0)
    small_dt = dt_max / 2
    times, fields = solve_heat_equation_2d(**default_kwargs, dt=small_dt)
    assert times.shape == (2,)
    assert fields.shape == (2, default_kwargs["ny"], default_kwargs["nx"])


def test_bad_safety_factor_raises(default_kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="safety_factor must be in"):
        solve_heat_equation_2d(**default_kwargs, safety_factor=1.5)


def test_save_times_outside_range_raises(default_kwargs: dict[str, Any]) -> None:
    bad_save_times = (0.0, default_kwargs["t_end"] * 2)
    with pytest.raises(ValueError, match="save_times must all lie in"):
        solve_heat_equation_2d(**default_kwargs, save_times=bad_save_times)


def test_default_save_times_returns_start_and_end(default_kwargs: dict[str, Any]) -> None:
    times, fields = solve_heat_equation_2d(**default_kwargs)
    assert times.shape == (2,)
    assert fields.shape == (2, default_kwargs["ny"], default_kwargs["nx"])
    assert times[0] == 0.0
    assert math.isclose(times[1], default_kwargs["t_end"])


def test_save_times_are_hit_exactly(default_kwargs: dict[str, Any]) -> None:
    save_times = (0.0, default_kwargs["t_end"] / 2, default_kwargs["t_end"])
    times, fields = solve_heat_equation_2d(**default_kwargs, save_times=save_times)
    np.testing.assert_allclose(times, save_times)
    assert fields.shape == (3, default_kwargs["ny"], default_kwargs["nx"])


def test_boundaries_stay_zero(default_kwargs: dict[str, Any]) -> None:
    save_times = (0.0, default_kwargs["t_end"] / 2, default_kwargs["t_end"])
    _, fields = solve_heat_equation_2d(**default_kwargs, save_times=save_times)
    assert np.all(fields[:, 0, :] == 0.0)
    assert np.all(fields[:, -1, :] == 0.0)
    assert np.all(fields[:, :, 0] == 0.0)
    assert np.all(fields[:, :, -1] == 0.0)


def test_zero_amplitude_stays_zero(default_kwargs: dict[str, Any]) -> None:
    kwargs = {**default_kwargs, "amplitude": 0.0}
    _, fields = solve_heat_equation_2d(**kwargs)
    assert np.all(fields == 0.0)


def test_positive_amplitude_gives_positive_interior(default_kwargs: dict[str, Any]) -> None:
    _, fields = solve_heat_equation_2d(**default_kwargs)
    final = fields[-1]
    assert np.all(final[1:-1, 1:-1] > 0.0)


def test_symmetric_source_gives_symmetric_field() -> None:
    _, fields = solve_heat_equation_2d(
        alpha=1.0,
        amplitude=1.0,
        x0=0.5,
        y0=0.5,
        sigma=0.15,
        nx=41,
        ny=41,
        lx=1.0,
        ly=1.0,
        t_end=0.02,
    )
    field = fields[-1]
    np.testing.assert_allclose(field, field.T, atol=1e-10)


def test_compute_stable_dt_matches_closed_form_for_square_grid() -> None:
    h, alpha = 0.02, 2.0
    expected = h**2 / (4.0 * alpha)
    assert math.isclose(compute_stable_dt(alpha, h, h, safety_factor=1.0), expected)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0, "dx": 0.1, "dy": 0.1}, "alpha must be positive"),
        ({"alpha": 1.0, "dx": 0.0, "dy": 0.1}, "dx and dy must be positive"),
        (
            {"alpha": 1.0, "dx": 0.1, "dy": 0.1, "safety_factor": 0.0},
            "safety_factor must be in",
        ),
    ],
)
def test_compute_stable_dt_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        compute_stable_dt(**kwargs)
