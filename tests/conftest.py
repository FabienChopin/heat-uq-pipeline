"""Shared pytest fixtures."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def default_kwargs() -> dict[str, Any]:
    """A small, valid default parameter set for solve_heat_equation_2d."""
    return {
        "alpha": 1.0,
        "amplitude": 1.0,
        "x0": 0.5,
        "y0": 0.5,
        "sigma": 0.1,
        "nx": 21,
        "ny": 21,
        "lx": 1.0,
        "ly": 1.0,
        "t_end": 0.01,
    }
