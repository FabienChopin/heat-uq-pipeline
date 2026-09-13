"""heatuq: a 2D transient heat-diffusion finite-difference solver."""

from heatuq.solver import compute_stable_dt, solve_heat_equation_2d

__version__ = "0.1.0"

__all__ = ["__version__", "compute_stable_dt", "solve_heat_equation_2d"]
