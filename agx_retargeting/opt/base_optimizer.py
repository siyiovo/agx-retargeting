from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from ..types import Residual as ResidualProblem

__all__ = ["BaseOptimizer", "Bounds", "OptimizerOptions", "ResidualProblem"]


@dataclass(frozen=True, slots=True)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True, slots=True)
class OptimizerOptions:
    max_evaluations: int
    gradient_tolerance: float
    cost_tolerance: float
    step_tolerance: float


class BaseOptimizer(ABC):
    def __init__(
        self,
        n_parameters: int,
        max_residuals: int,
        *,
        options: OptimizerOptions,
    ):
        self.n_parameters = n_parameters
        self.max_residuals = max_residuals
        self.options = options
        self._residual: np.ndarray = np.empty(self.max_residuals)
        self._jacobian: np.ndarray = np.empty((self.max_residuals, self.n_parameters))
        self._busy: bool = False

    @abstractmethod
    def solve(
        self,
        problem: ResidualProblem,
        initial_q: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizeResult: ...

    def _prepare(self, problem, initial_q, bounds):
        shape = (self.n_parameters,)
        q = np.asarray(initial_q, dtype=float)
        lower = (
            np.full(shape, -np.inf)
            if bounds is None
            else np.asarray(bounds.lower, dtype=float)
        )
        upper = (
            np.full(shape, np.inf)
            if bounds is None
            else np.asarray(bounds.upper, dtype=float)
        )
        return np.clip(q, lower, upper), lower, upper, problem.residual_count
