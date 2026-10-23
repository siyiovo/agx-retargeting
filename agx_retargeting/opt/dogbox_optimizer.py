from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from .base_optimizer import BaseOptimizer, Bounds, ResidualProblem


class DogboxOptimizer(BaseOptimizer):
    def solve(
        self,
        problem: ResidualProblem,
        initial_q: np.ndarray,
        bounds: Bounds | None = None,
    ) -> OptimizeResult:
        if self._busy:
            raise RuntimeError("An optimizer instance is not reentrant")
        self._busy = True
        started = perf_counter()
        try:
            q, lower, upper, rows = self._prepare(problem, initial_q, bounds)
            boundary_tolerance = max(
                self.options.step_tolerance, 8 * np.finfo(float).eps
            ) * max(1.0, np.linalg.norm(q))
            near_lower = q - lower <= boundary_tolerance
            near_upper = upper - q <= boundary_tolerance
            q[near_lower] = lower[near_lower]
            q[near_upper] = upper[near_upper]
            residual = self._residual[:rows]
            jacobian = self._jacobian[:rows]

            def fill(configuration, need_jacobian):
                problem.evaluate(
                    configuration, residual, jacobian if need_jacobian else None
                )

            free = np.flatnonzero(lower < upper)
            candidate = q.view()
            candidate.flags.writeable = False

            def evaluate(x, need_jacobian):
                q[free] = x
                fill(candidate, need_jacobian)
                if need_jacobian:
                    return jacobian[:, free]
                return residual.copy()

            if free.size == 0:
                fill(q, True)
                result = OptimizeResult(
                    x=q,
                    fun=residual.copy(),
                    cost=0.5 * float(residual @ residual),
                    jac=jacobian.copy(),
                    grad=jacobian.T @ residual,
                    optimality=0.0,
                    active_mask=np.ones(self.n_parameters, dtype=int),
                    nfev=1,
                    njev=1,
                    status=1,
                    success=True,
                    message="All parameters are fixed by bounds.",
                )
            else:
                options = self.options
                result = least_squares(
                    lambda x: evaluate(x, False),
                    q[free],
                    jac=lambda x: evaluate(x, True),
                    bounds=(lower[free], upper[free]),
                    method="dogbox",
                    loss="linear",
                    tr_solver="exact",
                    x_scale="jac",
                    max_nfev=options.max_evaluations,
                    gtol=options.gradient_tolerance,
                    ftol=options.cost_tolerance or None,
                    xtol=options.step_tolerance or None,
                )
                if free.size != self.n_parameters:
                    q[free] = result.x
                    result.x = q
                    result.jac = jacobian.copy()
                    result.grad = result.jac.T @ result.fun
                    active_mask = np.ones(self.n_parameters, dtype=int)
                    active_mask[free] = result.active_mask
                    result.active_mask = active_mask
            result.elapsed_ms = (perf_counter() - started) * 1000.0
            return result
        finally:
            self._busy = False
