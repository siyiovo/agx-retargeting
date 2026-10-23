from dataclasses import dataclass
from time import perf_counter

import numpy as np
import proxsuite
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from .base_optimizer import BaseOptimizer, Bounds, ResidualProblem


@dataclass(frozen=True, slots=True)
class ProxQPOptions:
    initial_damping: float = 1e-4
    absolute_tolerance: float = 1e-9
    max_iterations: int = 100


class ProxQPOptimizer(BaseOptimizer):
    def __init__(self, n_parameters, max_residuals, *, options, qp_options=None):
        super().__init__(n_parameters, max_residuals, options=options)
        self.qp_options = ProxQPOptions() if qp_options is None else qp_options
        settings = self.qp_options
        n = self.n_parameters
        self._qp = proxsuite.proxqp.dense.QP(n, 0, 0, True)
        self._qp.settings.eps_abs = settings.absolute_tolerance
        self._qp.settings.eps_rel = 0.0
        self._qp.settings.max_iter = settings.max_iterations
        self._qp.settings.initial_guess = (
            proxsuite.proxqp.WARM_START_WITH_PREVIOUS_RESULT
        )
        self._initialized = False
        self._hessian = np.empty((n, n))
        self._gradient = np.empty(n)
        self._scale = np.empty(n)
        self._lower = np.empty(n)
        self._upper = np.empty(n)
        self._trial_x = np.empty(n)
        self._trial_residual = np.empty(max_residuals)
        self._trial_jacobian = np.empty((max_residuals, n))
        self._model_step = np.empty(max_residuals)

    def _qp_tolerance(self, scale):
        return min(
            self.qp_options.absolute_tolerance,
            self.options.gradient_tolerance / float(np.max(scale)),
        )

    def _diagnostics(self, x, lower, upper, residual, jacobian):
        gradient = jacobian.T @ residual
        scale = np.maximum(np.linalg.norm(jacobian, axis=0), 1.0)
        tolerance = self._qp_tolerance(scale) / scale + 8 * np.spacing(np.abs(x))
        lower_distance, upper_distance = x - lower, upper - x
        near_lower = (lower_distance <= tolerance) & (lower_distance < upper_distance)
        near_upper = (upper_distance <= tolerance) & (upper_distance < lower_distance)
        active = np.zeros(self.n_parameters, dtype=int)
        active[near_lower] = -1
        active[near_upper] = 1
        active[lower == upper] = 1
        projected = gradient.copy()
        projected[
            (near_lower & (gradient > 0))
            | (near_upper & (gradient < 0))
            | (lower == upper)
        ] = 0
        return gradient, active, float(np.linalg.norm(projected, ord=np.inf))

    def solve(
        self,
        problem: ResidualProblem,
        initial_q: NDArray[np.float64],
        bounds: Bounds | None = None,
    ) -> OptimizeResult:
        if self._busy:
            raise RuntimeError("An optimizer instance is not reentrant")
        self._busy = True
        started = perf_counter()
        try:
            x, lower, upper, rows = self._prepare(problem, initial_q, bounds)
            residual = self._residual[:rows]
            jacobian = self._jacobian[:rows]
            trial_residual = self._trial_residual[:rows]
            trial_jacobian = self._trial_jacobian[:rows]

            def evaluate(point, values, derivative):
                candidate = point.view()
                candidate.flags.writeable = False
                problem.evaluate(candidate, values, derivative)
                value = 0.5 * float(values @ values)
                if not np.isfinite(value):
                    raise ValueError("The objective cost must be finite")
                return value

            cost = evaluate(x, residual, jacobian)
            nfev, nqp, qp_iterations, rejected = 1, 0, 0, 0
            qp_ms = 0.0
            damping = self.qp_options.initial_damping
            previous_damping = damping
            linearize = True
            status = 0
            message = "The maximum number of function evaluations is exceeded."
            options = self.options
            while True:
                if linearize:
                    gradient, active, optimality = self._diagnostics(
                        x, lower, upper, residual, jacobian
                    )
                    if optimality <= options.gradient_tolerance:
                        status, message = (
                            1,
                            "The projected gradient tolerance is satisfied.",
                        )
                        break
                    np.matmul(jacobian.T, jacobian, out=self._hessian)
                    np.sqrt(self._hessian.diagonal(), out=self._scale)
                    np.maximum(self._scale, 1.0, out=self._scale)
                    self._qp.settings.eps_abs = self._qp_tolerance(self._scale)
                    self._hessian /= self._scale[:, None]
                    self._hessian /= self._scale[None, :]
                    np.divide(gradient, self._scale, out=self._gradient)
                    self._hessian.flat[:: self.n_parameters + 1] += damping
                    previous_damping = damping
                    np.multiply(lower - x, self._scale, out=self._lower)
                    np.multiply(upper - x, self._scale, out=self._upper)
                    linearize = False
                if nfev >= options.max_evaluations:
                    break

                self._hessian.flat[:: self.n_parameters + 1] += (
                    damping - previous_damping
                )
                previous_damping = damping
                qp_started = perf_counter()
                if self._initialized:
                    self._qp.update(
                        H=self._hessian,
                        g=self._gradient,
                        l_box=self._lower,
                        u_box=self._upper,
                        update_preconditioner=True,
                    )
                else:
                    self._qp.init(
                        H=self._hessian,
                        g=self._gradient,
                        l_box=self._lower,
                        u_box=self._upper,
                    )
                    self._initialized = True
                self._qp.solve()
                qp_ms += (perf_counter() - qp_started) * 1000
                nqp += 1
                qp_iterations += self._qp.results.info.iter
                if (
                    self._qp.results.info.status
                    != proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
                ):
                    raise RuntimeError(f"ProxQP failed: {self._qp.results.info.status}")
                increment = self._qp.results.x
                violation = max(
                    float(np.max(self._lower - increment)),
                    float(np.max(increment - self._upper)),
                )
                if violation > 10 * self.qp_options.absolute_tolerance:
                    raise RuntimeError(
                        "ProxQP returned an increment outside its box bounds"
                    )
                np.divide(increment, self._scale, out=self._trial_x)
                self._trial_x += x
                np.clip(self._trial_x, lower, upper, out=self._trial_x)
                step = self._trial_x - x
                np.matmul(jacobian, step, out=self._model_step[:rows])
                change = self._model_step[:rows]
                predicted = -float(residual @ change) - 0.5 * float(change @ change)
                trial_cost = evaluate(self._trial_x, trial_residual, trial_jacobian)
                nfev += 1
                reduction = cost - trial_cost
                ratio = reduction / predicted if predicted > 0 else -np.inf
                if ratio > 0:
                    old_cost = cost
                    x, self._trial_x = self._trial_x, x
                    residual, trial_residual = trial_residual, residual
                    jacobian, trial_jacobian = trial_jacobian, jacobian
                    cost = trial_cost
                    if ratio > 0.75:
                        damping = max(damping * 0.25, 1e-12)
                    elif ratio < 0.25:
                        damping *= 4
                    linearize = True
                    cost_stop = (
                        options.cost_tolerance > 0
                        and reduction <= options.cost_tolerance * old_cost
                        and ratio > 0.25
                    )
                    step_stop = (
                        options.step_tolerance > 0
                        and np.linalg.norm(step)
                        <= options.step_tolerance
                        * (options.step_tolerance + np.linalg.norm(x))
                        and ratio > 0.25
                    )
                    if cost_stop or step_stop:
                        status = (
                            4 if cost_stop and step_stop else (2 if cost_stop else 3)
                        )
                        message = "The cost and/or step tolerance is satisfied."
                        break
                else:
                    damping *= 4
                    rejected += 1

            gradient, active, optimality = self._diagnostics(
                x, lower, upper, residual, jacobian
            )
            return OptimizeResult(
                x=x.copy(),
                fun=residual.copy(),
                cost=cost,
                jac=jacobian.copy(),
                grad=gradient,
                optimality=optimality,
                active_mask=active,
                nfev=nfev,
                njev=nfev,
                nit=nqp,
                status=status,
                success=status > 0,
                message=message,
                nqp=nqp,
                qp_iterations=qp_iterations,
                rejected_steps=rejected,
                qp_ms=qp_ms,
                elapsed_ms=(perf_counter() - started) * 1000,
            )
        finally:
            self._busy = False
