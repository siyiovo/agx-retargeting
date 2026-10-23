from dataclasses import replace
from functools import partial
from time import perf_counter

import numpy as np
import pinocchio as pin
import quaternion
from scipy.optimize import OptimizeResult
from threadpoolctl import ThreadpoolController

from ..opt import (
    Bounds,
    DogboxOptimizer,
    OptimizerOptions,
    ProxQPOptimizer,
    ProxQPOptions,
)
from ..types import HandInput
from .hand_model import HandConfig, HandModel
from .residual import HandResidual, VelocityProjection


class QMethodInitializer:
    def __init__(
        self,
        hand: HandModel,
        config: HandConfig,
        *,
        position_sigma: float = 0.05,
        smooth_weight: float = 2.5,
        options: OptimizerOptions,
        optimizer_factory=DogboxOptimizer,
    ):
        self.hand, self.config = hand, config
        self._keypoints = np.asarray(
            [value.keypoint for value in config.landmarks.values()]
        )
        self._frame_ids = np.asarray(
            [
                hand.model.getFrameId(value.link, pin.BODY)
                for value in config.landmarks.values()
            ]
        )
        self._wrist = tuple(config.landmarks).index("wrist")
        self._position_scale = 1 / position_sigma
        self._smooth_scale = np.sqrt(smooth_weight)
        count = len(self._frame_ids)
        self.residual_count = 3 * count + hand.nq
        self._kinematics = hand.bind_frames(self._frame_ids)
        self._positions = self._kinematics.positions
        self._jacobians = self._kinematics.jacobians
        self._targets = np.empty((count, 3))
        self._reference: np.ndarray | None = None
        self._cached_q = np.full(hand.nq, np.nan)
        self._residual = np.empty(self.residual_count)
        self._jacobian = np.zeros((self.residual_count, hand.nq))
        self._jacobian[-hand.nq :] = self._smooth_scale * np.eye(hand.nq)
        self._optimizer = optimizer_factory(
            hand.nq, self.residual_count, options=options
        )
        self._bounds = Bounds(hand.lower_limits, hand.upper_limits)

    def initialize(self, frame: HandInput, reference: np.ndarray) -> OptimizeResult:
        started = perf_counter()
        reference = np.asarray(reference, dtype=float)
        self._kinematics.evaluate(reference, jacobian=False)
        source = self.config.scale * (
            frame.landmarks[self._keypoints] - frame.landmarks[0]
        )
        wrist = self._positions[self._wrist]
        rotor = quaternion.optimal_alignment_in_Euclidean_metric(
            self._positions - wrist, source
        )
        rotation = quaternion.as_rotation_matrix(rotor)
        self._targets[:] = source @ rotation.T + wrist
        self._reference = reference
        self._cached_q.fill(np.nan)
        result = self._optimizer.solve(self, reference, self._bounds)
        result.rotation = rotation
        result.elapsed_ms = (perf_counter() - started) * 1000
        return result

    def evaluate(self, q, residual, jacobian=None) -> bool:
        if not np.array_equal(q, self._cached_q):
            self._kinematics.evaluate(q)
            stop = self.residual_count - self.hand.nq
            self._residual[:stop] = (
                self._position_scale * (self._positions - self._targets)
            ).ravel()
            self._residual[stop:] = self._smooth_scale * (q - self._reference)
            self._jacobian[:stop] = (self._position_scale * self._jacobians).reshape(
                -1, self.hand.nq
            )
            self._cached_q[:] = q
        residual[:] = self._residual
        if jacobian is not None:
            jacobian[:] = self._jacobian
        return True


class Retargeter:
    def __init__(self, config: HandConfig, params: dict, *, initial_q=None):
        self.config = config
        self.hand = HandModel(config.urdf_path)
        self.problem = HandResidual(self.hand, config, **params["residual"])
        solver = dict(params["solver"])
        backend = solver.pop("backend")
        qp_options = ProxQPOptions(**solver.pop("proxqp", {}))
        optimizer_factory = {
            "dogbox": DogboxOptimizer,
            "proxqp": partial(ProxQPOptimizer, qp_options=qp_options),
        }[backend]
        options = OptimizerOptions(**solver)
        init_params = params["initializer"]
        self.initializer = QMethodInitializer(
            self.hand,
            config,
            position_sigma=init_params["position_sigma"],
            smooth_weight=init_params["smooth_weight"],
            options=replace(options, max_evaluations=init_params["max_evaluations"]),
            optimizer_factory=optimizer_factory,
        )
        self.projection = VelocityProjection(self.problem)
        self.optimizer = optimizer_factory(
            self.projection.n_parameters, self.problem.residual_count, options=options
        )
        self._step_tolerance = options.step_tolerance
        self._blas = ThreadpoolController()
        chart_limit = 0.99 * np.pi / np.sqrt(3)
        self._bounds = Bounds(
            np.r_[
                self.hand.lower_limits,
                -self.hand.velocity_limits,
                [-np.inf] * 3,
                [-chart_limit] * 3,
            ],
            np.r_[
                self.hand.upper_limits,
                self.hand.velocity_limits,
                [np.inf] * 3,
                [chart_limit] * 3,
            ],
        )
        self._reduced_bounds = Bounds(
            np.r_[self._bounds.lower[: self.hand.nq], self._bounds.lower[-6:]],
            np.r_[self._bounds.upper[: self.hand.nq], self._bounds.upper[-6:]],
        )
        self._initial_q = self.hand.neutral_q.copy()
        finite = np.isfinite(self.hand.lower_limits) & np.isfinite(
            self.hand.upper_limits
        )
        self._initial_q[finite] = (
            self.hand.lower_limits[finite] + self.hand.upper_limits[finite]
        ) / 2
        if initial_q is None:
            reference = init_params.get("reference_q")
            if isinstance(reference, dict):
                reference = reference.get(config.side)
            if reference is not None:
                initial_q = reference
        self.reset(initial_q)

    def reset(self, initial_q=None):
        if initial_q is not None:
            q = np.asarray(initial_q, dtype=float)
            self._initial_q[:] = np.clip(
                q, self.hand.lower_limits, self.hand.upper_limits
            )
        self._previous = None
        self._last_output = None
        self._timestamp = None

    def step(self, frame: HandInput):
        started = perf_counter()
        with self._blas.limit(limits=1, user_api="blas"):
            result = self._step(frame)
        if result is not None:
            result.pipeline_ms = (perf_counter() - started) * 1000
        return result

    def _step(self, frame: HandInput):
        if frame.side != self.config.side:
            raise ValueError("HandInput and Retargeter must describe the same side")
        if not frame.tracked:
            self.reset()
            return None
        timestamp = float(frame.timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamp must be finite nonnegative seconds")
        dt = None if self._timestamp is None else timestamp - self._timestamp
        if dt is not None and dt <= 0:
            raise ValueError(
                "Frame timestamps must increase; reset after a clock change"
            )
        previous = self._previous
        initialization = None
        if previous is None:
            initialization = self.initializer.initialize(frame, self._initial_q)
            seed = initialization.x
        self.problem.set_frame(frame, previous, dt)
        if previous is not None:
            seed = np.clip(
                self.problem._prediction[: self.hand.nq],
                self.hand.lower_limits,
                self.hand.upper_limits,
            )
        initial = np.r_[seed, np.zeros(6) if previous is None else previous.eta]
        started = perf_counter()
        self.optimizer.options = replace(
            self.optimizer.options,
            step_tolerance=self._step_tolerance / self.projection.step_amplification,
        )
        result = self.optimizer.solve(self.projection, initial, self._reduced_bounds)
        self._previous = self.projection.finalize(result, self._bounds)
        result.elapsed_ms = (perf_counter() - started) * 1000
        result.initial_q = seed
        result.initialization_ms = (
            0.0 if initialization is None else initialization.elapsed_ms
        )
        result.initialization_success = (
            None if initialization is None else bool(initialization.success)
        )
        result.initialization_nfev = (
            0 if initialization is None else int(initialization.nfev)
        )
        result.target_positions = self.problem.target_positions
        output = result.x
        result.q_output = output
        result.dt = dt
        result.raw_fit = self.problem.fit_errors(result.x)
        result.output_fit = result.raw_fit
        result.output_delta_rms = None
        result.output_max_delta = None
        result.output_velocity_rms = None
        result.output_velocity = None
        if self._last_output is not None:
            delta = output - self._last_output
            result.output_velocity = delta / dt
            result.output_delta_rms = float(np.sqrt(np.mean(delta**2)))
            result.output_max_delta = float(np.max(np.abs(delta)))
            result.output_velocity_rms = result.output_delta_rms / dt
        self._last_output = output.copy()
        self._timestamp = timestamp
        return result
