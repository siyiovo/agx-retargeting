from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import numpy as np
import pinocchio as pin
from numba import boolean, float64, njit, types
from scipy.linalg import solve_triangular
from scipy.special import expit

from ..types import FINGERS, HandInput
from .hand_model import HandConfig, HandModel


@njit(cache=False)
def write_residual_block(value, derivative, info, weight, delta, robust, rout, jout):
    rows, columns = derivative.shape
    for row in range(rows):
        rout[row] = 0.0
        for inner in range(rows):
            rout[row] += info[row, inner] * value[inner]
        for column in range(columns):
            jout[row, column] = 0.0
            for inner in range(rows):
                jout[row, column] += info[row, inner] * derivative[inner, column]
    if robust:
        squared = 0.0
        for row in range(rows):
            squared += rout[row] * rout[row]
        norm = np.sqrt(squared)
        if norm > delta:
            rho = 2 * delta * norm - delta * delta
            factor = np.sqrt(rho / squared)
            slope = (delta / norm - rho / squared) / (factor * squared)
            for column in range(columns):
                projection = 0.0
                for row in range(rows):
                    projection += rout[row] * jout[row, column]
                for row in range(rows):
                    jout[row, column] = (
                        factor * jout[row, column] + slope * rout[row] * projection
                    )
            for row in range(rows):
                rout[row] *= factor
    for row in range(rows):
        rout[row] *= weight
        for column in range(columns):
            jout[row, column] *= weight


class GeometryTasks(NamedTuple):
    targets: np.ndarray
    directions: np.ndarray
    pairs: np.ndarray
    information: np.ndarray
    weights: np.ndarray
    delta: float
    direction_reference: np.ndarray
    direction_targets: np.ndarray


class MotionTask(NamedTuple):
    targets: np.ndarray
    previous_error: np.ndarray
    basis: np.ndarray
    world: np.ndarray
    world_rotation: np.ndarray
    information: np.ndarray
    scale: float
    dt: float


@njit(cache=False, inline="always")
def direction_log(direction, reference, value):
    cosine = 0.0
    for axis in range(3):
        cosine += reference[axis] * direction[axis]
    squared = 0.0
    for axis in range(3):
        value[axis] = direction[axis] - cosine * reference[axis]
        squared += value[axis] * value[axis]
    if cosine > 0 and squared < 1e-10:
        distance = 1 - cosine
        factor = 1 + distance / 3 + 2 * distance * distance / 15
        slope = -1 / 3 - 4 * distance / 15
    else:
        if squared <= 1e-20:
            raise ValueError("Terminal direction is antipodal to its reference")
        sine = np.sqrt(squared)
        factor = np.arctan2(sine, cosine) / sine
        slope = (factor * cosine - 1) / squared
    for axis in range(3):
        value[axis] *= factor
    return factor, slope


@njit(cache=False)
def write_direction_targets(directions, references, targets):
    for finger in range(5):
        direction_log(directions[finger], references[finger], targets[finger])


@njit(cache=False)
def write_geometry(positions, derivatives, tasks, residual, jacobian):
    n = derivatives.shape[2]
    values = np.empty((14, 3))
    gradients = np.empty((14, 3, n))
    for finger in range(5):
        length2 = 0.0
        for axis in range(3):
            value = positions[finger, axis] - positions[5 + finger, axis]
            values[9 + finger, axis] = value
            length2 += value * value
            values[finger, axis] = positions[finger, axis] - tasks.targets[finger, axis]
            for column in range(n):
                gradients[finger, axis, column] = derivatives[finger, axis, column]
        length = np.sqrt(length2)
        direction = values[9 + finger] / length
        reference = tasks.direction_reference[finger]
        factor, slope = direction_log(direction, reference, values[9 + finger])
        for column in range(n):
            projection = 0.0
            for axis in range(3):
                projection += direction[axis] * (
                    derivatives[finger, axis, column]
                    - derivatives[5 + finger, axis, column]
                )
            axial = 0.0
            for axis in range(3):
                axial += (
                    reference[axis]
                    * (
                        derivatives[finger, axis, column]
                        - derivatives[5 + finger, axis, column]
                        - direction[axis] * projection
                    )
                    / length
                )
            for axis in range(3):
                gradients[9 + finger, axis, column] = (
                    factor
                    * (
                        derivatives[finger, axis, column]
                        - derivatives[5 + finger, axis, column]
                        - direction[axis] * projection
                    )
                    / length
                    + (
                        slope / factor * values[9 + finger, axis]
                        - factor * reference[axis]
                    )
                    * axial
                )
        for axis in range(3):
            values[9 + finger, axis] -= tasks.direction_targets[finger, axis]
    for finger in range(4):
        for axis in range(3):
            values[5 + finger, axis] = (
                positions[1 + finger, axis]
                - positions[0, axis]
                - tasks.pairs[finger, axis]
            )
            for column in range(n):
                gradients[5 + finger, axis, column] = (
                    derivatives[1 + finger, axis, column] - derivatives[0, axis, column]
                )
    for block in range(14):
        category = 0 if block < 5 else (1 if block < 9 else 2)
        weight = tasks.weights[block] if block < 9 else 1.0
        start = 3 * block
        write_residual_block(
            values[block],
            gradients[block],
            tasks.information[category],
            weight,
            tasks.delta,
            True,
            residual[start : start + 3],
            jacobian[start : start + 3, :n],
        )


@njit(cache=False)
def write_motion(
    points,
    point_jacobians,
    rotation,
    right,
    translation,
    task,
    tip_error,
    residual,
    jacobian,
):
    n = point_jacobians.shape[2]
    effective = task.basis @ rotation @ task.basis.T
    predicted = points @ effective.T + task.scale * (
        task.basis @ np.ascontiguousarray(translation)
    )
    tip_error[:] = (predicted - task.targets) @ task.world.T
    world_joint = task.world @ effective
    world_rotation_derivative = -task.world_rotation @ rotation
    rotation_point = np.empty((3, 3))
    derivative = np.zeros((3, 2 * n + 6))
    for finger in range(5):
        if task.dt == 0:
            continue
        derivative[:, :n] = world_joint @ point_jacobians[finger]
        derivative[:, -6:-3] = task.world_rotation
        point = task.basis.T @ points[finger]
        rotation_point[0] = (0.0, -point[2], point[1])
        rotation_point[1] = (point[2], 0.0, -point[0])
        rotation_point[2] = (-point[1], point[0], 0.0)
        derivative[:, -3:] = (
            world_rotation_derivative @ rotation_point @ right / task.scale
        )
        value = (tip_error[finger] - task.previous_error[finger]) / task.dt
        start = 3 * finger
        write_residual_block(
            value,
            derivative / task.dt,
            task.information,
            1.0,
            1.0,
            False,
            residual[start : start + 3],
            jacobian[start : start + 3],
        )


@njit(cache=False)
def measure_fit(positions, tasks):
    position_error, pair_error, direction_error = 0.0, 0.0, 0.0
    for finger in range(5):
        length, cosine = 0.0, 0.0
        for axis in range(3):
            error = positions[finger, axis] - tasks.targets[finger, axis]
            position_error += error * error
            bone = positions[finger, axis] - positions[5 + finger, axis]
            length += bone * bone
        length = np.sqrt(length)
        for axis in range(3):
            cosine += (
                (positions[finger, axis] - positions[5 + finger, axis])
                / length
                * tasks.directions[finger, axis]
            )
            if finger:
                error = (
                    positions[finger, axis]
                    - positions[0, axis]
                    - tasks.pairs[finger - 1, axis]
                )
                pair_error += error * error
        angle = np.arccos(min(1.0, max(-1.0, cosine)))
        direction_error += angle * angle
    return (
        np.sqrt(position_error / 5),
        np.sqrt(direction_error / 5),
        np.sqrt(pair_error / 4),
    )


@njit(cache=False)
def project_velocity(reduced, prediction, gain, limits, state, derivative):
    n = limits.size
    state[:n], state[-6:] = reduced[:n], reduced[-6:]
    for joint in range(n):
        if gain == 0:
            state[n + joint] = 0.0
            derivative[joint] = 0.0
        else:
            speed = (reduced[joint] - prediction[joint]) * gain + prediction[n + joint]
            state[n + joint] = min(limits[joint], max(-limits[joint], speed))
            derivative[joint] = gain if -limits[joint] < speed < limits[joint] else 0.0


def damped_joint_process(dt, decay_s):
    x = dt / decay_s
    a = -np.expm1(-x)
    if x < 1e-3:
        qpp = dt**3 * (
            1 / 3 + x * (-1 / 4 + x * (7 / 60 + x * (-1 / 24 + x * 31 / 2520)))
        )
    else:
        qpp = decay_s**2 * (dt - decay_s * a - 0.5 * decay_s * a**2)
    qpv = 0.5 * decay_s**2 * a**2
    qvv = 0.5 * decay_s * a * (2 - a)
    return decay_s * a, np.exp(-x), qpp, qpv, qvv


@cache
def prepare_residual_kernels():
    vector = types.Array(float64, 1, "A", readonly=True)
    matrix = types.Array(float64, 2, "A", readonly=True)
    tensor = types.Array(float64, 3, "C", readonly=True)
    contiguous = types.Array(float64, 2, "C", readonly=True)
    column_major = types.Array(float64, 2, "F", readonly=True)
    geometry = types.NamedTuple(
        (
            float64[:, ::1],
            float64[:, ::1],
            float64[:, ::1],
            float64[:, :, ::1],
            float64[::1],
            float64,
            contiguous,
            float64[:, ::1],
        ),
        GeometryTasks,
    )
    motion = types.NamedTuple(
        (
            float64[:, ::1],
            float64[:, ::1],
            contiguous,
            float64[:, ::1],
            float64[::1, :],
            float64[:, ::1],
            float64,
            float64,
        ),
        MotionTask,
    )
    write_residual_block.compile(
        (vector, matrix, matrix, float64, float64, boolean, float64[:], float64[:, :])
    )
    write_residual_block.disable_compile()
    write_direction_targets.compile((matrix, contiguous, float64[:, ::1]))
    write_geometry.compile((matrix, tensor, geometry, float64[:], float64[:, :]))
    write_motion.compile(
        (
            contiguous,
            tensor,
            column_major,
            contiguous,
            vector,
            motion,
            float64[:, :],
            float64[:],
            float64[:, :],
        )
    )
    measure_fit.compile((matrix, geometry))
    project_velocity.compile(
        (
            vector,
            vector,
            float64,
            vector,
            float64[:],
            float64[:],
        )
    )
    for kernel in (
        write_direction_targets,
        write_geometry,
        write_motion,
        measure_fit,
        project_velocity,
    ):
        kernel.disable_compile()


def sqrt_info(sigma):
    sigma = np.asarray(sigma, dtype=float)
    covariance = np.diag(sigma**2)
    return solve_triangular(
        np.linalg.cholesky(covariance), np.eye(sigma.size), lower=True
    )


@dataclass(slots=True)
class PreviousState:
    q: np.ndarray
    v: np.ndarray
    eta: np.ndarray
    tip_error: np.ndarray | None


@dataclass(frozen=True, slots=True)
class PinchWeights:
    center_distance: float = 0.040
    slope: float = 100.0


class HandResidual:
    motion_enabled = True

    def __init__(
        self,
        hand: HandModel,
        config: HandConfig,
        *,
        pair_sigma,
        bone_sigma,
        shape_sigma,
        wrist_sigma,
        wrist_process_sigma,
        acceleration_sigma,
        initial_velocity_sigma,
        motion_sigma,
        motion_weight,
        huber_delta,
        pinch,
    ):
        self.hand, self.config = hand, config
        self.n_parameters = 2 * hand.nq + 6
        self.pinch = PinchWeights(**pinch)
        self._geometry_info = np.ascontiguousarray(
            np.stack(
                [
                    sqrt_info(np.full(3, sigma))
                    for sigma in (shape_sigma, pair_sigma, bone_sigma)
                ]
            )
        )
        self._wrist_info = sqrt_info(wrist_sigma)
        self._wrist_process_info = sqrt_info(wrist_process_sigma)
        self._initial_velocity_info = sqrt_info(
            np.full(hand.nq, initial_velocity_sigma)
        )
        self._acceleration_info = sqrt_info(
            np.broadcast_to(acceleration_sigma, (hand.nq,))
        )
        self._motion_info = np.ascontiguousarray(
            np.sqrt(motion_weight) * sqrt_info(np.full(3, motion_sigma))
        )
        self._basis = np.array(config.robot_basis, dtype=float, order="C", copy=True)
        self._basis.flags.writeable = False
        self._huber_delta = float(huber_delta)
        self.names = (
            *[f"{finger}_tip" for finger in FINGERS],
            "thumb_ip",
            *[f"{finger}_dip" for finger in FINGERS[1:]],
        )
        self.keypoints = np.array(
            [config.landmarks[name].keypoint for name in self.names]
        )
        self.ids = np.array(
            [
                hand.model.getFrameId(config.landmarks[name].link, pin.BODY)
                for name in self.names
            ]
        )
        self.tip_rows = np.arange(5)
        self.dip_rows = np.arange(5, 10)
        self._tip_indices = self.keypoints[self.tip_rows]
        palm_id = hand.model.getFrameId(config.landmarks["wrist"].link, pin.BODY)
        palm = hand.model.frames[palm_id]
        if palm.parentJoint != 0:
            raise ValueError("The hand palm must be fixed to the URDF root")
        self.base = palm.placement.translation.copy()
        self._kinematics = hand.bind_frames(self.ids)
        self.positions = self._kinematics.positions
        self.point_jacobian = self._kinematics.jacobians
        self._fit_kinematics = hand.bind_frames(self.ids)
        reference_q = hand.neutral_q.copy()
        bounded = np.isfinite(hand.lower_limits) & np.isfinite(hand.upper_limits)
        reference_q[bounded] = (
            hand.lower_limits[bounded] + hand.upper_limits[bounded]
        ) / 2
        self._kinematics.evaluate(reference_q, jacobian=False)
        reference_bones = self.positions[:5] - self.positions[5:]
        reference_lengths = np.linalg.norm(reference_bones, axis=1, keepdims=True)
        if np.any(reference_lengths <= 1e-10):
            raise ValueError(
                "Robot terminal bone endpoints must have distinct positions"
            )
        self._direction_reference = np.ascontiguousarray(
            reference_bones / reference_lengths
        )
        self._direction_reference.flags.writeable = False
        self._direction_targets = np.empty((5, 3))
        self._tip_error = np.empty((5, 3))
        self._process_info = np.zeros((2 * hand.nq, 2 * hand.nq))
        self._prediction = np.empty(2 * hand.nq)
        self._velocity_gain = 0.0
        self.blocks = {
            "geometry": (0, 27),
            "direction": (27, 42),
            "wrist": (42, 54),
            "process": (54, 54 + 2 * hand.nq),
        }
        self.residual_count = 54 + 2 * hand.nq
        if self.motion_enabled:
            self.blocks["motion"] = (self.residual_count, self.residual_count + 15)
            self.residual_count += 15
        self._r = np.empty(self.residual_count)
        self._j = np.empty((self.residual_count, self.n_parameters))
        self._cached_x = np.full(self.n_parameters, np.nan)
        self._previous: PreviousState | None = None
        self._dt = 0.0
        self.tip_error: np.ndarray | None = None
        prepare_residual_kernels()

    @staticmethod
    def correction(eta):
        return pin.SE3(pin.exp3(eta[3:]), eta[:3])

    def set_frame(self, frame: HandInput, previous=None, dt=None):
        self._previous = previous
        self._dt = 0.0 if dt is None else float(dt)
        self._velocity_gain = 0.0
        self._cached_x.fill(np.nan)
        self.reference = pin.SE3(self.hand.estimate_coord(frame), frame.landmarks[0])
        self.source = (
            self.config.scale
            * (frame.landmarks - frame.landmarks[0])
            @ self.reference.rotation
            @ self.config.robot_basis.T
        )
        self._targets = self.source + self.base
        tips = self.source[self._tip_indices]
        vectors = tips[1:] - tips[0]
        distances = np.linalg.norm(vectors, axis=1) / self.config.scale
        self.pair_targets = self.build_pair_targets(vectors, distances)
        self.activation = expit(
            -self.pinch.slope * (distances - self.pinch.center_distance)
        )
        self.shape_weights = np.r_[1 - np.max(self.activation), 1 - self.activation]
        self.geometry_weights = np.sqrt(np.r_[self.shape_weights, self.activation])
        bones = tips - self.source[self.keypoints[self.dip_rows]]
        lengths = np.linalg.norm(bones, axis=1, keepdims=True)
        if np.any(lengths <= 1e-10):
            raise ValueError(
                "Input terminal bone endpoints must have distinct positions"
            )
        self.source_directions = bones / lengths
        write_direction_targets(
            self.source_directions, self._direction_reference, self._direction_targets
        )
        self._geometry = GeometryTasks(
            self._targets[self._tip_indices],
            self.source_directions,
            self.pair_targets,
            self._geometry_info,
            self.geometry_weights,
            self._huber_delta,
            self._direction_reference,
            self._direction_targets,
        )
        self._world_from_local = (
            self.reference.rotation @ self.config.robot_basis.T / self.config.scale
        )
        if previous is not None:
            self._previous_correction_inverse = self.correction(previous.eta).inverse()
            self._wrist_time_info = self._wrist_process_info / np.sqrt(dt)
            n = self.hand.nq
            if self.config.velocity_decay_s is None:
                root = np.sqrt(dt)
                self._process_info[:n, :n] = (
                    np.sqrt(3) * self._acceleration_info / (dt * root)
                )
                self._process_info[n:, :n] = -3 * self._acceleration_info / (dt * root)
                self._process_info[n:, n:] = 2 * self._acceleration_info / root
                self._prediction[:n] = previous.q + dt * previous.v
                self._prediction[n:] = previous.v
                self._velocity_gain = 1.5 / dt
            else:
                step, retention, qpp, qpv, qvv = damped_joint_process(
                    dt, self.config.velocity_decay_s
                )
                conditional = np.sqrt(qvv - qpv**2 / qpp)
                self._process_info[:n, :n] = self._acceleration_info / np.sqrt(qpp)
                self._process_info[n:, :n] = (
                    -qpv / (qpp * conditional) * self._acceleration_info
                )
                self._process_info[n:, n:] = self._acceleration_info / conditional
                self._prediction[:n] = previous.q + step * previous.v
                self._prediction[n:] = retention * previous.v
                self._velocity_gain = qpv / qpp
        if self.motion_enabled:
            previous_error = (
                self._tip_error
                if previous is None
                else np.require(
                    previous.tip_error, dtype=float, requirements=["C", "W"]
                )
            )
            self._motion = MotionTask(
                tips,
                previous_error,
                self._basis,
                self._world_from_local,
                self.reference.rotation,
                self._motion_info,
                float(self.config.scale),
                0.0 if previous is None else self._dt,
            )

    def build_pair_targets(self, vectors, distances):
        return vectors

    def _update(self, state):
        if not np.array_equal(state, self._cached_x):
            self._compute(np.asarray(state, dtype=float))
            self._cached_x[:] = state

    def evaluate(self, state, residual, jacobian=None):
        self._update(state)
        residual[:] = self._r
        if jacobian is not None:
            jacobian[:] = self._j
        return True

    @property
    def target_positions(self):
        return self._targets.copy()

    def _compute(self, state):
        n = self.hand.nq
        q, eta = state[:n], state[-6:]
        self._kinematics.evaluate(q)
        local_tips = self.positions[:5] - self.base
        self._r.fill(0)
        self._j.fill(0)
        write_geometry(
            self.positions, self.point_jacobian, self._geometry, self._r, self._j
        )
        correction = self.correction(eta)
        rotation = correction.rotation
        right = pin.Jexp3(eta[3:])
        self.wrist = self.reference * correction
        tangent = np.zeros((6, 6))
        tangent[:3, :3], tangent[3:, 3:] = rotation.T, right
        write_residual_block(
            pin.log6(correction).vector,
            pin.Jlog6(correction) @ tangent,
            self._wrist_info,
            1.0,
            self._huber_delta,
            False,
            self._r[42:48],
            self._j[42:48, -6:],
        )
        if self._previous is not None:
            error = self._previous_correction_inverse * correction
            write_residual_block(
                pin.log6(error).vector,
                pin.Jlog6(error) @ tangent,
                self._wrist_time_info,
                1.0,
                self._huber_delta,
                False,
                self._r[48:54],
                self._j[48:54, -6:],
            )
            self._r[54 : 54 + 2 * n] = self._process_info @ (
                state[: 2 * n] - self._prediction
            )
            self._j[54 : 54 + 2 * n, : 2 * n] = self._process_info
        else:
            self._r[54 + n : 54 + 2 * n] = (
                self._initial_velocity_info @ state[n : 2 * n]
            )
            self._j[54 + n : 54 + 2 * n, n : 2 * n] = self._initial_velocity_info
        self.tip_error = None
        if self.motion_enabled:
            self.tip_error = self._tip_error
            start = self.blocks["motion"][0]
            write_motion(
                local_tips,
                self.point_jacobian[:5],
                rotation,
                right,
                eta[:3],
                self._motion,
                self.tip_error,
                self._r[start:],
                self._j[start:],
            )

    def commit(self, state):
        self._update(state)
        saved = state.copy()
        n = self.hand.nq
        return PreviousState(
            saved[:n],
            saved[n : 2 * n],
            saved[-6:],
            None if self.tip_error is None else self.tip_error.copy(),
        )

    def fit_errors(self, q):
        positions = self.positions
        if not np.array_equal(q, self._cached_x[: self.hand.nq]):
            self._fit_kinematics.evaluate(q, jacobian=False)
            positions = self._fit_kinematics.positions
        return dict(
            zip(
                ("tip_position_rms_m", "tip_direction_rms_rad", "pinch_rms_m"),
                measure_fit(positions, self._geometry),
            )
        )


class VelocityProjection:
    def __init__(self, problem: HandResidual, velocity_limits=None):
        self.problem = problem
        self.nq = problem.hand.nq
        self.n_parameters = self.nq + 6
        self.residual_count = problem.residual_count
        limits = (
            problem.hand.velocity_limits if velocity_limits is None else velocity_limits
        )
        self._velocity_limits = np.array(limits, dtype=float, copy=True)
        self._state = np.empty(problem.n_parameters)
        self._velocity_derivative = np.empty(self.nq)
        self._full_jacobian = np.empty((self.residual_count, problem.n_parameters))
        self._jacobian_view = self._full_jacobian.view()
        self._jacobian_view.flags.writeable = False

    @property
    def full_jacobian(self):
        return self._jacobian_view

    @property
    def step_amplification(self):
        return np.hypot(1.0, self.problem._velocity_gain)

    def expand(self, reduced, out=None):
        project_velocity(
            np.asarray(reduced, dtype=float),
            self.problem._prediction,
            self.problem._velocity_gain,
            self._velocity_limits,
            self._state,
            self._velocity_derivative,
        )
        if out is not None:
            out[:] = self._state
            return out
        return self._state

    def recover(self, reduced):
        return self.expand(reduced).copy()

    def finalize(self, result, bounds):
        state = self.recover(result.x)
        self.problem.evaluate(state, result.fun, self._full_jacobian)
        result.jac = self._full_jacobian.copy()
        result.grad = result.jac.T @ result.fun
        n = self.nq
        velocity_mask = np.zeros(n, dtype=int)
        velocity_mask[state[n : 2 * n] == -self._velocity_limits] = -1
        velocity_mask[state[n : 2 * n] == self._velocity_limits] = 1
        result.active_mask = np.concatenate(
            (
                result.active_mask[:n],
                velocity_mask,
                result.active_mask[-6:],
            )
        )
        gradient = result.grad.copy()
        gradient[
            (result.active_mask * gradient < 0) | (bounds.lower == bounds.upper)
        ] = 0.0
        result.optimality = float(np.linalg.norm(gradient, ord=np.inf))
        result.state, result.x = state, state[:n]
        result.velocity, result.wrist_correction = state[n : 2 * n], state[-6:]
        result.wrist = self.problem.wrist.homogeneous
        return self.problem.commit(state)

    def evaluate(self, reduced, residual, jacobian=None):
        state = self.expand(reduced)
        self.problem.evaluate(state, residual, self._full_jacobian)
        if jacobian is not None:
            n = self.nq
            np.multiply(
                self._full_jacobian[:, n : 2 * n],
                self._velocity_derivative,
                out=jacobian[:, :n],
            )
            jacobian[:, :n] += self._full_jacobian[:, :n]
            jacobian[:, -6:] = self._full_jacobian[:, -6:]
        return True
