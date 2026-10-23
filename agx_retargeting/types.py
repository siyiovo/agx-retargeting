from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pinocchio as pin

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ROLES = (
    ("cmc", "mcp", "ip", "tip"),
    *(("mcp", "pip", "dip", "tip"),) * 4,
)
KEYPOINT_NAMES = (
    "wrist",
    *(
        f"{finger}_{role}"
        for finger, roles in zip(FINGERS, FINGER_ROLES)
        for role in roles
    ),
)
_UNIT_TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}


@dataclass(frozen=True, slots=True)
class HandInput:
    landmarks: np.ndarray | None
    side: str
    timestamp: float
    tracked: bool = True
    frame_id: str = ""
    quaternions: np.ndarray | None = None
    velocities: np.ndarray | None = None

    def __post_init__(self):
        if not self.tracked:
            object.__setattr__(self, "landmarks", None)
            object.__setattr__(self, "quaternions", None)
            object.__setattr__(self, "velocities", None)
            return
        points = np.array(self.landmarks, dtype=np.float64, copy=True)
        points.flags.writeable = False
        object.__setattr__(self, "landmarks", points)
        if self.quaternions is not None:
            rotations = np.array(self.quaternions, dtype=np.float64, copy=True)
            lengths = np.linalg.norm(rotations, axis=1, keepdims=True)
            if np.any(lengths == 0):
                raise ValueError("quaternions must have non-zero norms")
            rotations /= lengths
            rotations.flags.writeable = False
            object.__setattr__(self, "quaternions", rotations)
        if self.velocities is not None:
            velocities = np.array(self.velocities, dtype=np.float64, copy=True)
            velocities.flags.writeable = False
            object.__setattr__(self, "velocities", velocities)

    @classmethod
    def from_positions(
        cls,
        positions,
        side: str,
        timestamp: float,
        *,
        unit: str = "m",
        tracked: bool = True,
        frame_id: str = "",
        quaternions: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> "HandInput":
        if not tracked:
            return cls(None, side, timestamp, tracked, frame_id)
        points = np.asarray(positions, dtype=np.float64)
        scale = _UNIT_TO_METRES[unit]
        speed = None if velocities is None else np.asarray(velocities)
        if scale != 1.0:
            points = points * scale
            if speed is not None:
                speed = speed * scale
        return cls(points, side, timestamp, tracked, frame_id, quaternions, speed)


@dataclass(frozen=True, slots=True)
class Keypoint:
    position: np.ndarray
    velocity: np.ndarray | None
    quaternion: np.ndarray | None
    timestamp: float
    type: Literal["input", "output"]


class InputModel(Protocol):
    def read(self, timeout: float = 0.1) -> Mapping[str, "HandInput"] | None: ...

    def close(self) -> None: ...


class OutputModel(Protocol):
    nq: int
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    velocity_limits: np.ndarray
    neutral_q: np.ndarray

    def evaluate(
        self,
        q: np.ndarray,
        frame_ids: Sequence[int] | np.ndarray,
        positions: np.ndarray,
        jacobians: np.ndarray | None = None,
    ) -> None: ...

    def evaluate_poses(
        self,
        q: np.ndarray,
        frame_ids: Sequence[int] | np.ndarray,
        poses: list[pin.SE3],
        jacobians: np.ndarray | None = None,
    ) -> None: ...


class Residual(Protocol):
    residual_count: int

    def evaluate(
        self,
        parameters: np.ndarray,
        residual: np.ndarray,
        jacobian: np.ndarray | None = None,
    ) -> bool: ...
