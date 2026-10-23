from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from pyAgxArm.protocols.can_protocol.drivers import Revo2DriverDefault

    from agx_retargeting.core import HandModel

CHANNELS = (
    "thumb_tip",
    "thumb_base",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "pinky_finger",
)
_JOINTS = (
    "thumb_proximal_joint",
    "thumb_metacarpal_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
)


@dataclass(frozen=True)
class Revo2Feedback:
    """Copied SDK measurements in CHANNELS order, retaining source timestamps."""

    position: tuple[int, ...] | None
    position_timestamp: float | None
    motor_status: tuple[int, ...] | None
    status_timestamp: float | None
    left_or_right: int | None


class Revo2Adapter:
    def __init__(self, hand: "HandModel", *, side: str):
        joint_names = tuple(hand.joint_names)
        expected = tuple(f"{side}_{name}" for name in _JOINTS)
        self._indices = np.array([joint_names.index(name) for name in expected])
        self._endpoints = np.array(
            [hand.lower_limits[self._indices], hand.upper_limits[self._indices]]
        )

    def adapt(self, q: np.ndarray) -> tuple[int, ...]:
        q = np.asarray(q, dtype=float)
        lower, upper = self._endpoints
        return tuple(
            int(value)
            for value in np.rint(
                np.clip(100 * (q[self._indices] - lower) / (upper - lower), 0, 100)
            )
        )


class Revo2Output:
    def __init__(
        self,
        driver: "Revo2DriverDefault",
        hand: "HandModel",
        *,
        side: str,
        arrival_time: float = 0.04,
    ):
        self.adapter = Revo2Adapter(hand, side=side)
        self._time_ticks = int(np.rint(arrival_time * 100))
        self._driver = driver
        self._send_lock = Lock()

    @property
    def arrival_time(self) -> float:
        return self._time_ticks / 100

    def write(self, q) -> tuple[int, ...]:
        positions = self.adapter.adapt(q)
        with self._send_lock:
            started = monotonic()
            commands: tuple[tuple[Literal["pos", "time"], dict[str, int]], ...] = (
                ("pos", dict(zip(CHANNELS, positions))),
                ("time", dict.fromkeys(CHANNELS, self._time_ticks)),
            )
            for mode, values in commands:
                self._driver.position_time_ctrl(mode=mode, **values)
                error = self._driver._ctx.get_comm_error()
                if error is not None:
                    raise error
            if monotonic() - started > 0.05:
                raise TimeoutError(
                    "Revo2 position/time pair exceeded 50 ms; device acceptance is unconfirmed"
                )
        return positions

    def read(self) -> Revo2Feedback:
        """Copy available feedback; each stream keeps its own SDK timestamp.

        SDK caches update asynchronously; the two streams are not synchronized.
        """
        position = self._driver.get_finger_pos()
        status = self._driver.get_hand_status()
        return Revo2Feedback(
            position=(
                tuple(int(getattr(position.msg, name)) for name in CHANNELS)
                if position is not None
                else None
            ),
            position_timestamp=(
                float(position.timestamp) if position is not None else None
            ),
            motor_status=(
                tuple(int(getattr(status.msg, name)) for name in CHANNELS)
                if status is not None
                else None
            ),
            status_timestamp=float(status.timestamp) if status is not None else None,
            left_or_right=int(status.msg.left_or_right) if status is not None else None,
        )
