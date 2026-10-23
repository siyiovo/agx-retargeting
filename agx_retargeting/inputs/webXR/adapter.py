from typing import TYPE_CHECKING

import numpy as np

from agx_retargeting.types import HandInput

if TYPE_CHECKING:
    from xr_media.xr.frame import TrackedHand

WEBXR_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    *(
        f"{finger}-finger-{segment}"
        for finger in ("index", "middle", "ring", "pinky")
        for segment in (
            "phalanx-proximal",
            "phalanx-intermediate",
            "phalanx-distal",
            "tip",
        )
    ),
)


class WebXRAdapter:
    @staticmethod
    def adapt(hand: "TrackedHand", *, side: str, timestamp: float) -> HandInput:
        points = None
        quaternions = None
        if hand.tracked and all(name in hand.joints for name in WEBXR_JOINT_NAMES):
            points = np.array(
                [
                    (hand.joints[name].px, hand.joints[name].py, hand.joints[name].pz)
                    for name in WEBXR_JOINT_NAMES
                ],
                dtype=np.float64,
            )
            quaternions = np.array(
                [
                    (
                        hand.joints[name].qx,
                        hand.joints[name].qy,
                        hand.joints[name].qz,
                        hand.joints[name].qw,
                    )
                    for name in WEBXR_JOINT_NAMES
                ],
                dtype=np.float64,
            )
        return HandInput(
            points,
            side,
            timestamp,
            tracked=points is not None,
            frame_id="webxr_z_up",
            quaternions=quaternions,
        )
