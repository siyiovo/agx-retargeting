from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from agx_retargeting.types import FINGERS


@dataclass(slots=True)
class HandComparison:
    landmarks: np.ndarray
    robot_tips: np.ndarray
    target_tips: np.ndarray
    robot_directions: np.ndarray
    target_directions: np.ndarray
    metrics: dict


def compare_fingertips(model, config, output):
    targets = np.asarray(output["target_positions_m"], dtype=float)
    names = tuple(config.landmarks)
    frame_ids = [
        model.model.getFrameId(config.landmarks[name].link, pin.BODY) for name in names
    ]
    positions = np.empty((len(names), 3))
    q = np.array(
        [
            output["q_output"][output["joint_names"].index(name)]
            for name in model.joint_names
        ]
    )
    model.evaluate(q, frame_ids, positions)
    tips = [names.index(f"{finger}_tip") for finger in FINGERS]
    dips = [
        names.index("thumb_ip" if finger == "thumb" else f"{finger}_dip")
        for finger in FINGERS
    ]
    tip_keys = [config.landmarks[names[index]].keypoint for index in tips]
    dip_keys = [config.landmarks[names[index]].keypoint for index in dips]
    robot_tips = positions[tips]
    target_tips = targets[tip_keys]
    robot_directions = robot_tips - positions[dips]
    target_directions = target_tips - targets[dip_keys]
    for directions in (robot_directions, target_directions):
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths < 1e-10):
            raise ValueError("Fingertip direction requires a nonzero distal segment")
        directions /= lengths[:, None]
    cosine = np.einsum("ij,ij->i", robot_directions, target_directions)
    position_error = np.linalg.norm(robot_tips - target_tips, axis=1)
    direction_error = np.arccos(np.clip(cosine, -1, 1))
    robot_relative = robot_tips[1:] - robot_tips[0]
    target_relative = target_tips[1:] - target_tips[0]
    robot_distance = np.linalg.norm(robot_relative, axis=1)
    target_distance = np.linalg.norm(target_relative, axis=1)
    metrics = {
        "timestamp_s": output["timestamp_s"],
        "source": "joint_targets",
        "tip_position_mm": (position_error * 1000).tolist(),
        "tip_direction_deg": np.rad2deg(direction_error).tolist(),
        "tip_position_rms_m": float(np.sqrt(np.mean(position_error**2))),
        "tip_direction_rms_rad": float(np.sqrt(np.mean(direction_error**2))),
        "raw_fit": output["raw_fit"],
        "output_fit": output["output_fit"],
        "pinches": [
            {
                "finger": finger,
                "target_distance_mm": float(target_distance[index] * 1000),
                "robot_distance_mm": float(robot_distance[index] * 1000),
                "distance_error_mm": float(
                    (robot_distance[index] - target_distance[index]) * 1000
                ),
                "relative_position_error_mm": float(
                    np.linalg.norm(robot_relative[index] - target_relative[index])
                    * 1000
                ),
            }
            for index, finger in enumerate(FINGERS[1:])
        ],
    }
    return HandComparison(
        positions,
        robot_tips,
        target_tips,
        robot_directions,
        target_directions,
        metrics,
    )
