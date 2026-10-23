from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pinocchio as pin
import yaml
from scipy.spatial.transform import Rotation

from ..types import FINGERS, KEYPOINT_NAMES, HandInput


@dataclass(frozen=True)
class Landmark:
    link: str
    keypoint: int


@dataclass(frozen=True)
class HandConfig:
    side: str
    urdf_path: Path
    landmarks: dict[str, Landmark]
    robot_basis: np.ndarray
    scale: float
    velocity_decay_s: float | None = None

    @classmethod
    def load(cls, path: str | Path, side: str) -> "HandConfig":
        """Expand links and resolve URDFs under description/urdf/."""
        path = Path(path).expanduser().resolve(strict=True)
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        model = document["model"]
        preprocess = document.get("preprocess", {})
        substitutions = {"side": side, "side_short": side[0]}
        urdf_path = Path(model["urdf_path"].format(**substitutions)).expanduser()
        if not urdf_path.is_absolute():
            urdf_root = path.parent.parent / "description/urdf"
            if len(urdf_path.parts) == 1:
                matches = [p for p in urdf_root.rglob(urdf_path.name) if p.is_file()]
                if not matches:
                    raise FileNotFoundError(
                        f"URDF '{urdf_path}' not found under {urdf_root}"
                    )
                if len(matches) > 1:
                    raise ValueError(
                        f"URDF '{urdf_path}' is ambiguous; "
                        "include its subdirectory under description/urdf/"
                    )
                urdf_path = matches[0]
            else:
                urdf_path = urdf_root / urdf_path
        urdf_path = urdf_path.resolve(strict=True)
        links = model["links"]
        fingers = links["fingers"]

        def link_name(template, finger=""):
            return template.format(finger=finger, **substitutions)

        landmarks = {"wrist": Landmark(link_name(links["palm"]), 0)}
        finger_names = dict(zip(FINGERS, fingers))
        for keypoint, name in enumerate(KEYPOINT_NAMES[1:], start=1):
            logical, role = name.rsplit("_", 1)
            template = links.get(logical, {}).get(role, links.get(role))
            if template is not None:
                landmarks[name] = Landmark(
                    link_name(template, finger_names[logical]), keypoint
                )

        def rotation_matrix(value):
            value = value.get(side, value)
            return Rotation.from_euler(
                "xyz", [value.get(axis, 0.0) for axis in "xyz"], degrees=True
            ).as_matrix()

        if "basis_rotation" in preprocess:
            basis = rotation_matrix(preprocess["basis_rotation"])
        else:
            hand = HandModel(urdf_path)
            names = (
                "wrist",
                *model.get(
                    "palm_landmarks",
                    ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"),
                ),
            )
            frame_ids = [
                hand.model.getFrameId(landmarks[name].link, pin.BODY) for name in names
            ]
            positions = np.empty((5, 3))
            hand.evaluate(hand.neutral_q, frame_ids, positions)
            points = np.zeros((21, 3))
            points[[0, 5, 9, 13, 17]] = positions
            basis = hand.estimate_coord(HandInput(points, side, 0.0))
        basis = rotation_matrix(preprocess.get("input_rotation", {})) @ basis
        basis.flags.writeable = False
        scale = float(preprocess.get("scale", 1.0))
        return cls(
            side,
            urdf_path,
            landmarks,
            basis,
            scale,
            document.get("motion", {}).get("velocity_decay_s"),
        )


class HandModel:
    """Use native Pinocchio mimic for FK and independent-coordinate Jacobians."""

    def __init__(self, urdf_path: str | Path):
        urdf_path = Path(urdf_path).expanduser().resolve(strict=True)
        nodes = {
            joint.attrib["name"]: joint
            for joint in ElementTree.parse(urdf_path).getroot().findall("joint")
            if joint.attrib["type"] != "fixed"
        }
        for name, node in nodes.items():
            if node.attrib["type"] not in ("revolute", "prismatic"):
                raise ValueError(f"Joint '{name}' must be revolute or prismatic")
            mimic = node.find("mimic")
            if mimic is not None:
                source = mimic.attrib["joint"]
                if nodes[source].find("mimic") is not None:
                    raise ValueError(
                        f"Mimic joint '{name}' must reference an independent joint"
                    )

        self.model = pin.buildModelFromUrdf(str(urdf_path), mimic=True)
        self.data = self.model.createData()
        self.joint_names = tuple(
            name
            for name in tuple(self.model.names)[1:]
            if nodes[name].find("mimic") is None
        )
        self.nq = self.model.nq

        self.lower_limits, self.upper_limits, self.velocity_limits = self._build_limits(
            nodes
        )
        self.neutral_q = np.clip(
            np.zeros(self.nq), self.lower_limits, self.upper_limits
        )
        for array in (
            self.lower_limits,
            self.upper_limits,
            self.velocity_limits,
            self.neutral_q,
        ):
            array.flags.writeable = False

    def _build_limits(self, nodes):
        """Intersect each joint's limits in independent-coordinate space."""
        lower = np.full(self.nq, -np.inf)
        upper = np.full(self.nq, np.inf)
        velocity = np.full(self.nq, np.inf)
        for name, node in nodes.items():
            mimic = node.find("mimic")
            if mimic is None:
                source, scale, offset = name, 1.0, 0.0
            else:
                source = mimic.attrib["joint"]
                scale = float(mimic.get("multiplier", "1"))
                offset = float(mimic.get("offset", "0"))
            limit = node.find("limit")
            lo = float(limit.attrib["lower"])
            hi = float(limit.attrib["upper"])
            speed = float(limit.attrib["velocity"])
            if lo > hi or speed < 0:
                raise ValueError(f"Invalid limits for '{name}'")
            if scale == 0:
                if not lo <= offset <= hi:
                    raise ValueError(
                        f"Constant mimic joint '{name}' violates its limits"
                    )
                continue
            column = self.model.joints[self.model.getJointId(source)].idx_q
            endpoints = ((lo - offset) / scale, (hi - offset) / scale)
            lower[column] = max(lower[column], min(endpoints))
            upper[column] = min(upper[column], max(endpoints))
            velocity[column] = min(velocity[column], speed / abs(scale))
        if (
            np.any(lower > upper)
            or np.isposinf(lower).any()
            or np.isneginf(upper).any()
        ):
            raise ValueError("Joint and mimic limits have no feasible intersection")
        return lower, upper, velocity

    def bind_frames(self, frame_ids):
        return FrameKinematics(self, frame_ids)

    @staticmethod
    def estimate_coord(hand_input: HandInput) -> np.ndarray:
        """Return palm axes using a four-point plane and wrist-to-middle-MCP ray."""

        points = hand_input.landmarks[[0, 5, 13, 17]]
        _, _, vh = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
        normal = vh[-1]

        longitudinal = hand_input.landmarks[9] - hand_input.landmarks[0]
        z = longitudinal - normal * (normal @ longitudinal)
        z /= np.linalg.norm(z)
        y = np.cross(z, normal)
        lateral = (points[1] - points[2]) * (1 if hand_input.side == "right" else -1)
        if y @ lateral < 0:
            normal *= -1
            y *= -1
        return np.column_stack((normal, y, z))

    def build_targets(
        self, hand_input: HandInput, config: HandConfig, targets: np.ndarray
    ) -> bool:
        """Write all 21 target positions in the URDF root frame, in metres."""

        if hand_input.landmarks is None:
            return False
        palm_id = self.model.getFrameId(config.landmarks["wrist"].link, pin.BODY)
        palm = self.model.frames[palm_id]
        if palm.parentJoint != 0:
            raise ValueError("The configured palm must be fixed to the URDF root")

        coord = self.estimate_coord(hand_input)
        np.subtract(hand_input.landmarks, hand_input.landmarks[0], out=targets)
        np.matmul(targets, coord, out=targets)
        np.matmul(targets, config.robot_basis.T, out=targets)
        targets *= config.scale
        targets += palm.placement.translation
        return True

    def evaluate(
        self,
        q: np.ndarray,
        frame_ids: Sequence[int] | np.ndarray,
        positions: np.ndarray,
        jacobians: np.ndarray | None = None,
    ) -> None:
        """evaluate each frame's positions and optional jacobians dp/dq (F, 3, nq)."""

        q = np.asarray(q, dtype=np.float64)
        if jacobians is None:
            pin.forwardKinematics(self.model, self.data, q)
        else:
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for i, frame_id in enumerate(frame_ids):
            positions[i] = self.data.oMf[int(frame_id)].translation
            if jacobians is not None:
                jacobians[i] = pin.getFrameJacobian(
                    self.model, self.data, int(frame_id), pin.LOCAL_WORLD_ALIGNED
                ).reshape(6, self.nq)[:3]

    def evaluate_poses(
        self,
        q: np.ndarray,
        frame_ids: Sequence[int] | np.ndarray,
        poses: list[pin.SE3],
        jacobians: np.ndarray | None = None,
    ) -> None:
        q = np.asarray(q, dtype=np.float64)
        if jacobians is None:
            pin.forwardKinematics(self.model, self.data, q)
        else:
            pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for row, frame_id in enumerate(frame_ids):
            transform = self.data.oMf[int(frame_id)]
            poses[row].rotation = transform.rotation
            poses[row].translation = transform.translation
            if jacobians is not None:
                jacobians[row] = pin.getFrameJacobian(
                    self.model, self.data, int(frame_id), pin.LOCAL
                ).reshape(6, self.nq)


class FrameKinematics:
    def __init__(self, hand: HandModel, frame_ids):
        self.hand = hand
        self._ids = tuple(map(int, frame_ids))
        self._data = hand.model.createData()
        self._placements = tuple(self._data.oMf[index] for index in self._ids)
        self.positions = np.empty((len(self._ids), 3))
        self.jacobians = np.empty((len(self._ids), 3, hand.nq))

    def evaluate(self, q, *, jacobian=True):
        q = np.asarray(q, dtype=np.float64)
        model, data = self.hand.model, self._data
        if jacobian:
            pin.computeJointJacobians(model, data, q)
        else:
            pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        for row, (frame_id, placement) in enumerate(zip(self._ids, self._placements)):
            self.positions[row] = placement.translation
            if jacobian:
                self.jacobians[row] = pin.getFrameJacobian(
                    model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).reshape(6, self.hand.nq)[:3]
