import argparse
import asyncio
import json
import logging
import signal
import ssl
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from time import monotonic

import aiohttp
import numpy as np
import viser
from viser.extras import ViserUrdf
from viser.transforms import SO3

from agx_retargeting.config import CONFIG_DIR
from agx_retargeting.core import HandConfig, HandModel
from agx_retargeting.inputs.webXR.adapter import WEBXR_JOINT_NAMES
from agx_retargeting.types import HandInput

from .metrics import HandComparison, compare_fingertips

log = logging.getLogger("agx.visualization")
PALM_DISPLAY = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FINGER_LABELS = ("拇指", "食指", "中指", "无名指", "小指")
FINGER_COLORS = np.array(
    [(255, 65, 180), (0, 220, 255), (255, 215, 0), (100, 255, 65), (175, 125, 255)],
    dtype=np.uint8,
)
CHAINS = (
    (
        "wrist",
        "thumb-metacarpal",
        "thumb-phalanx-proximal",
        "thumb-phalanx-distal",
        "thumb-tip",
    ),
    *(
        (
            "wrist",
            *(
                f"{finger}-finger-{segment}"
                for segment in (
                    "metacarpal",
                    "phalanx-proximal",
                    "phalanx-intermediate",
                    "phalanx-distal",
                    "tip",
                )
            ),
        )
        for finger in ("index", "middle", "ring", "pinky")
    ),
)
JOINT_NAMES = tuple(dict.fromkeys(name for chain in CHAINS for name in chain))
EDGES = np.array(
    [
        (JOINT_NAMES.index(start), JOINT_NAMES.index(end))
        for chain in CHAINS
        for start, end in pairwise(chain)
    ]
)
JOINT_COLORS = np.array(
    [
        (255, 255, 255)
        if name == "wrist"
        else FINGER_COLORS[next(i for i, chain in enumerate(CHAINS) if name in chain)]
        for name in JOINT_NAMES
    ],
    dtype=np.uint8,
)
TIP_INDICES = np.array([JOINT_NAMES.index(chain[-1]) for chain in CHAINS])


def upright_skeleton(hand, side, scale):
    joints = hand["joints"]
    frame = HandInput(
        np.array([joints[name][:3] for name in WEBXR_JOINT_NAMES]),
        side,
        hand["timestamp_s"],
        quaternions=np.array([joints[name][3:] for name in WEBXR_JOINT_NAMES]),
    )
    basis = HandModel.estimate_coord(frame)
    points = np.array([joints[name][:3] for name in JOINT_NAMES])
    return (points - points[0]) @ basis @ PALM_DISPLAY.T * scale


class HandScene:
    def __init__(self, server, config):
        self.server = server
        self.config = config
        self.model = HandModel(config.urdf_path)
        self.comparison: HandComparison | None = None
        self.robot_timestamp = None
        self.skeleton_timestamp = None
        self.robot_updates = 0
        self.skeleton_updates = 0
        self.robot_root = server.scene.add_frame(
            "/robot",
            show_axes=False,
            position=(-0.17, 0.0, 0.0),
            wxyz=SO3.from_matrix(PALM_DISPLAY @ config.robot_basis.T).wxyz,
        )
        self.robot = ViserUrdf(
            server,
            config.urdf_path,
            root_node_name="/robot",
            mesh_color_override=(0.65, 0.70, 0.78, 0.45),
        )
        self.robot.update_cfg(
            self.model.neutral_q[
                [
                    self.model.joint_names.index(name)
                    for name in self.robot.get_actuated_joint_names()
                ]
            ]
        )
        neutral_positions = np.array(
            [
                self.robot._urdf.get_transform(landmark.link)[:3, 3]
                for landmark in config.landmarks.values()
            ]
        )
        self.skeleton_root = server.scene.add_frame(
            "/webxr",
            show_axes=False,
            position=(0.17, 0.0, 0.0),
            visible=False,
        )
        self.bones = server.scene.add_line_segments(
            "/webxr/bones",
            points=np.empty((0, 2, 3)),
            colors=(56, 189, 248),
            thickness=0.0025,
        )
        self.joints = server.scene.add_point_cloud(
            "/webxr/joints",
            points=np.empty((0, 3)),
            colors=np.empty((0, 3), dtype=np.uint8),
            point_size=0.006,
            point_shape="circle",
            precision="float32",
        )
        self.skeleton_tips = server.scene.add_point_cloud(
            "/webxr/tips",
            points=np.zeros((5, 3)),
            colors=FINGER_COLORS,
            point_size=0.011,
            point_shape="circle",
            precision="float32",
        )
        landmark_colors = np.array(
            [
                (255, 255, 255)
                if landmark.keypoint == 0
                else FINGER_COLORS[(landmark.keypoint - 1) // 4]
                for landmark in config.landmarks.values()
            ],
            dtype=np.uint8,
        )
        self.robot_joints = server.scene.add_point_cloud(
            "/robot/joints",
            points=neutral_positions,
            colors=landmark_colors,
            point_size=0.006,
            point_shape="circle",
            precision="float32",
        )
        self.robot_tips = server.scene.add_point_cloud(
            "/robot/tips",
            points=neutral_positions[
                [
                    tuple(config.landmarks).index(f"{finger}_tip")
                    for finger in ("thumb", "index", "middle", "ring", "pinky")
                ]
            ],
            colors=FINGER_COLORS,
            point_size=0.011,
            point_shape="circle",
            precision="float32",
        )
        self.comparison_root = server.scene.add_frame(
            "/robot/comparison", show_axes=False, visible=False
        )
        self.target_tips = server.scene.add_point_cloud(
            "/robot/comparison/targets",
            points=np.zeros((5, 3)),
            colors=(255, 255, 255),
            point_size=0.009,
            point_shape="diamond",
            precision="float32",
        )
        self.tip_errors = server.scene.add_line_segments(
            "/robot/comparison/errors",
            points=np.zeros((5, 2, 3)),
            colors=(255, 95, 55),
            thickness=2.5,
            thickness_units="screen",
        )
        self.robot_directions = server.scene.add_line_segments(
            "/robot/comparison/robot_directions",
            points=np.zeros((5, 2, 3)),
            colors=np.repeat(FINGER_COLORS[:, None], 2, axis=1),
            thickness=3.5,
            thickness_units="screen",
        )
        self.target_directions = server.scene.add_line_segments(
            "/robot/comparison/target_directions",
            points=np.zeros((5, 2, 3)),
            colors=(255, 255, 255),
            thickness=1.5,
            thickness_units="screen",
        )
        self.pinch_lines = server.scene.add_line_segments(
            "/robot/comparison/pinch",
            points=np.zeros((2, 2, 3)),
            colors=(255, 255, 255),
            thickness=2,
            thickness_units="screen",
        )
        side_name = "左手" if config.side == "left" else "右手"
        self.robot_label = server.scene.add_label(
            "/robot_label",
            text=f"Revo2 {side_name} · 中立模型"
            if "revo2" in config.urdf_path.name
            else f"机器人{side_name} · 中立模型",
            position=(-0.17, 0.0, -0.055),
            anchor="center-center",
        )
        server.scene.add_label(
            "/webxr_label",
            text=f"WebXR {side_name} · 25 关节",
            position=(0.17, 0.0, -0.055),
            anchor="center-center",
        )
        server.scene.set_up_direction("+z")
        server.scene.world_axes.visible = False
        server.scene.add_grid(
            "/grid",
            width=0.9,
            height=0.55,
            plane="xz",
            cell_size=0.025,
            section_size=0.1,
            position=(0.0, 0.08, 0.1),
        )
        server.gui.configure_theme(
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            control_width="medium",
        )
        server.gui.add_markdown(
            "**掌面竖直 · 手指朝上 · 左右并排**\n\n"
            "拇指：品红 · 食指：青色 · 中指：黄色 · 无名指：绿色 · 小指：紫色\n\n"
            "彩色圆点：机器人关节目标；白色菱形：观测掌面中的指尖目标；"
            "橙色连线：指尖位置误差；指尖短线：末节骨段方向。"
        )
        self.status = server.gui.add_markdown("等待手部追踪")
        self.tip_metrics = server.gui.add_markdown("指尖误差等待同帧输入")
        self.pinch = server.gui.add_dropdown(
            "捏合连线", options=FINGER_LABELS[1:], initial_value="食指"
        )
        self.pinch.on_update(lambda _: self.show_pinch())
        self.pinch_metrics = server.gui.add_markdown("捏合误差等待同帧输入")
        server.gui.add_markdown(
            "误差使用同帧观测掌面中的指尖目标，以及关节目标的 FK。"
            "方向误差衡量 DIP 至 TIP 骨段的夹角，拇指使用 IP 至 TIP。"
            "真机位置反馈尚未标定为关节角度。"
        )
        reset = server.gui.add_button("恢复竖直视角")
        reset.on_click(lambda _: self.reset_camera())
        self.reset_camera()

    def reset_camera(self):
        position = (0.0, -0.65, 0.10)
        look_at = (0.0, 0.0, 0.10)
        self.server.initial_camera.position = position
        self.server.initial_camera.look_at = look_at
        for client in self.server.get_clients().values():
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def show_pinch(self):
        if self.comparison is None:
            return
        index = FINGER_LABELS.index(self.pinch.value)
        self.pinch_lines.points = np.array(
            [
                self.comparison.robot_tips[[0, index]],
                self.comparison.target_tips[[0, index]],
            ]
        )
        self.pinch_lines.colors = np.repeat(
            np.array([FINGER_COLORS[index], (255, 255, 255)], dtype=np.uint8)[:, None],
            2,
            axis=1,
        )

    def update_comparison(self, output):
        comparison = compare_fingertips(self.model, self.config, output)
        self.comparison = comparison
        self.robot_joints.points = comparison.landmarks
        self.robot_tips.points = comparison.robot_tips
        self.target_tips.points = comparison.target_tips
        self.tip_errors.points = np.stack(
            (comparison.robot_tips, comparison.target_tips), axis=1
        )
        self.robot_directions.points = np.stack(
            (
                comparison.robot_tips,
                comparison.robot_tips + 0.018 * comparison.robot_directions,
            ),
            axis=1,
        )
        self.target_directions.points = np.stack(
            (
                comparison.target_tips,
                comparison.target_tips + 0.018 * comparison.target_directions,
            ),
            axis=1,
        )
        self.comparison_root.visible = True
        self.show_pinch()
        metrics = comparison.metrics
        self.tip_metrics.content = (
            "**指尖误差**\n\n| 手指 | 位置 mm | 方向 ° |\n"
            "| --- | ---: | ---: |\n"
            + "\n".join(
                f"| {name} | {position:.1f} | {angle:.1f} |"
                for name, position, angle in zip(
                    FINGER_LABELS,
                    metrics["tip_position_mm"],
                    metrics["tip_direction_deg"],
                )
            )
        )
        self.pinch_metrics.content = (
            "**拇指捏合**\n\n"
            "| 对指 | 目标间距 | FK 间距 | 相对位置误差 |\n"
            "| --- | ---: | ---: | ---: |\n"
            + "\n".join(
                f"| {name} | {pinch['target_distance_mm']:.1f} | "
                f"{pinch['robot_distance_mm']:.1f} | "
                f"{pinch['relative_position_error_mm']:.1f} |"
                for name, pinch in zip(FINGER_LABELS[1:], metrics["pinches"])
            )
            + "\n\n单位 mm；相对位置误差为两组拇指至对指向量之差的长度。"
        )

    def update(self, state):
        output = state["hands"][self.config.side]
        hand = state["input"].get(self.config.side)
        complete = (
            hand is not None
            and hand["tracked"]
            and all(name in hand["joints"] for name in JOINT_NAMES)
        )
        with self.server.atomic():
            if output["tracking"] and output["timestamp_s"] != self.robot_timestamp:
                names = output["joint_names"]
                q = np.asarray(output["q_output"], dtype=float)
                self.robot.update_cfg(
                    q[
                        [
                            names.index(name)
                            for name in self.robot.get_actuated_joint_names()
                        ]
                    ]
                )
                self.update_comparison(output)
                self.robot_label.text = self.robot_label.text.replace(
                    "中立模型", "关节目标"
                )
                self.robot_root.visible = True
                self.robot_timestamp = output["timestamp_s"]
                self.robot_updates += 1
            self.skeleton_root.visible = complete
            if complete and hand["timestamp_s"] != self.skeleton_timestamp:
                points = upright_skeleton(hand, self.config.side, self.config.scale)
                self.bones.points = points[EDGES]
                self.bones.colors = np.repeat(
                    JOINT_COLORS[EDGES[:, 1], None], 2, axis=1
                )
                self.joints.points = points
                self.joints.colors = JOINT_COLORS
                self.skeleton_tips.points = points[TIP_INDICES]
                self.skeleton_timestamp = hand["timestamp_s"]
                self.skeleton_updates += 1
            if not output["tracking"] or not complete:
                self.comparison_root.visible = False
                self.comparison = None
                self.tip_metrics.content = "指尖误差等待同帧输入"
                self.pinch_metrics.content = "捏合误差等待同帧输入"
        age = state["input_age_s"]
        self.status.content = (
            f"**追踪正常** · 输入年龄 {age * 1000:.0f} ms\n\n机器人显示求解得到的关节目标。"
            if complete and output["tracking"]
            else "**等待手部追踪**\n\n机器人保留最后收到的关节目标。"
            if self.robot_timestamp is not None
            else "**等待 WebXR 数据**\n\n机器人显示 URDF 中立姿态。"
            "请在 Quest 页面重新进入 XR，并让所选手部保持可见。"
        )
        return {
            "side": self.config.side,
            "tracking": bool(complete),
            "robot_updates": self.robot_updates,
            "skeleton_updates": self.skeleton_updates,
            "skeleton_joints": len(JOINT_NAMES) if complete else 0,
            "skeleton_bones": len(EDGES) if complete else 0,
            "robot_timestamp_s": self.robot_timestamp,
            "skeleton_timestamp_s": self.skeleton_timestamp,
            "clients": len(self.server.get_clients()),
            "robot_visible": self.robot_root.visible,
            "robot_pose_source": "urdf_neutral"
            if self.robot_timestamp is None
            else "joint_targets",
            "comparison": None if self.comparison is None else self.comparison.metrics,
        }


@contextmanager
def open_viewer(config, host="127.0.0.1", port=8080):
    server = viser.ViserServer(host=host, port=port, label="AGX 手部观察")
    try:
        if server.get_port() != port:
            raise OSError(f"Viewer port {port} is already in use")
        yield HandScene(server, config)
    finally:
        server.stop()


def parser():
    result = argparse.ArgumentParser(
        description="Display upright robot joints, WebXR bones and fingertip errors."
    )
    result.add_argument("--source", default="https://localhost:9443")
    result.add_argument("--cert", type=Path)
    result.add_argument("--model", choices=("revo2", "wuji_hand"), default="revo2")
    result.add_argument("--hand", choices=("left", "right"), default="left")
    result.add_argument("--config", type=Path)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8080)
    result.add_argument("--status-file", type=Path)
    return result


async def run(args):
    config = HandConfig.load(
        args.config or CONFIG_DIR / f"{args.model}_params.yaml", args.hand
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    with open_viewer(config, args.host, args.port) as scene:
        context = ssl.create_default_context(cafile=args.cert)
        last_report = 0.0
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3)
        ) as client:
            while not stop.is_set():
                async with client.get(
                    args.source.rstrip("/") + "/api/retargeting/state", ssl=context
                ) as response:
                    response.raise_for_status()
                    report = scene.update(await response.json())
                if monotonic() - last_report >= 1:
                    log.info("%s", json.dumps(report))
                    if args.status_file:
                        temporary = args.status_file.with_suffix(".partial")
                        temporary.write_text(json.dumps(report) + "\n")
                        temporary.replace(args.status_file)
                    last_report = monotonic()
                await asyncio.sleep(1 / 60)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="[agx-view] %(message)s")
    asyncio.run(run(parser().parse_args(argv)))


if __name__ == "__main__":
    main()
