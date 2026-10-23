import argparse
import json
import logging
import os
import signal
import subprocess
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import yaml

from agx_retargeting.core import HandConfig, Retargeter
from agx_retargeting.outputs import CHANNELS, Revo2Output

from .session import SessionConfig

log = logging.getLogger("agx.webxr")


def parser():
    result = argparse.ArgumentParser(
        description="Run Quest hand tracking through the AGX retargeter.",
        argument_default=argparse.SUPPRESS,
    )
    result.add_argument("--session", type=Path, default=None)
    result.add_argument("--model", choices=("revo2", "wuji_hand"))
    result.add_argument("--hand", choices=("both", "left", "right"))
    result.add_argument("--config", type=Path, help="Override the model YAML")
    result.add_argument("--params", type=Path)
    result.add_argument("--host")
    result.add_argument("--port", type=int)
    result.add_argument("--tracking-timeout", type=float, metavar="SECONDS")
    result.add_argument("--cert", type=Path)
    result.add_argument("--key", type=Path)
    result.add_argument(
        "--init-certs",
        action="store_true",
        default=False,
        help="Create missing development TLS files; keep existing files",
    )
    result.add_argument(
        "--output", type=Path, help="Write per-frame joint results as JSON Lines"
    )
    result.add_argument(
        "--revo2-left-can", metavar="CHANNEL", help="Enable left Revo2 output"
    )
    result.add_argument(
        "--revo2-right-can", metavar="CHANNEL", help="Enable right Revo2 output"
    )
    result.add_argument("--revo2-arrival-time", type=float, metavar="SECONDS")
    result.add_argument("--revo2-robot", help="SDK robot profile for hand connections")
    result.add_argument(
        "--revo2-firmware",
        help="SDK firmware profile for hand connections",
    )
    result.add_argument("--viewer", action=argparse.BooleanOptionalAction)
    result.add_argument("--viewer-hand", choices=("left", "right"))
    result.add_argument("--viewer-host")
    result.add_argument("--viewer-port", type=int)
    return result


def initialize_certificates():
    from xr_media.paths import static_dir, tls_dir

    target = tls_dir()
    paths = [target / "cert.pem", target / "key.pem"]
    if all(path.is_file() for path in paths):
        log.info("Using existing certificates: %s", target)
        return
    if any(path.exists() for path in paths):
        raise ValueError(
            "Incomplete TLS pair; specify a complete pair with --cert and --key"
        )
    script = static_dir().parent / "scripts/make_dev_certs.sh"
    subprocess.run(
        ["bash", str(script)],
        check=True,
        env={**os.environ, "XR_MEDIA_TLS_DIR": str(target)},
    )


@contextmanager
def revo2_outputs(
    retargeters, channels, *, arrival_time=0.04, robot="piper", firmware="default"
):
    """Own explicitly selected SDK connections for this application session."""
    if not channels:
        yield {}
        return
    if not channels.keys() <= retargeters.keys():
        raise ValueError("Revo2 output sides must be selected by --hand")
    if len(set(channels.values())) != len(channels):
        raise ValueError("Each Revo2 hand needs a distinct CAN channel")
    from pyAgxArm import AgxArmFactory, create_agx_arm_config

    with ExitStack() as stack:
        outputs = {}
        for side, channel in channels.items():
            arm = AgxArmFactory.create_arm(
                create_agx_arm_config(
                    robot=robot, channel=channel, firmeware_version=firmware
                )
            )
            stack.callback(arm.disconnect)
            driver = arm.init_effector(arm.OPTIONS.EFFECTOR.REVO2)
            outputs[side] = Revo2Output(
                driver, retargeters[side].hand, side=side, arrival_time=arrival_time
            )
            arm.connect()
            log.info(
                "%s Revo2: robot=%s firmware=%s channel=%s arrival_time=%g s",
                side,
                robot,
                firmware,
                channel,
                outputs[side].arrival_time,
            )
        yield outputs


def solve_frame(retargeter, frame, output=None):
    result = retargeter.step(frame)
    payload = {
        "tracking": result is not None,
        "timestamp_s": frame.timestamp,
        "joint_names": list(retargeter.hand.joint_names),
        "q_output": None,
    }
    if result is not None:
        payload.update(
            input_landmarks_m=frame.landmarks.tolist(),
            input_quaternions_xyzw=None if frame.quaternions is None else frame.quaternions.tolist(),
            target_positions_m=result.target_positions.tolist(),
            q_raw=result.x.tolist(),
            q_output=result.q_output.tolist(),
            velocity_rad_s=result.velocity.tolist(),
            wrist_pose=result.wrist.tolist(),
            wrist_correction=result.wrist_correction.tolist(),
            initial_q=result.initial_q.tolist(),
            initialization_ms=result.initialization_ms,
            initialization_success=result.initialization_success,
            solve_ms=result.elapsed_ms,
            optimizer=type(retargeter.optimizer).__name__,
            qp_ms=result.get("qp_ms"),
            nqp=result.get("nqp"),
            qp_iterations=result.get("qp_iterations"),
            rejected_steps=result.get("rejected_steps"),
            pipeline_ms=result.pipeline_ms,
            nfev=int(result.nfev),
            status=int(result.status),
            success=bool(result.success),
            sample_dt_s=result.dt,
            raw_fit=result.raw_fit,
            output_fit=result.output_fit,
        )
    if output is not None:
        payload["hardware"] = {
            "channels": CHANNELS,
            "arrival_time_s": output.arrival_time,
            "command_position": None
            if result is None
            else output.write(result.q_output),
            "feedback": asdict(output.read()),
        }
    return payload


@contextmanager
def viewer_session(session, configs):
    if not session.viewer.enabled:
        yield None
        return
    from agx_retargeting.visualization.viewer import open_viewer

    with open_viewer(
        configs[session.viewer_hand], session.viewer.host, session.viewer.port
    ) as scene:
        yield scene


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[agx-webxr] %(message)s")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    from .server import WebXRInput

    if args.init_certs:
        initialize_certificates()
        return 0
    session = (
        SessionConfig() if args.session is None else SessionConfig.load(args.session)
    ).with_cli(args)
    params = yaml.safe_load(session.model.params.read_text())
    sides = session.sides
    configs = {side: HandConfig.load(session.hand_config, side) for side in sides}
    retargeters = {side: Retargeter(config, params) for side, config in configs.items()}
    source = WebXRInput(
        host=session.input.host,
        port=session.input.port,
        tracking_timeout=session.input.tracking_timeout,
        cert_path=session.input.cert,
        key_path=session.input.key,
    )
    stop = threading.Event()
    old_handlers = {
        sig: signal.signal(sig, lambda *_: stop.set())
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    output = None
    try:
        if session.output.recording:
            output = session.output.recording.open("w", encoding="utf-8", buffering=1)
        with (
            revo2_outputs(
                retargeters,
                session.channels,
                arrival_time=session.output.arrival_time,
                robot=session.output.robot,
                firmware=session.output.firmware,
            ) as hardware,
            source,
            viewer_session(session, configs) as viewer,
        ):
            log.info(
                "USB (adb reverse required): https://localhost:%d/", session.input.port
            )
            log.info("Wi-Fi: open the computer's LAN HTTPS address above in Quest")
            log.info(
                "Joint results: https://localhost:%d/api/retargeting/state",
                session.input.port,
            )
            last_report = 0.0
            while not stop.is_set():
                frames = source.read()
                if frames is not None:
                    packet = {}
                    for side, retargeter in retargeters.items():
                        packet[side] = solve_frame(
                            retargeter, frames[side], hardware.get(side)
                        )
                        source.set_result(side, packet[side])
                    if output:
                        output.write(json.dumps(packet, allow_nan=False) + "\n")
                state = source.state()
                if viewer is not None:
                    viewer.update(state)
                now = monotonic()
                if now - last_report >= 1:
                    summary = {
                        side: {
                            key: value.get(key)
                            for key in ("tracking", "solve_ms", "nfev", "error")
                        }
                        for side, value in state["hands"].items()
                        if side in sides
                    }
                    log.info(
                        "age=%s received=%d replaced=%d hands=%s",
                        state["input_age_s"],
                        state["received"],
                        state["replaced"],
                        summary,
                    )
                    last_report = now
    finally:
        if output:
            output.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 0
