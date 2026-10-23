import hashlib
import threading
from pathlib import Path
from time import monotonic

from aiohttp import web
from xr_media import MediaServerConfig, XrMediaServer
from xr_media.paths import static_dir, tls_dir

from agx_retargeting.types import HandInput

from .adapter import WEBXR_JOINT_NAMES, WebXRAdapter
from .client import patched_client


class WebXRServer(XrMediaServer):
    def __init__(self, config, owner):
        self.owner = owner
        self.client = patched_client((static_dir() / "xr.js").read_text())
        self.asset_version = hashlib.sha256(self.client.encode()).hexdigest()[:12]
        super().__init__(config, on_xr=owner._on_xr)

    def configure_app(self, app):
        app.router.add_get("/xr/xr.js", self._client)
        app.router.add_get("/api/retargeting/state", self._retargeting_state)
        super().configure_app(app)

    async def _client(self, request):
        return web.Response(
            text=self.client,
            content_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )

    async def _index(self, request):
        page = (static_dir() / "index.html").read_text()
        page = page.replace("<title>XR Media</title>", "<title>AGX WebXR</title>")
        page = page.replace(
            "</head>",
            f'<link rel="stylesheet" href="/xr/xr.css?v={self.asset_version}"></head>',
            1,
        ).replace(
            "</body>",
            '<canvas id="xr-stage" aria-hidden="true"></canvas>'
            f'<script type="module" src="/xr/xr.js?v={self.asset_version}"></script></body>',
            1,
        )
        return web.Response(
            text=page, content_type="text/html", headers={"Cache-Control": "no-store"}
        )

    async def _xr_signal(self, request):
        if not self._xr_peers:
            self.owner._new_session()
        return await super()._xr_signal(request)

    async def _retargeting_state(self, request):
        return web.json_response(
            self.owner.state(), headers={"Cache-Control": "no-store"}
        )


class WebXRInput:
    """Receive both hands; read() consumes only the latest unprocessed sample.

    Browser timestamps determine dt. Host monotonic receipt times determine
    staleness; these two clocks are never subtracted from each other.
    """

    def __init__(
        self,
        *,
        host="0.0.0.0",
        port=9443,
        tracking_timeout=0.25,
        cert_path: Path | None = None,
        key_path: Path | None = None,
    ):
        cert = Path(cert_path) if cert_path else tls_dir() / "cert.pem"
        key = Path(key_path) if key_path else tls_dir() / "key.pem"
        self.tracking_timeout = tracking_timeout
        self._condition = threading.Condition()
        self._pending = None
        self._last_received = None
        self._last_timestamp = None
        self._reset_pending = False
        self._lost = False
        self._closed = False
        self._received = 0
        self._replaced = 0
        self._hands = {side: {"tracking": False} for side in ("left", "right")}
        self._input = {}
        self.server = WebXRServer(
            MediaServerConfig(
                host=host, port=port, max_clients=2, cert_path=cert, key_path=key
            ),
            self,
        )

    def start(self):
        self.server.start_background()
        return self

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.server.stop_background()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    def _new_session(self):
        with self._condition:
            self._pending = None
            self._last_received = None
            self._last_timestamp = None
            self._input = {}
            self._reset_pending = True
            self._condition.notify_all()

    def _on_xr(self, sample):
        with self._condition:
            if self._closed:
                return
            self._received += 1
            self._replaced += self._pending is not None
            received = monotonic()
            self._pending = (sample, received)
            self._last_received = received
            self._input = {}
            for side in ("left", "right"):
                hand = getattr(sample, side)
                joints = {}
                for name, pose in hand.joints.items():
                    values = [
                        pose.px,
                        pose.py,
                        pose.pz,
                        pose.qx,
                        pose.qy,
                        pose.qz,
                        pose.qw,
                    ]
                    joints[name] = values
                self._input[side] = {
                    "tracked": hand.tracked,
                    "source_type": hand.source_type,
                    "timestamp_s": sample.t_ms / 1000.0,
                    "frame_id": "webxr_z_up",
                    "joints": joints,
                    "joint_count": len(hand.joints),
                    "missing_joints": [
                        name for name in WEBXR_JOINT_NAMES if name not in hand.joints
                    ],
                }
            self._condition.notify_all()

    @staticmethod
    def _untracked():
        return {
            side: HandInput(
                None, side, monotonic(), tracked=False, frame_id="webxr_z_up"
            )
            for side in ("left", "right")
        }

    def read(self, timeout=0.1):
        """Return {left, right} frames, None while waiting, or loss frames once.

        Missing joints are loss frames. A new XR session or a backwards source
        clock first emits loss frames so Retargeter clears its temporal state.
        """
        deadline = monotonic() + timeout
        with self._condition:
            while not self._closed:
                now = monotonic()
                if self._reset_pending:
                    self._reset_pending = False
                    self._lost = False
                    return self._untracked()
                if self._pending is not None:
                    sample, received = self._pending
                    if now - received < self.tracking_timeout:
                        timestamp = sample.t_ms / 1000.0
                        if (
                            self._last_timestamp is not None
                            and timestamp < self._last_timestamp
                        ):
                            self._last_timestamp = None
                            return self._untracked()
                        self._pending = None
                        if timestamp == self._last_timestamp:
                            continue
                        self._last_timestamp = timestamp
                        self._lost = False
                        break
                    self._pending = None
                if (
                    self._last_received is not None
                    and now - self._last_received >= self.tracking_timeout
                    and not self._lost
                ):
                    self._lost = True
                    self._last_timestamp = None
                    return self._untracked()
                remaining = deadline - now
                if remaining <= 0:
                    return None
                if self._last_received is not None and not self._lost:
                    remaining = min(
                        remaining,
                        max(0, self._last_received + self.tracking_timeout - now),
                    )
                self._condition.wait(remaining)
            else:
                return None
        return {
            side: WebXRAdapter.adapt(
                getattr(sample, side), side=side, timestamp=timestamp
            )
            for side in ("left", "right")
        }

    def set_result(self, side, result):
        with self._condition:
            self._hands[side] = result

    def state(self):
        with self._condition:
            age = (
                None
                if self._last_received is None
                else monotonic() - self._last_received
            )
            hands = {side: dict(value) for side, value in self._hands.items()}
            inputs = {side: dict(value) for side, value in self._input.items()}
            if age is None or age >= self.tracking_timeout:
                for value in hands.values():
                    value.update(tracking=False, q_output=None)
                for value in inputs.values():
                    value.update(tracked=False, joints={})
            return {
                "input_age_s": age,
                "received": self._received,
                "replaced": self._replaced,
                "hands": hands,
                "input": inputs,
            }
