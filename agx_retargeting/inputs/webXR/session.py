from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from agx_retargeting.config import CONFIG_DIR


@dataclass(frozen=True, slots=True)
class WebXRSettings:
    host: str = "0.0.0.0"
    port: int = 9443
    tracking_timeout: float = 0.25
    cert: Path | None = None
    key: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelSettings:
    name: str = "revo2"
    hand: str = "both"
    config: Path | None = None
    params: Path = CONFIG_DIR / "optimizer_params.yaml"


@dataclass(frozen=True, slots=True)
class OutputSettings:
    recording: Path | None = None
    left_can: str | None = None
    right_can: str | None = None
    arrival_time: float = 0.04
    robot: str = "piper"
    firmware: str = "default"


@dataclass(frozen=True, slots=True)
class ViewerSettings:
    enabled: bool = False
    hand: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080


@dataclass(frozen=True, slots=True)
class SessionConfig:
    input: WebXRSettings = field(default_factory=WebXRSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    viewer: ViewerSettings = field(default_factory=ViewerSettings)

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve(strict=True)
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)

        def section(values, settings, paths):
            values = dict(values)
            for key in paths:
                if values.get(key) is not None:
                    value = Path(values[key]).expanduser()
                    values[key] = (
                        value if value.is_absolute() else path.parent / value
                    ).resolve()
            return settings(**values)

        settings = {
            "input": (WebXRSettings, ("cert", "key")),
            "model": (ModelSettings, ("config", "params")),
            "output": (OutputSettings, ("recording",)),
            "viewer": (ViewerSettings, ()),
        }
        return cls(
            **{
                name: section(values, *settings[name])
                for name, values in document.items()
            }
        )

    def with_cli(self, args):
        values = vars(args)

        def override(current, mapping):
            return replace(
                current,
                **{
                    field_name: values[argument]
                    for argument, field_name in mapping.items()
                    if argument in values
                },
            )

        return replace(
            self,
            input=override(
                self.input,
                {
                    "host": "host",
                    "port": "port",
                    "tracking_timeout": "tracking_timeout",
                    "cert": "cert",
                    "key": "key",
                },
            ),
            model=override(
                self.model,
                {
                    "model": "name",
                    "hand": "hand",
                    "config": "config",
                    "params": "params",
                },
            ),
            output=override(
                self.output,
                {
                    "output": "recording",
                    "revo2_left_can": "left_can",
                    "revo2_right_can": "right_can",
                    "revo2_arrival_time": "arrival_time",
                    "revo2_robot": "robot",
                    "revo2_firmware": "firmware",
                },
            ),
            viewer=override(
                self.viewer,
                {
                    "viewer": "enabled",
                    "viewer_hand": "hand",
                    "viewer_host": "host",
                    "viewer_port": "port",
                },
            ),
        )

    @property
    def sides(self):
        return ("left", "right") if self.model.hand == "both" else (self.model.hand,)

    @property
    def hand_config(self):
        return self.model.config or CONFIG_DIR / f"{self.model.name}_params.yaml"

    @property
    def channels(self):
        return {
            side: channel
            for side, channel in (
                ("left", self.output.left_can),
                ("right", self.output.right_can),
            )
            if channel is not None
        }

    @property
    def viewer_hand(self):
        return self.viewer.hand or self.sides[0]
