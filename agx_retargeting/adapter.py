from typing import Protocol, TypeVar

import numpy as np

from .types import HandInput

Source_contra = TypeVar("Source_contra", contravariant=True)
Command_co = TypeVar("Command_co", covariant=True)


class InputAdapter(Protocol[Source_contra]):
    def adapt(self, source: Source_contra, /, *, side: str, timestamp: float) -> HandInput: ...


class OutputAdapter(Protocol[Command_co]):
    def adapt(self, q: np.ndarray) -> Command_co: ...
