"""Immutable in-memory screenshot model used by future vision components."""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True, eq=False)
class Screenshot:
    """Own one immutable, contiguous, three-channel BGR image captured in memory.

    The model copies caller-owned pixels and marks its internal array read-only so
    future detectors can safely share a snapshot without observing external
    mutations. File paths and encoded PNG data deliberately do not belong here.
    """

    image: NDArray[np.uint8]
    width: int
    height: int
    timestamp: datetime

    def __post_init__(self) -> None:
        """Validate BGR shape, dimensions, timestamp, and freeze an owned copy."""

        if not isinstance(self.image, np.ndarray):
            raise TypeError("Screenshot image must be a numpy.ndarray.")
        if self.image.dtype != np.uint8:
            raise TypeError("Screenshot image must use the uint8 data type.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Screenshot width and height must both be positive.")
        if self.image.shape != (self.height, self.width, 3):
            raise ValueError(
                "Screenshot image shape must be exactly (height, width, 3) for BGR."
            )
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Screenshot timestamp must include timezone information.")

        immutable_buffer = np.ascontiguousarray(self.image).tobytes()
        immutable_image = np.frombuffer(immutable_buffer, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )
        object.__setattr__(self, "image", immutable_image)
