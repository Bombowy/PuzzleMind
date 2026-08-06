"""Boundary for puzzle-neutral color observations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.core.coordinates import Coordinates
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class ColorObservation:
    """Record a normalized color label observed at one logical coordinate.

    TODO: Replace the textual label with calibrated color-space data and preserve
    raw samples when the Cats parser defines its classification requirements.
    """

    coordinates: Coordinates
    label: str
    confidence: float


class ColorDetector(ABC):
    """Define the port for extracting normalized color observations from cells."""

    @abstractmethod
    def detect(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection,
    ) -> tuple[ColorObservation, ...]:
        """Return color observations aligned with detected logical coordinates.

        TODO: Implement sampling, illumination normalization, clustering, and
        confidence calibration after representative screenshot fixtures exist.
        """

        raise NotImplementedError
