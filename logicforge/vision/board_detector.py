"""Boundary for locating a puzzle board inside a screenshot."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class BoardDetection:
    """Describe a rectangular board hypothesis in screenshot pixel coordinates.

    TODO: Add rotation, perspective transform, and diagnostic evidence in v0.2
    once detector backends define a common confidence-calibration strategy.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float


class BoardDetector(ABC):
    """Define the application-facing port for board localization.

    Implementations may use classical CV, learned models, or fixture data, but
    callers depend only on this contract and its deterministic result record.
    """

    @abstractmethod
    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Locate the most likely puzzle-board boundary in ``screenshot``.

        TODO: Implement backend-specific localization, confidence calibration,
        and explicit errors for missing or ambiguous boards in v0.2.
        """

        raise NotImplementedError
