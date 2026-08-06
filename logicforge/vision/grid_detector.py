"""Boundary for recovering logical grid geometry from a detected board."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class GridDetection:
    """Describe detected grid-line positions and the confidence of that geometry.

    TODO: Add irregular-grid geometry and cell polygons when supported puzzle
    plugins prove that axis-aligned line coordinates are insufficient.
    """

    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]
    confidence: float


class GridDetector(ABC):
    """Define the port that converts a board boundary into grid geometry."""

    @abstractmethod
    def detect(self, screenshot: Screenshot, board: BoardDetection) -> GridDetection:
        """Recover cell boundaries within a previously detected board.

        TODO: Implement line extraction, de-duplication, geometry validation, and
        uncertainty reporting as part of the v0.2 screenshot parser milestone.
        """

        raise NotImplementedError
