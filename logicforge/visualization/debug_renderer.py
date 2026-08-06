"""Diagnostic overlay rendering boundary for vision development."""

from abc import ABC, abstractmethod

from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot
from logicforge.visualization.board_renderer import RenderArtifact


class DebugRenderer(ABC):
    """Render detector evidence without coupling production models to CV tooling."""

    @abstractmethod
    def render_detection(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection | None = None,
    ) -> RenderArtifact:
        """Overlay selected board and grid hypotheses for human inspection.

        TODO: Implement backend-specific drawing, confidence labels, and artifact
        redaction in v0.2 when screenshot fixtures and privacy policy exist.
        """

        raise NotImplementedError
