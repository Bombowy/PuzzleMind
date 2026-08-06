"""Ports and transfer objects for screenshot-to-board interpretation."""

from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionAnalysis,
    BoardDetectionDiagnostics,
    BoardDetectionError,
    BoardDetector,
)
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionError,
    ColorDetectionResult,
    ColorDetector,
    ColorObservation,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionDiagnostics,
    GridDetectionError,
    GridDetector,
)
from logicforge.vision.parser import PuzzleParser
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowCaptureService,
    WindowInfo,
)

__all__ = [
    "BoardDetection",
    "BoardDetectionAnalysis",
    "BoardDetectionDiagnostics",
    "BoardDetectionError",
    "BoardDetector",
    "CellBounds",
    "ColorDetectionDiagnostics",
    "ColorDetectionError",
    "ColorDetectionResult",
    "ColorDetector",
    "ColorObservation",
    "GridDetection",
    "GridDetectionDiagnostics",
    "GridDetectionError",
    "GridDetector",
    "PuzzleParser",
    "Screenshot",
    "WindowBounds",
    "WindowCaptureService",
    "WindowInfo",
]
