"""Ports and transfer objects for screenshot-to-board interpretation."""

from logicforge.vision.parser import PuzzleParser
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    CaptureResult,
    WindowBounds,
    WindowCaptureService,
    WindowInfo,
)

__all__ = [
    "CaptureResult",
    "PuzzleParser",
    "Screenshot",
    "WindowBounds",
    "WindowCaptureService",
    "WindowInfo",
]
