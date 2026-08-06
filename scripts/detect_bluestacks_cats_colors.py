"""Capture one Cats frame, fit tiles, and run the unchanged LAB color detector."""

import sys
from pathlib import Path
from typing import Final

from logicforge.config.settings import ColorDetectionSettings
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detection_renderer import (
    ColorDebugRenderError,
    OpenCvColorDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetection,
    CatsTileGridDetectionError,
)
from logicforge.vision.color_detector import ColorDetectionError, ColorDetectionResult
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/cats_color_detection.png")


def format_color_matrix(result: ColorDetectionResult) -> str:
    """Format immutable logical color IDs as aligned row-major text."""

    width = max(len(color_id) for row in result.color_matrix for color_id in row)
    return "\n".join(
        " ".join(color_id.rjust(width) for color_id in row)
        for row in result.color_matrix
    )


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    tile_grid: CatsTileGridDetection,
    colors: ColorDetectionResult,
) -> None:
    """Print tile geometry and the existing color classifier's complete result."""

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"Grid: {tile_grid.grid.rows}x{tile_grid.grid.columns}")
    print(f"Cells: {len(tile_grid.grid.cells)}")
    print(f"Detected color classes: {colors.color_count}")
    print("Color matrix:")
    print(format_color_matrix(colors))
    print(f"Color confidence: {colors.mean_confidence:.3f}")
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Capture once, fit tile geometry, classify colors, and never click."""

    capture_service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
    )
    try:
        window = capture_service.locate_window()
        screenshot = capture_service.capture_window(window, debug=False)
    except WindowCaptureError as error:
        print(f"BlueStacks capture failed: {error}", file=sys.stderr)
        return 1

    try:
        tile_grid = OpenCvCatsTileGridDetector().detect(screenshot)
    except CatsTileGridDetectionError as error:
        print(f"Cats tile-grid detection failed: {error}", file=sys.stderr)
        return 2
    try:
        color_settings = ColorDetectionSettings()
        colors = OpenCvColorDetector(color_settings).detect(
            screenshot,
            tile_grid.grid,
        )
    except ColorDetectionError as error:
        print(f"Cats color detection failed: {error}", file=sys.stderr)
        return 3

    try:
        saved_path = OpenCvColorDetectionDebugRenderer(
            color_settings
        ).save_debug_overlay(
            screenshot,
            tile_grid.board,
            tile_grid.grid,
            colors,
            DEBUG_OUTPUT_PATH,
            debug=True,
            draw_sample_regions=True,
        )
    except ColorDebugRenderError as error:
        print(f"Cats color debug rendering failed: {error}", file=sys.stderr)
        return 4
    if saved_path is None:
        print("Cats color debug rendering produced no output.", file=sys.stderr)
        return 4

    print_detection_information(window, screenshot, tile_grid, colors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
