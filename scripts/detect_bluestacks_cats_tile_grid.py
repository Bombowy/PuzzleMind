"""Capture one BlueStacks frame and diagnose Cats tile-grid-first geometry."""

import sys
from pathlib import Path
from typing import Final

from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_renderer import (
    CatsTileGridDebugRenderError,
    OpenCvCatsTileGridDebugRenderer,
)
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetection,
    CatsTileGridDetectionError,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/cats_tile_grid_detection.png")


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detection: CatsTileGridDetection,
) -> None:
    """Print complete primitive lattice evidence for one captured frame."""

    diagnostics = detection.diagnostics
    board = detection.board
    grid = detection.grid
    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"Raw components: {diagnostics.component_count}")
    print(f"Tile candidates: {diagnostics.candidate_tile_count}")
    print(f"Accepted tiles: {diagnostics.selected_tile_count}")
    print(f"Rows: {grid.rows}")
    print(f"Columns: {grid.columns}")
    print(f"Median tile width: {diagnostics.median_tile_width:.3f}")
    print(f"Median tile height: {diagnostics.median_tile_height:.3f}")
    print(f"Horizontal pitch: {diagnostics.horizontal_pitch:.3f}")
    print(f"Vertical pitch: {diagnostics.vertical_pitch:.3f}")
    print(f"Horizontal pitch CV: {diagnostics.horizontal_pitch_cv:.3f}")
    print(f"Vertical pitch CV: {diagnostics.vertical_pitch_cv:.3f}")
    print(f"Occupancy: {diagnostics.occupancy_ratio:.3f}")
    print(
        f"Board: x={board.x}, y={board.y}, width={board.width}, "
        f"height={board.height}"
    )
    print(f"Cell count: {len(grid.cells)}")
    print(f"Grid confidence: {grid.confidence:.3f}")
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Capture once, detect once, render once, and never automate the mouse."""

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
        detection = OpenCvCatsTileGridDetector().detect(screenshot)
    except CatsTileGridDetectionError as error:
        print(f"Cats tile-grid detection failed: {error}", file=sys.stderr)
        for reason in error.diagnostics.rejection_reasons:
            print(f"- {reason}", file=sys.stderr)
        return 2

    renderer = OpenCvCatsTileGridDebugRenderer()
    try:
        saved_path = renderer.save_debug_overlay(
            screenshot,
            detection,
            DEBUG_OUTPUT_PATH,
            debug=True,
        )
    except CatsTileGridDebugRenderError as error:
        print(f"Cats tile-grid debug rendering failed: {error}", file=sys.stderr)
        return 3
    if saved_path is None:
        print("Cats tile-grid debug rendering produced no output.", file=sys.stderr)
        return 3

    print_detection_information(window, screenshot, detection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
