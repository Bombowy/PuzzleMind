"""Capture BlueStacks and expose validated grid boundaries plus cell geometry."""

import sys
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Final

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_grid_detection_renderer import (
    OpenCvGridDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.vision.board_detector import BoardDetection, BoardDetectionError
from logicforge.vision.grid_detector import GridDetection, GridDetectionError
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG: Final = True
DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/grid_detection.png")
MIN_RECOMMENDED_SCREENSHOT_WIDTH: Final[int] = 440
MIN_RECOMMENDED_SCREENSHOT_HEIGHT: Final[int] = 470


def is_screenshot_below_recommended_size(screenshot: Screenshot) -> bool:
    """Return whether either dimension is below the operational recommendation.

    This check is intentionally advisory: callers still run both detectors, and
    these values never participate in board or grid acceptance decisions.
    """

    return (
        screenshot.width < MIN_RECOMMENDED_SCREENSHOT_WIDTH
        or screenshot.height < MIN_RECOMMENDED_SCREENSHOT_HEIGHT
    )


def build_small_screenshot_recommendation(screenshot: Screenshot) -> str | None:
    """Build a cautious resizing recommendation for a failed detection attempt."""

    if not is_screenshot_below_recommended_size(screenshot):
        return None
    return (
        "BlueStacks window may be too small for reliable detection.\n"
        "Enlarge the BlueStacks window and try again.\n"
        f"Captured resolution: {screenshot.width}x{screenshot.height}.\n"
        "Recommended minimum: "
        f"{MIN_RECOMMENDED_SCREENSHOT_WIDTH}x"
        f"{MIN_RECOMMENDED_SCREENSHOT_HEIGHT}."
    )


def print_small_screenshot_recommendation(screenshot: Screenshot) -> None:
    """Print the advisory only when the captured image is below recommendation."""

    recommendation = build_small_screenshot_recommendation(screenshot)
    if recommendation is not None:
        print(recommendation, file=sys.stderr)


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    board: BoardDetection,
    grid: GridDetection,
    board_elapsed_seconds: float,
    grid_elapsed_seconds: float,
) -> None:
    """Print complete operational evidence for one successful manual run."""

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(
        "Board: "
        f"x={board.x}, y={board.y}, width={board.width}, height={board.height}"
    )
    print(f"Board confidence: {board.confidence:.3f}")
    print(f"Detected rows: {grid.rows}")
    print(f"Detected columns: {grid.columns}")
    print(f"Horizontal boundary count: {len(grid.horizontal_lines)}")
    print(f"Vertical boundary count: {len(grid.vertical_lines)}")
    print(f"Horizontal lines: {grid.horizontal_lines}")
    print(f"Vertical lines: {grid.vertical_lines}")
    print(
        "Row heights: "
        f"{tuple(bottom - top for top, bottom in pairwise(grid.horizontal_lines))}"
    )
    print(
        "Column widths: "
        f"{tuple(right - left for left, right in pairwise(grid.vertical_lines))}"
    )
    print(f"Cell count: {len(grid.cells)}")
    print(f"Grid confidence: {grid.confidence:.3f}")
    print(f"Board detection time: {board_elapsed_seconds:.4f} seconds")
    print(f"Grid detection time: {grid_elapsed_seconds:.4f} seconds")
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Compose capture, board detection, grid extraction, and explicit rendering."""

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

    shared_settings = BoardDetectionSettings()
    board_detector = OpenCvBoardDetector(shared_settings)
    board_started_at = perf_counter()
    try:
        board = board_detector.detect(screenshot)
    except BoardDetectionError as error:
        print(f"Board detection failed: {error}", file=sys.stderr)
        print_small_screenshot_recommendation(screenshot)
        return 2
    board_elapsed_seconds = perf_counter() - board_started_at

    grid_detector = OpenCvGridDetector(shared_settings)
    grid_started_at = perf_counter()
    try:
        grid = grid_detector.detect(screenshot, board)
    except GridDetectionError as error:
        grid_elapsed_seconds = perf_counter() - grid_started_at
        renderer = OpenCvGridDetectionDebugRenderer()
        renderer.save_failure_debug_overlay(
            screenshot,
            error.diagnostics,
            DEBUG_OUTPUT_PATH,
            debug=DEBUG,
        )
        print(
            f"Grid detection failed after {grid_elapsed_seconds:.4f} seconds: {error}",
            file=sys.stderr,
        )
        print_small_screenshot_recommendation(screenshot)
        if DEBUG:
            print(
                f"Rejected diagnostics saved to: {DEBUG_OUTPUT_PATH.as_posix()}",
                file=sys.stderr,
            )
        return 3
    grid_elapsed_seconds = perf_counter() - grid_started_at

    renderer = OpenCvGridDetectionDebugRenderer()
    saved_path = renderer.save_debug_overlay(
        screenshot,
        board,
        grid,
        DEBUG_OUTPUT_PATH,
        debug=DEBUG,
    )
    if DEBUG and saved_path is None:
        print("Grid debug rendering produced no output path.", file=sys.stderr)
        return 4
    print_detection_information(
        window,
        screenshot,
        board,
        grid,
        board_elapsed_seconds,
        grid_elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
