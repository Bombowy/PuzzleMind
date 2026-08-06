"""Capture BlueStacks and classify every detected grid cell by logical color."""

import sys
from pathlib import Path
from time import perf_counter
from typing import Final

from logicforge.config.settings import (
    BoardDetectionSettings,
    ColorDetectionSettings,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_color_detection_renderer import (
    ColorDebugRenderError,
    OpenCvColorDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.vision.board_detector import BoardDetection, BoardDetectionError
from logicforge.vision.color_detector import (
    ColorDetectionError,
    ColorDetectionResult,
)
from logicforge.vision.grid_detector import GridDetection, GridDetectionError
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG: Final = True
DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/color_detection.png")


def format_color_matrix(result: ColorDetectionResult) -> str:
    """Format the immutable matrix as aligned terminal rows for human inspection."""

    identifier_width = max(
        len(color_id) for row in result.color_matrix for color_id in row
    )
    return "\n".join(
        " ".join(color_id.rjust(identifier_width) for color_id in row)
        for row in result.color_matrix
    )


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    board: BoardDetection,
    grid: GridDetection,
    result: ColorDetectionResult,
    color_elapsed_seconds: float,
) -> None:
    """Print complete puzzle-neutral evidence for one successful manual run."""

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(
        "Board: "
        f"x={board.x}, y={board.y}, width={board.width}, height={board.height}"
    )
    print(f"Grid: {grid.rows} rows x {grid.columns} columns")
    print(f"Colors detected: {result.color_count}")
    print("Color matrix:")
    print(format_color_matrix(result))
    print(f"Mean color confidence: {result.mean_confidence:.3f}")
    print(f"Color detection time: {color_elapsed_seconds:.4f} seconds")
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Compose capture, board, grid, color classification, and debug rendering."""

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

    board_settings = BoardDetectionSettings()
    try:
        board = OpenCvBoardDetector(board_settings).detect(screenshot)
    except BoardDetectionError as error:
        print(f"Board detection failed: {error}", file=sys.stderr)
        return 2

    try:
        grid = OpenCvGridDetector(board_settings).detect(screenshot, board)
    except GridDetectionError as error:
        print(f"Grid detection failed: {error}", file=sys.stderr)
        return 3

    color_started_at = perf_counter()
    try:
        result = OpenCvColorDetector(ColorDetectionSettings()).detect(
            screenshot,
            grid,
        )
    except ColorDetectionError as error:
        print(f"Color detection failed: {error}", file=sys.stderr)
        return 4
    color_elapsed_seconds = perf_counter() - color_started_at

    renderer = OpenCvColorDetectionDebugRenderer()
    try:
        saved_path = renderer.save_debug_overlay(
            screenshot,
            board,
            grid,
            result,
            DEBUG_OUTPUT_PATH,
            debug=DEBUG,
        )
    except ColorDebugRenderError as error:
        print(f"Color debug rendering failed: {error}", file=sys.stderr)
        return 5
    if DEBUG and saved_path is None:
        print("Color debug rendering produced no output path.", file=sys.stderr)
        return 5

    print_detection_information(
        window,
        screenshot,
        board,
        grid,
        result,
        color_elapsed_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
