"""Capture a visible Cats board and run current deductions without interaction."""

import sys
from collections.abc import Sequence

from logicforge.config.settings import (
    BoardDetectionSettings,
    ColorDetectionSettings,
)
from logicforge.core import Board, BoardStateError
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.plugins.cats import apply_cats_rules_until_stalled
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


def format_matrix(values: Sequence[Sequence[str]]) -> str:
    """Format any rectangular logical matrix using its widest identifier."""

    identifier_width = max(
        (len(value) for row in values for value in row),
        default=0,
    )
    return "\n".join(
        " ".join(value.rjust(identifier_width) for value in row) for row in values
    )


def collect_cat_coordinates(board: Board) -> tuple[tuple[int, int], ...]:
    """Return every confirmed cat in deterministic zero-based row-major order."""

    return tuple(
        (row, column)
        for row, values in enumerate(board.cells)
        for column in range(len(values))
        if board.is_cat(row, column)
    )


def count_unresolved_cells(board: Board) -> int:
    """Count only current unresolved ``C<n>`` entries in the mutable Board."""

    return sum(
        board.is_unknown(row, column)
        for row, values in enumerate(board.cells)
        for column in range(len(values))
    )


def classify_result(board: Board) -> str:
    """Classify script output as COMPLETE or safely stalled with unknown cells."""

    return "COMPLETE" if count_unresolved_cells(board) == 0 else "STALLED"


def print_solve_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detected_board: BoardDetection,
    grid: GridDetection,
    color_result: ColorDetectionResult,
    logical_board: Board,
    successful_applications: int,
) -> None:
    """Print capture evidence, immutable input, and the mutated deduction result."""

    cats = collect_cat_coordinates(logical_board)
    unresolved_cells = count_unresolved_cells(logical_board)

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(
        "Board: "
        f"x={detected_board.x}, y={detected_board.y}, "
        f"width={detected_board.width}, height={detected_board.height}"
    )
    print(f"Grid: {grid.rows} rows x {grid.columns} columns")
    print(f"Colors detected: {color_result.color_count}")
    print(f"Mean color confidence: {color_result.mean_confidence:.3f}")
    print("\nInitial board:")
    print(format_matrix(color_result.color_matrix))
    print(f"\nSuccessful rule applications: {successful_applications}")
    print("\nFinal board:")
    print(format_matrix(logical_board.cells))
    print(f"\nCats found: {len(cats)}")
    for row, column in cats:
        print(f"K: row={row}, column={column}")
    print(f"\nUnresolved cells: {unresolved_cells}")
    print(f"Status: {classify_result(logical_board)}")


def main() -> int:
    """Run capture, vision, one mutable Board, and Cats rules to a fixed point."""

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
        detected_board = OpenCvBoardDetector(board_settings).detect(screenshot)
    except BoardDetectionError as error:
        print(f"Board detection failed: {error}", file=sys.stderr)
        return 2

    try:
        grid = OpenCvGridDetector(board_settings).detect(
            screenshot,
            detected_board,
        )
    except GridDetectionError as error:
        print(f"Grid detection failed: {error}", file=sys.stderr)
        return 3

    try:
        color_result = OpenCvColorDetector(ColorDetectionSettings()).detect(
            screenshot,
            grid,
        )
    except ColorDetectionError as error:
        print(f"Color detection failed: {error}", file=sys.stderr)
        return 4

    logical_board = Board(color_result)
    try:
        successful_applications = apply_cats_rules_until_stalled(logical_board)
    except BoardStateError as error:
        print(f"Cats deduction failed: {error}", file=sys.stderr)
        return 5

    print_solve_information(
        window,
        screenshot,
        detected_board,
        grid,
        color_result,
        logical_board,
        successful_applications,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
