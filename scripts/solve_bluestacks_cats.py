"""Thin CLI composition root for capturing and solving one visible Cats board."""

import argparse
import sys
from collections.abc import Sequence

from logicforge.application.cats import (
    CatClickExecutionError,
    CatClickPlanError,
    CatClickTarget,
    CatsBoardInput,
    CatsSolvedBoard,
    CatsSolveStatus,
    build_cat_click_plan,
    classify_result,
    collect_cat_coordinates,
    count_unresolved_cells,
    create_cat_click_target,
    execute_cat_click_plan,
    format_matrix,
    get_grid_cell,
    print_cat_click_plan,
    print_solve_information,
    solve_analyzed_cats_board,
)
from logicforge.application.cats.analysis import (
    analyze_captured_cats_board as analyze_cats_board_with_ports,
)
from logicforge.config.settings import BoardDetectionSettings, ColorDetectionSettings
from logicforge.core import BoardStateError
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.windows import (
    MouseAutomationError,
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
    Win32MouseController,
)
from logicforge.plugins.cats import CatsExactSearchError
from logicforge.plugins.cats.existing_cat import CatsExistingCatDetectionError
from logicforge.vision.board_detector import BoardDetectionError
from logicforge.vision.color_detector import ColorDetectionError
from logicforge.vision.grid_detector import GridDetectionError
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

__all__ = [
    "CatClickExecutionError",
    "CatClickPlanError",
    "CatClickTarget",
    "CatsBoardInput",
    "CatsSolveStatus",
    "CatsSolvedBoard",
    "analyze_captured_cats_board",
    "build_cat_click_plan",
    "classify_result",
    "collect_cat_coordinates",
    "count_unresolved_cells",
    "create_cat_click_target",
    "execute_cat_click_plan",
    "format_matrix",
    "get_grid_cell",
    "main",
    "parse_arguments",
    "print_cat_click_plan",
    "print_solve_information",
    "solve_analyzed_cats_board",
    "solve_captured_cats_board",
]


def _non_negative_int(value: str) -> int:
    """Parse an integer CLI value while rejecting negative click delays."""

    parsed_value = int(value)
    if parsed_value < 0:
        raise argparse.ArgumentTypeError(
            "click delay must be greater than or equal to 0"
        )
    return parsed_value


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse opt-in click execution and its deterministic inter-click delay."""

    parser = argparse.ArgumentParser(
        description="Capture, solve, and optionally execute Cats click targets."
    )
    parser.add_argument(
        "--execute-clicks",
        action="store_true",
        help="Execute every planned cat target as a left double-click.",
    )
    parser.add_argument(
        "--click-delay-ms",
        type=_non_negative_int,
        default=10,
        help=(
            "Delay between consecutive low-level clicks in milliseconds "
            "(default: 10)."
        ),
    )
    return parser.parse_args(arguments)


def analyze_captured_cats_board(screenshot: Screenshot) -> CatsBoardInput:
    """Compose the unchanged Cats analysis pipeline from concrete OpenCV adapters."""

    def fallback_geometry_detectors() -> tuple[
        OpenCvBoardDetector,
        OpenCvGridDetector,
    ]:
        board_settings = BoardDetectionSettings()
        return (
            OpenCvBoardDetector(board_settings),
            OpenCvGridDetector(board_settings),
        )

    return analyze_cats_board_with_ports(
        screenshot,
        tile_grid_detector=OpenCvCatsTileGridDetector(),
        fallback_geometry_detectors=fallback_geometry_detectors,
        color_detector=OpenCvColorDetector(ColorDetectionSettings()),
        existing_cat_detector=OpenCvCatsExistingCatDetector(),
    )


def solve_captured_cats_board(
    window: WindowInfo,
    screenshot: Screenshot,
) -> CatsSolvedBoard:
    """Compose the reusable capture solve with this CLI's concrete analyzer."""

    return solve_analyzed_cats_board(window, analyze_captured_cats_board(screenshot))


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one solve pipeline and optionally execute its complete click plan."""

    parsed_arguments = parse_arguments(() if arguments is None else arguments)

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
        board_input = analyze_captured_cats_board(screenshot)
    except BoardDetectionError as error:
        print(f"Board detection failed: {error}", file=sys.stderr)
        return 2
    except GridDetectionError as error:
        print(f"Grid detection failed: {error}", file=sys.stderr)
        return 3
    except ColorDetectionError as error:
        print(f"Color detection failed: {error}", file=sys.stderr)
        return 4
    except CatsExistingCatDetectionError as error:
        print(f"Existing cat detection failed: {error}", file=sys.stderr)
        return 4

    try:
        solved = solve_analyzed_cats_board(window, board_input)
    except (BoardStateError, CatsExactSearchError) as error:
        print(f"Cats deduction failed: {error}", file=sys.stderr)
        return 5
    except CatClickPlanError as error:
        print(f"Cats click-plan mapping failed: {error}", file=sys.stderr)
        return 6

    print_solve_information(
        window,
        screenshot,
        solved.board_input.detected_board,
        solved.board_input.grid,
        solved.board_input.color_result,
        solved.logical_board,
        solved.successful_applications,
        exact_search_result=solved.exact_search_result,
        status=solved.status,
    )
    print_cat_click_plan(solved.click_plan)

    if not parsed_arguments.execute_clicks:
        return 0
    if not solved.click_plan:
        print("Executed cat double-click targets: 0")
        return 0

    try:
        executed_targets = execute_cat_click_plan(
            solved.click_plan,
            Win32MouseController(),
            click_delay_seconds=parsed_arguments.click_delay_ms / 1000.0,
        )
    except (CatClickExecutionError, MouseAutomationError) as error:
        print(f"Cats click execution failed: {error}", file=sys.stderr)
        return 7

    print(f"Executed cat double-click targets: {executed_targets}")
    print(f"Low-level left clicks emitted: {executed_targets * 2}")
    print(f"Click delay: {parsed_arguments.click_delay_ms} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
