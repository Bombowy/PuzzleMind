"""Capture and solve a visible Cats board, with optional explicit click execution."""

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.config.settings import (
    BoardDetectionSettings,
    ColorDetectionSettings,
)
from logicforge.core import Board, BoardStateError
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
from logicforge.plugins.cats import (
    CatsExactSearchError,
    CatsExactSearchResult,
    CatsExactSearchStatus,
    apply_cats_rules_until_stalled,
    apply_unique_cats_exact_solution,
    place_cat,
    solve_cats_exact,
)
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
    CatsExistingCatDiagnostics,
)
from logicforge.plugins.cats.tile_grid import CatsTileGridDetectionError
from logicforge.vision.board_detector import BoardDetection, BoardDetectionError
from logicforge.vision.color_detector import (
    ColorDetectionError,
    ColorDetectionResult,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionError,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)


class CatClickPlanError(RuntimeError):
    """Report inconsistent logical or detected geometry during dry-run mapping."""


class CatClickExecutionError(RuntimeError):
    """Report invalid or failed orchestration of the explicit click plan."""


@dataclass(frozen=True, slots=True)
class CatClickTarget:
    """Describe one future cat click without emitting any pointer input.

    Screenshot coordinates come directly from the detected cell center. Desktop
    coordinates add the captured window's virtual-desktop origin and may remain
    negative when BlueStacks is positioned left of or above the primary monitor.
    """

    row: int
    column: int
    screenshot_x: int
    screenshot_y: int
    desktop_x: int
    desktop_y: int


@dataclass(frozen=True, slots=True)
class CatsBoardInput:
    """Retain immutable vision output required to solve one captured Cats board."""

    detected_board: BoardDetection
    grid: GridDetection
    color_result: ColorDetectionResult
    existing_cat_detection: CatsExistingCatDetection = field(
        default_factory=lambda: CatsExistingCatDetection(
            cats=(),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        )
    )


@dataclass(frozen=True, slots=True)
class CatsSolvedBoard:
    """Retain one logical solve result and its deterministic click plan."""

    board_input: CatsBoardInput
    logical_board: Board
    successful_applications: int
    click_plan: tuple[CatClickTarget, ...]
    status: str
    exact_search_result: CatsExactSearchResult | None = None


type SleepFunction = Callable[[float], None]


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


def get_grid_cell(
    grid: GridDetection,
    row: int,
    column: int,
) -> CellBounds:
    """Return one row-major cell or raise a typed coordinate-consistency error."""

    requested_coordinates = f"row={row}, column={column}"
    if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) is outside the "
            f"detected {grid.rows}x{grid.columns} grid."
        )

    index = row * grid.columns + column
    try:
        cell = grid.cells[index]
    except IndexError as error:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) has no row-major "
            f"entry at index {index}."
        ) from error

    if cell.row != row or cell.column != column:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) resolved to "
            f"cell row={cell.row}, column={cell.column} at row-major index {index}."
        )
    return cell


def create_cat_click_target(
    window: WindowInfo,
    grid: GridDetection,
    row: int,
    column: int,
) -> CatClickTarget:
    """Map one logical cat coordinate to screenshot and desktop centers."""

    cell = get_grid_cell(grid, row, column)
    return CatClickTarget(
        row=row,
        column=column,
        screenshot_x=cell.center_x,
        screenshot_y=cell.center_y,
        desktop_x=window.bounds.x + cell.center_x,
        desktop_y=window.bounds.y + cell.center_y,
    )


def build_cat_click_plan(
    board: Board,
    grid: GridDetection,
    window: WindowInfo,
    *,
    existing_cat_coordinates: Sequence[tuple[int, int]] = (),
) -> tuple[CatClickTarget, ...]:
    """Map only new K cells after validating dimensions and exclusions first."""

    if len(board.cells) != grid.rows:
        raise CatClickPlanError(
            f"Board has {len(board.cells)} rows but detected grid has "
            f"{grid.rows} rows."
        )
    for row, values in enumerate(board.cells):
        if len(values) != grid.columns:
            raise CatClickPlanError(
                f"Board row {row} has {len(values)} columns but detected grid "
                f"has {grid.columns} columns."
            )

    excluded = tuple(existing_cat_coordinates)
    if len(excluded) != len(set(excluded)):
        raise CatClickPlanError("Existing cat coordinates contain duplicates.")
    for row, column in excluded:
        if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
            raise CatClickPlanError(
                f"Existing cat coordinate ({row}, {column}) is outside the "
                f"detected {grid.rows}x{grid.columns} grid."
            )
        if not board.is_cat(row, column):
            raise CatClickPlanError(
                f"Existing cat coordinate ({row}, {column}) is not K on final Board."
            )
    excluded_set = set(excluded)
    new_cats = tuple(
        coordinate
        for coordinate in collect_cat_coordinates(board)
        if coordinate not in excluded_set
    )
    return tuple(
        create_cat_click_target(window, grid, row, column) for row, column in new_cats
    )


def analyze_captured_cats_board(screenshot: Screenshot) -> CatsBoardInput:
    """Fit Cats tiles first, then classify immutable colors without a Board."""

    try:
        tile_grid = OpenCvCatsTileGridDetector().detect(screenshot)
    except CatsTileGridDetectionError:
        board_settings = BoardDetectionSettings()
        detected_board = OpenCvBoardDetector(board_settings).detect(screenshot)
        grid = OpenCvGridDetector(board_settings).detect(screenshot, detected_board)
    else:
        detected_board = tile_grid.board
        grid = tile_grid.grid
    color_result = OpenCvColorDetector(ColorDetectionSettings()).detect(
        screenshot,
        grid,
    )
    existing_cat_detection = OpenCvCatsExistingCatDetector().detect(
        screenshot,
        grid,
        color_result,
    )
    return CatsBoardInput(
        detected_board=detected_board,
        grid=grid,
        color_result=color_result,
        existing_cat_detection=existing_cat_detection,
    )


def solve_analyzed_cats_board(
    window: WindowInfo,
    board_input: CatsBoardInput,
    *,
    maximum_search_nodes: int = 250_000,
) -> CatsSolvedBoard:
    """Run preferred rules, then a bounded uniqueness proof only if stalled."""

    logical_board = Board(board_input.color_result)
    existing_coordinates = tuple(
        (cat.row, cat.column) for cat in board_input.existing_cat_detection.cats
    )
    for row, column in existing_coordinates:
        place_cat(logical_board, row, column)
    successful_applications = apply_cats_rules_until_stalled(logical_board)
    status = classify_result(logical_board)
    exact_search_result: CatsExactSearchResult | None = None
    if status == "STALLED":
        print("[solver] rules stalled; exact search started")
        exact_search_result = solve_cats_exact(
            logical_board,
            board_input.color_result.color_matrix,
            maximum_search_nodes=maximum_search_nodes,
        )
        print(
            "[solver] exact search "
            f"{exact_search_result.status.value} "
            f"nodes={exact_search_result.search_nodes} "
            f"propagation={exact_search_result.propagation_steps}"
        )
        if exact_search_result.status is CatsExactSearchStatus.UNIQUE:
            apply_unique_cats_exact_solution(
                logical_board,
                exact_search_result,
                original_color_matrix=board_input.color_result.color_matrix,
            )
            status = classify_result(logical_board)
            if status != "COMPLETE":
                raise CatsExactSearchError(
                    "Applying a UNIQUE exact solution did not complete Board."
                )
        elif exact_search_result.status is CatsExactSearchStatus.UNSAT:
            status = "UNSAT"
        elif exact_search_result.status is CatsExactSearchStatus.AMBIGUOUS:
            status = "AMBIGUOUS"
        else:
            status = "SEARCH_LIMIT"

    click_plan: tuple[CatClickTarget, ...] = ()
    if status == "COMPLETE":
        if existing_coordinates:
            click_plan = build_cat_click_plan(
                logical_board,
                board_input.grid,
                window,
                existing_cat_coordinates=existing_coordinates,
            )
        else:
            click_plan = build_cat_click_plan(
                logical_board,
                board_input.grid,
                window,
            )
    return CatsSolvedBoard(
        board_input=board_input,
        logical_board=logical_board,
        successful_applications=successful_applications,
        click_plan=click_plan,
        status=status,
        exact_search_result=exact_search_result,
    )


def solve_captured_cats_board(
    window: WindowInfo,
    screenshot: Screenshot,
) -> CatsSolvedBoard:
    """Compose immutable vision analysis with exactly one logical solve."""

    return solve_analyzed_cats_board(
        window,
        analyze_captured_cats_board(screenshot),
    )


def print_cat_click_plan(targets: tuple[CatClickTarget, ...]) -> None:
    """Print a dry-run plan without invoking any mouse or automation API."""

    print(f"Planned cat click targets: {len(targets)}")
    for target in targets:
        print(
            "CLICK: "
            f"row={target.row}, column={target.column}, "
            f"screenshot=({target.screenshot_x}, {target.screenshot_y}), "
            f"desktop=({target.desktop_x}, {target.desktop_y})"
        )


def execute_cat_click_plan(
    targets: tuple[CatClickTarget, ...],
    mouse_controller: MouseController,
    *,
    click_delay_seconds: float = 0.01,
    sleep_function: SleepFunction = time.sleep,
) -> int:
    """Double-click every target in order with one delay between all clicks."""

    if click_delay_seconds < 0:
        raise CatClickExecutionError(
            "Click delay must be greater than or equal to zero seconds."
        )

    for target_index, target in enumerate(targets):
        point = ScreenPoint(x=target.desktop_x, y=target.desktop_y)
        mouse_controller.click(point, MouseButton.LEFT)
        sleep_function(click_delay_seconds)
        mouse_controller.click(point, MouseButton.LEFT)
        if target_index < len(targets) - 1:
            sleep_function(click_delay_seconds)

    return len(targets)


def print_solve_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detected_board: BoardDetection,
    grid: GridDetection,
    color_result: ColorDetectionResult,
    logical_board: Board,
    successful_applications: int,
    *,
    exact_search_result: CatsExactSearchResult | None = None,
    status: str | None = None,
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
    if exact_search_result is None:
        print("Exact search: not needed")
    else:
        print(f"Exact search: {exact_search_result.status.value}")
        print(f"Search nodes: {exact_search_result.search_nodes}")
        print(f"Propagation steps: {exact_search_result.propagation_steps}")
    print("\nFinal board:")
    print(format_matrix(logical_board.cells))
    print(f"\nCats found: {len(cats)}")
    for row, column in cats:
        print(f"K: row={row}, column={column}")
    print(f"\nUnresolved cells: {unresolved_cells}")
    print(f"Status: {status or classify_result(logical_board)}")


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
