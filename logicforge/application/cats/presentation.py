"""Stable terminal presentation shared by Cats CLI composition roots."""

from collections.abc import Sequence

from logicforge.application.cats.click_plan import collect_cat_coordinates
from logicforge.application.cats.models import CatClickTarget, CatsSolveStatus
from logicforge.application.cats.solving import classify_result, count_unresolved_cells
from logicforge.core import Board
from logicforge.plugins.cats.exact_search import CatsExactSearchResult
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.color_detector import ColorDetectionResult
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowInfo


def format_matrix(values: Sequence[Sequence[str]]) -> str:
    """Format any rectangular logical matrix using its widest identifier."""

    identifier_width = max(
        (len(value) for row in values for value in row),
        default=0,
    )
    return "\n".join(
        " ".join(value.rjust(identifier_width) for value in row) for row in values
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
    status: CatsSolveStatus | None = None,
) -> None:
    """Print capture evidence, immutable input, and the logical solve result."""

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
