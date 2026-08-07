"""Rules-first Cats solving with the unchanged deterministic exact fallback."""

from collections.abc import Callable

from logicforge.application.cats.click_plan import build_cat_click_plan
from logicforge.application.cats.models import (
    CatClickTarget,
    CatsBoardInput,
    CatsSolvedBoard,
    CatsSolveStatus,
)
from logicforge.core import Board
from logicforge.plugins.cats import (
    CatsExactSearchError,
    CatsExactSearchResult,
    CatsExactSearchStatus,
    apply_cats_rules_until_stalled,
    apply_unique_cats_exact_solution,
    place_cat,
    solve_cats_exact,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowInfo

type AnalyzeBoardFunction = Callable[[Screenshot], CatsBoardInput]


def count_unresolved_cells(board: Board) -> int:
    """Count only current unresolved ``C<n>`` entries in the mutable Board."""

    return sum(
        board.is_unknown(row, column)
        for row, values in enumerate(board.cells)
        for column in range(len(values))
    )


def classify_result(board: Board) -> CatsSolveStatus:
    """Classify a board as complete or safely stalled with unknown cells."""

    if count_unresolved_cells(board) == 0:
        return CatsSolveStatus.COMPLETE
    return CatsSolveStatus.STALLED


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
    if status is CatsSolveStatus.STALLED:
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
            if status is not CatsSolveStatus.COMPLETE:
                raise CatsExactSearchError(
                    "Applying a UNIQUE exact solution did not complete Board."
                )
        elif exact_search_result.status is CatsExactSearchStatus.UNSAT:
            status = CatsSolveStatus.UNSAT
        elif exact_search_result.status is CatsExactSearchStatus.AMBIGUOUS:
            status = CatsSolveStatus.AMBIGUOUS
        else:
            status = CatsSolveStatus.SEARCH_LIMIT

    click_plan: tuple[CatClickTarget, ...] = ()
    if status is CatsSolveStatus.COMPLETE:
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
    *,
    analyze_board: AnalyzeBoardFunction,
) -> CatsSolvedBoard:
    """Compose immutable vision analysis with exactly one logical solve."""

    return solve_analyzed_cats_board(window, analyze_board(screenshot))
