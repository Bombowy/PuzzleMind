"""Tests for the deterministic, non-mutating Cats exact-search fallback."""

from __future__ import annotations

from copy import deepcopy
from inspect import getsource

import pytest

from logicforge.core.board import Board
from logicforge.plugins.cats import (
    CatsExactSearchError,
    CatsExactSearchResult,
    CatsExactSearchStatus,
    apply_cats_rules_until_stalled,
    apply_unique_cats_exact_solution,
    exact_search,
    solve_cats_exact,
)
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)

type Coordinates = tuple[int, int]
type ColorMatrix = tuple[tuple[str, ...], ...]

_FOUR_BY_FOUR_SOLUTIONS = (
    ((0, 1), (1, 3), (2, 0), (3, 2)),
    ((0, 2), (1, 0), (2, 3), (3, 1)),
)
_SEVEN_BY_SEVEN_SOLUTION = (
    (0, 6),
    (1, 0),
    (2, 5),
    (3, 3),
    (4, 1),
    (5, 4),
    (6, 2),
)
_SEVEN_BY_SEVEN_ORIGINAL = (
    ("C1", "C3", "C6", "C1", "C4", "C1", "C0"),
    ("C1", "C5", "C1", "C0", "C3", "C3", "C3"),
    ("C2", "C0", "C1", "C5", "C6", "C2", "C1"),
    ("C6", "C6", "C1", "C3", "C5", "C1", "C3"),
    ("C4", "C4", "C6", "C3", "C5", "C3", "C0"),
    ("C4", "C4", "C3", "C4", "C5", "C3", "C6"),
    ("C2", "C5", "C6", "C2", "C3", "C0", "C0"),
)
_SEVEN_BY_SEVEN_STALLED = (
    ("X", "X", "C6", "C1", "C4", "X", "C0"),
    ("C1", "X", "C1", "X", "C3", "X", "C3"),
    ("C2", "C0", "X", "X", "C6", "C2", "C1"),
    ("X", "C6", "C1", "C3", "X", "X", "C3"),
    ("X", "C4", "C6", "X", "C5", "C3", "X"),
    ("X", "C4", "C3", "C4", "C5", "X", "C6"),
    ("C2", "C5", "C6", "C2", "X", "C0", "X"),
)


def _color_result(matrix: ColorMatrix) -> ColorDetectionResult:
    rows = len(matrix)
    columns = len(matrix[0])
    color_ids = sorted(
        {value for row in matrix for value in row},
        key=lambda value: int(value[1:]),
    )
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=matrix[row][column],
            confidence=1.0,
            representative_lab=(30.0 + index, 128.0, 128.0),
        )
        for row in range(rows)
        for column in range(columns)
        for index in (int(matrix[row][column][1:]),)
    )
    return ColorDetectionResult(
        observations=observations,
        color_count=len(color_ids),
        color_matrix=matrix,
        mean_confidence=1.0,
        diagnostics=ColorDetectionDiagnostics(
            rows=rows,
            columns=columns,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(16,) * (rows * columns),
            within_cell_spreads=(0.0,) * (rows * columns),
            cluster_centers_lab=tuple(
                (30.0 + index, 128.0, 128.0) for index in range(len(color_ids))
            ),
            minimum_intercluster_distance=20.0,
        ),
    )


def _column_colors(size: int) -> ColorMatrix:
    return tuple(tuple(f"C{column}" for column in range(size)) for _ in range(size))


def _latin_colors(size: int) -> ColorMatrix:
    return tuple(
        tuple(f"C{(row + column) % size}" for column in range(size))
        for row in range(size)
    )


def _board_with_candidates(
    original: ColorMatrix,
    candidates: set[Coordinates] | frozenset[Coordinates],
    *,
    fixed_cats: tuple[Coordinates, ...] = (),
) -> Board:
    board = Board(_color_result(original))
    fixed = frozenset(fixed_cats)
    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            coordinate = (row, column)
            if coordinate in fixed:
                board.cells[row][column] = "K"
            elif coordinate not in candidates:
                board.cells[row][column] = "X"
    return board


def _solution_candidates(columns: tuple[int, ...]) -> set[Coordinates]:
    return {(row, column) for row, column in enumerate(columns)}


def _assert_complete_board(
    board: Board,
    solution: tuple[Coordinates, ...],
) -> None:
    solution_set = frozenset(solution)
    for row, values in enumerate(board.cells):
        for column, value in enumerate(values):
            assert value == ("K" if (row, column) in solution_set else "X")


def test_already_complete_board_is_unique() -> None:
    original = _column_colors(4)
    solution = _FOUR_BY_FOUR_SOLUTIONS[0]
    board = _board_with_candidates(original, set(), fixed_cats=solution)

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.UNIQUE
    assert result.solution == solution
    assert result.search_nodes == 1


def test_simple_unique_four_by_four_solution() -> None:
    original = _column_colors(4)
    first, second = _FOUR_BY_FOUR_SOLUTIONS
    candidates = set(first) | set(second)
    candidates.remove(second[0])
    board = _board_with_candidates(original, candidates)

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.UNIQUE
    assert result.solution == first


def test_unique_solution_can_be_applied_to_same_board() -> None:
    original = _column_colors(4)
    solution = _FOUR_BY_FOUR_SOLUTIONS[0]
    board = _board_with_candidates(original, set(solution))
    board_identity = id(board)

    result = solve_cats_exact(board, original)
    apply_unique_cats_exact_solution(
        board,
        result,
        original_color_matrix=original,
    )

    assert id(board) == board_identity
    _assert_complete_board(board, solution)


def test_unsat_when_one_color_has_zero_candidates() -> None:
    original = _latin_colors(4)
    candidates = {
        (row, column)
        for row in range(4)
        for column in range(4)
        if original[row][column] != "C0"
    }

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.status is CatsExactSearchStatus.UNSAT


def test_unsat_when_one_row_has_zero_candidates() -> None:
    original = _column_colors(4)
    candidates = {(row, column) for row in range(1, 4) for column in range(4)}

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.status is CatsExactSearchStatus.UNSAT


def test_unsat_when_one_column_has_zero_candidates() -> None:
    original = _column_colors(4)
    candidates = {(row, column) for row in range(4) for column in range(1, 4)}

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.status is CatsExactSearchStatus.UNSAT


@pytest.mark.parametrize(
    ("original", "fixed_cats"),
    (
        (_column_colors(4), ((0, 0), (1, 1))),
        (_column_colors(4), ((0, 0), (0, 2))),
        (_column_colors(4), ((0, 0), (2, 0))),
        (_latin_colors(4), ((0, 0), (2, 2))),
    ),
    ids=(
        "touching",
        "duplicate-row",
        "duplicate-column",
        "duplicate-original-color",
    ),
)
def test_contradictory_fixed_cats_are_unsat(
    original: ColorMatrix,
    fixed_cats: tuple[Coordinates, ...],
) -> None:
    board = _board_with_candidates(original, set(), fixed_cats=fixed_cats)

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.UNSAT
    assert result.search_nodes == 0


def test_full_four_by_four_board_has_exactly_two_solutions() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.AMBIGUOUS
    assert result.solution is None
    assert result.solutions_found == 2


def test_search_stops_after_the_second_solution() -> None:
    original = _column_colors(5)
    board = Board(_color_result(original))

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.AMBIGUOUS
    assert result.solutions_found == 2


def test_node_limit_is_not_treated_as_unique() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))

    result = solve_cats_exact(board, original, maximum_search_nodes=1)

    assert result.status is CatsExactSearchStatus.LIMIT_REACHED
    assert result.solution is None
    assert result.search_nodes == 1


def test_repeated_search_is_fully_deterministic() -> None:
    original = _column_colors(5)
    board = Board(_color_result(original))

    results = tuple(solve_cats_exact(board, original) for _ in range(4))

    assert results == (results[0],) * 4


def test_search_source_uses_no_random_or_board_clone() -> None:
    source = getsource(exact_search).casefold()

    assert "random" not in source
    assert "deepcopy" not in source
    assert "copy.copy" not in source


def test_search_does_not_mutate_board_or_original_matrix() -> None:
    original = _latin_colors(4)
    board = Board(_color_result(original))
    board_before = deepcopy(board.cells)
    original_before = deepcopy(original)

    solve_cats_exact(board, original)

    assert board.cells == board_before
    assert original == original_before


def test_rectangular_input_is_rejected() -> None:
    original = (
        ("C0", "C1", "C2"),
        ("C1", "C2", "C0"),
    )
    board = Board(_color_result(original))

    with pytest.raises(CatsExactSearchError, match="square"):
        solve_cats_exact(board, original)


def test_invalid_current_board_value_is_rejected() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))
    board.cells[1][2] = "CAT"

    with pytest.raises(CatsExactSearchError, match="Invalid current Board value"):
        solve_cats_exact(board, original)


def test_original_matrix_shape_mismatch_is_rejected() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))

    with pytest.raises(CatsExactSearchError, match="shape"):
        solve_cats_exact(board, original[:-1])


def test_unresolved_value_must_match_original_color() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))
    board.cells[0][0] = "C1"

    with pytest.raises(CatsExactSearchError, match="does not match"):
        solve_cats_exact(board, original)


def test_fixed_existing_cat_is_retained_in_unique_solution() -> None:
    original = _column_colors(4)
    solution = _FOUR_BY_FOUR_SOLUTIONS[0]
    fixed = solution[0]
    board = _board_with_candidates(
        original,
        set(solution) - {fixed},
        fixed_cats=(fixed,),
    )

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.UNIQUE
    assert result.solution == solution


def test_mrv_chooses_smallest_color_group() -> None:
    original = _latin_colors(4)
    candidates = {
        (row, column)
        for row in range(4)
        for column in range(4)
        if original[row][column] != "C0" or (row, column) in {(0, 0), (2, 2)}
    }
    board = _board_with_candidates(original, candidates)

    result = solve_cats_exact(board, original)

    assert result.branch_groups[0] == "color:C0"


def test_mrv_equal_size_tie_prefers_color_then_numeric_identifier() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))

    result = solve_cats_exact(board, original)

    assert result.branch_groups[0] == "color:C0"


def test_color_singleton_propagates_first() -> None:
    original = _latin_colors(4)
    candidates = {
        (row, column)
        for row in range(4)
        for column in range(4)
        if original[row][column] != "C0" or (row, column) == (0, 0)
    }

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.propagation_groups[0] == "color:C0"


def test_row_singleton_propagates_when_colors_are_not_singletons() -> None:
    original = _latin_colors(4)
    candidates = {(0, 0)} | {
        (row, column) for row in range(1, 4) for column in range(4)
    }

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.propagation_groups[0] == "row:0"


def test_column_singleton_propagates_when_other_groups_are_not_singletons() -> None:
    original = _latin_colors(4)
    candidates = {(0, 0)} | {
        (row, column) for row in range(4) for column in range(1, 4)
    }

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.propagation_groups[0] == "column:0"


def test_adjacency_filtering_creates_a_forced_row_singleton() -> None:
    original = _column_colors(4)
    candidates = {
        (row, column)
        for row in range(4)
        for column in range(4)
        if column != 1 or (row, column) == (0, 1)
    }

    result = solve_cats_exact(
        _board_with_candidates(original, candidates),
        original,
    )

    assert result.propagation_groups[:2] == ("color:C1", "row:1")


def test_apply_rejects_non_unique_result_without_mutating_board() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))
    before = deepcopy(board.cells)
    result = CatsExactSearchResult(
        status=CatsExactSearchStatus.AMBIGUOUS,
        solution=None,
        solutions_found=2,
        search_nodes=5,
        propagation_steps=0,
    )

    with pytest.raises(CatsExactSearchError, match="Only a UNIQUE"):
        apply_unique_cats_exact_solution(board, result)

    assert board.cells == before


@pytest.mark.parametrize(
    ("size", "columns"),
    (
        (7, (6, 0, 5, 3, 1, 4, 2)),
        (8, (0, 2, 4, 6, 1, 3, 7, 5)),
        (9, (0, 2, 4, 6, 8, 1, 3, 7, 5)),
        (10, (0, 2, 4, 6, 8, 1, 3, 5, 9, 7)),
    ),
)
def test_uniquely_constrained_sizes_finish_below_node_limit(
    size: int,
    columns: tuple[int, ...],
) -> None:
    original = _column_colors(size)
    candidates = _solution_candidates(columns)
    board = _board_with_candidates(original, candidates)

    result = solve_cats_exact(board, original)

    assert result.status is CatsExactSearchStatus.UNIQUE
    assert result.solution == tuple(sorted(candidates))
    assert result.search_nodes < 250_000


def test_live_like_seven_by_seven_stalls_rules_then_searches_uniquely() -> None:
    board = Board(_color_result(_SEVEN_BY_SEVEN_ORIGINAL))
    board.cells = [list(row) for row in _SEVEN_BY_SEVEN_STALLED]
    before_search = deepcopy(board.cells)

    successful_applications = apply_cats_rules_until_stalled(board)
    result = solve_cats_exact(board, _SEVEN_BY_SEVEN_ORIGINAL)

    assert successful_applications == 0
    assert result.status is CatsExactSearchStatus.UNIQUE
    assert result.solution == _SEVEN_BY_SEVEN_SOLUTION
    assert result.search_nodes == 8
    assert result.propagation_steps == 11
    assert board.cells == before_search


def test_invalid_search_limits_are_rejected() -> None:
    original = _column_colors(4)
    board = Board(_color_result(original))

    with pytest.raises(CatsExactSearchError, match="exactly two"):
        solve_cats_exact(board, original, maximum_solutions=1)
    with pytest.raises(CatsExactSearchError, match="positive"):
        solve_cats_exact(board, original, maximum_search_nodes=0)
