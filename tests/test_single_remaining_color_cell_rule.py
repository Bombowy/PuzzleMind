"""Tests for the first concrete, apply-style Cats deduction rule."""

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import SingleRemainingColorCellRule
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)


def _board_from_values(values: tuple[tuple[str, ...], ...]) -> Board:
    """Create a Board, then configure terminal values solely as test fixture state."""

    rows = len(values)
    columns = len(values[0])
    base_matrix = tuple(tuple("C0" for _ in row) for row in values)
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id="C0",
            confidence=1.0,
            representative_lab=(120.0, 130.0, 140.0),
        )
        for row in range(rows)
        for column in range(columns)
    )
    result = ColorDetectionResult(
        observations=observations,
        color_count=1,
        color_matrix=base_matrix,
        mean_confidence=1.0,
        diagnostics=ColorDetectionDiagnostics(
            rows=rows,
            columns=columns,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * (rows * columns),
            within_cell_spreads=(1.0,) * (rows * columns),
            cluster_centers_lab=((120.0, 130.0, 140.0),),
            minimum_intercluster_distance=None,
        ),
    )
    board = Board(result)
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            board.cells[row][column] = value
    return board


def _values(board: Board) -> tuple[tuple[str, ...], ...]:
    """Capture test-only values for all-or-nothing assertions."""

    return tuple(tuple(row) for row in board.cells)


def _positive_board() -> Board:
    """Return the documented case where only C0 has one unresolved cell."""

    return _board_from_values(
        (
            ("C0", "C1", "C1", "C2"),
            ("X", "C3", "C2", "C3"),
            ("X", "C4", "C4", "C2"),
            ("X", "C5", "C5", "C3"),
        )
    )


def test_single_remaining_c0_becomes_cat() -> None:
    """Find the only unresolved C0 and delegate its C0 to K transition."""

    board = _positive_board()

    SingleRemainingColorCellRule().apply(board)

    assert board.get(0, 0) == "K"


def test_apply_returns_true_after_placing_forced_cat() -> None:
    """Report a real mutation rather than merely finding a singleton color."""

    board = _positive_board()

    assert SingleRemainingColorCellRule().apply(board) is True


def test_rule_uses_place_cat_full_exclusion_propagation() -> None:
    """Observe row, column, and neighbor X values produced by the shared action."""

    board = _positive_board()

    SingleRemainingColorCellRule().apply(board)

    assert tuple(board.get(0, column) for column in range(4)) == (
        "K",
        "X",
        "X",
        "X",
    )
    assert board.is_blocked(1, 0)
    assert board.is_blocked(1, 1)


def test_color_with_two_unknown_cells_does_not_trigger_move() -> None:
    """Require exactly one current coordinate rather than one or more."""

    board = _board_from_values((("C0", "C0"), ("C1", "C1")))

    SingleRemainingColorCellRule().apply(board)

    assert not any(value == "K" for row in board.cells for value in row)


def test_apply_returns_false_when_no_color_is_singleton() -> None:
    """Signal explicitly that the rule found no forced move."""

    board = _board_from_values((("C0", "C0"), ("C1", "C1")))

    assert SingleRemainingColorCellRule().apply(board) is False


def test_false_result_leaves_entire_board_unchanged() -> None:
    """Keep the sole matrix untouched when every color has multiple candidates."""

    board = _board_from_values((("C0", "C0"), ("C1", "C1")))
    expected = _values(board)

    SingleRemainingColorCellRule().apply(board)

    assert _values(board) == expected


def test_blocked_cells_are_not_counted_as_color_candidates() -> None:
    """Treat one current C0 plus terminal X values as a singleton C0 class."""

    board = _board_from_values((("C0", "X", "C1"), ("X", "C1", "X")))

    SingleRemainingColorCellRule().apply(board)

    assert board.is_cat(0, 0)


def test_existing_cats_are_not_counted_as_color_candidates() -> None:
    """Ignore terminal K entries while finding the only unresolved C0."""

    board = _board_from_values(
        (
            ("C0", "C1", "C1", "C2"),
            ("C2", "C3", "C3", "C4"),
            ("C4", "C5", "C5", "C6"),
            ("C6", "C7", "C7", "K"),
        )
    )

    SingleRemainingColorCellRule().apply(board)

    assert board.is_cat(0, 0)
    assert board.is_cat(3, 3)


def test_one_apply_places_at_most_one_of_multiple_singletons() -> None:
    """Return immediately after the first place_cat call."""

    board = _board_from_values(
        (
            ("C2", "C6", "C6", "C6"),
            ("C6", "C6", "C6", "C6"),
            ("C6", "C6", "C6", "C6"),
            ("C6", "C6", "C6", "C5"),
        )
    )

    SingleRemainingColorCellRule().apply(board)

    assert sum(value == "K" for row in board.cells for value in row) == 1


def test_lowest_numeric_color_id_is_selected_first() -> None:
    """Order singleton classes numerically rather than by position or text."""

    board = _board_from_values(
        (
            ("C3", "C4", "C4", "C4"),
            ("C4", "C4", "C4", "C4"),
            ("C4", "C4", "C4", "C4"),
            ("C4", "C4", "C4", "C1"),
        )
    )

    SingleRemainingColorCellRule().apply(board)

    assert board.is_cat(3, 3)
    assert board.get(0, 0) == "C3"


def test_c2_is_selected_before_c10() -> None:
    """Use integer suffix ordering so C10 does not sort before C2."""

    board = _board_from_values(
        (
            ("C2", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C10"),
        )
    )

    SingleRemainingColorCellRule().apply(board)

    assert board.is_cat(0, 0)
    assert board.get(3, 3) == "C10"


def test_second_apply_can_place_next_remaining_singleton() -> None:
    """Re-scan current cells on every call without storing rule state."""

    board = _board_from_values(
        (
            ("C2", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C10"),
        )
    )
    rule = SingleRemainingColorCellRule()

    assert rule.apply(board) is True
    assert rule.apply(board) is True

    assert board.is_cat(0, 0)
    assert board.is_cat(3, 3)


def test_place_cat_conflict_propagates_as_board_state_error() -> None:
    """Do not catch or translate an existing-cat contradiction."""

    board = _board_from_values(
        (
            ("C0", "C1", "C1", "K"),
            ("C1", "C1", "C1", "C1"),
            ("C1", "C1", "C1", "C1"),
            ("C1", "C1", "C1", "C1"),
        )
    )

    with pytest.raises(BoardStateError, match="existing cat"):
        SingleRemainingColorCellRule().apply(board)


def test_place_cat_conflict_leaves_complete_board_unchanged() -> None:
    """Retain atomic action behavior when the rule surfaces a contradiction."""

    board = _board_from_values(
        (
            ("C0", "C1", "C1", "K"),
            ("C1", "C1", "C1", "C1"),
            ("C1", "C1", "C1", "C1"),
            ("C1", "C1", "C1", "C1"),
        )
    )
    expected = _values(board)

    with pytest.raises(BoardStateError):
        SingleRemainingColorCellRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_board() -> None:
    """Scan row-major rectangular geometry without square-board assumptions."""

    board = _board_from_values((("C0", "C1", "C1", "C1"), ("C1", "C1", "C1", "C1")))

    changed = SingleRemainingColorCellRule().apply(board)

    assert changed is True
    assert board.is_cat(0, 0)
