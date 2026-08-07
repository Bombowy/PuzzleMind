"""Tests for the Cats color-confined-to-row-or-column deduction."""

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import ColorConfinedToLineRule
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)


def _board_from_values(values: tuple[tuple[str, ...], ...]) -> Board:
    """Build Board transport, then configure terminal values as test-only state."""

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
    """Capture a test-only value view for all-or-nothing assertions."""

    return tuple(tuple(row) for row in board.cells)


def _row_case() -> Board:
    """Return the documented board whose two C0 candidates share row zero."""

    return _board_from_values(
        (
            ("C0", "C1", "C0", "C2"),
            ("C3", "C1", "C4", "C2"),
            ("C3", "C5", "C4", "C5"),
        )
    )


def _column_case() -> Board:
    """Return a column-confined C0 case with one other candidate in its column."""

    return _board_from_values(
        (
            ("C0", "C2", "C3"),
            ("C1", "C2", "C3"),
            ("C0", "C4", "C5"),
            ("X", "C4", "C5"),
        )
    )


def test_row_confined_color_blocks_other_colors_in_that_row() -> None:
    """Block C1 and C2 because every unresolved C0 lies in row zero."""

    board = _row_case()

    ColorConfinedToLineRule().apply(board)

    assert tuple(board.get(0, column) for column in range(4)) == (
        "C0",
        "X",
        "C0",
        "X",
    )


def test_row_confined_color_cells_remain_unresolved() -> None:
    """Never block the C0 cells whose possible cat position caused the deduction."""

    board = _row_case()

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 0) == "C0"
    assert board.get(0, 2) == "C0"


def test_apply_returns_true_after_at_least_one_new_block() -> None:
    """Report an actual C<n> to X change rather than confinement alone."""

    board = _row_case()

    assert ColorConfinedToLineRule().apply(board) is True


def test_column_confined_color_blocks_other_colors_in_that_column() -> None:
    """Block C1 because all unresolved C0 cells share column zero."""

    board = _column_case()

    ColorConfinedToLineRule().apply(board)

    assert board.get(1, 0) == "X"


def test_column_confined_color_cells_remain_unresolved() -> None:
    """Preserve every studied C0 coordinate in the confined column."""

    board = _column_case()

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 0) == "C0"
    assert board.get(2, 0) == "C0"


def test_color_spanning_multiple_rows_and_columns_causes_no_move() -> None:
    """Reject diagonal distributions as neither row nor column confinement."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))

    ColorConfinedToLineRule().apply(board)

    assert not any(value == "X" for row in board.cells for value in row)


def test_apply_returns_false_when_no_color_is_confined() -> None:
    """Return False when every multi-candidate color spans both axes."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))

    assert ColorConfinedToLineRule().apply(board) is False


def test_false_result_leaves_complete_board_unchanged() -> None:
    """Perform no mutation when no color supplies line-based information."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))
    expected = _values(board)

    ColorConfinedToLineRule().apply(board)

    assert _values(board) == expected


def test_existing_x_remains_x_during_row_deduction() -> None:
    """Treat an already satisfied exclusion as valid without rewriting it."""

    board = _board_from_values((("C0", "X", "C0", "C2"), ("C1", "C2", "C1", "C2")))

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(0, 3) == "X"


def test_k_and_x_are_not_grouped_as_color_candidates() -> None:
    """Group only current C<n> values while preserving distant terminal cells."""

    board = _board_from_values(
        (
            ("C0", "C1", "C0", "C1"),
            ("C2", "X", "C2", "C3"),
            ("C4", "C5", "C4", "C5"),
            ("C6", "C7", "C6", "K"),
        )
    )

    ColorConfinedToLineRule().apply(board)

    assert board.get(1, 1) == "X"
    assert board.get(3, 3) == "K"


def test_singleton_color_is_skipped_for_next_confined_color() -> None:
    """Leave C0 to SingleRemainingColorCellRule and apply the C1 row deduction."""

    board = _board_from_values(
        (
            ("C0", "C3", "C3", "C4"),
            ("C1", "C2", "C1", "C2"),
            ("C4", "C5", "C4", "C5"),
        )
    )

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 0) == "C0"
    assert board.get(1, 1) == "X"
    assert board.get(1, 3) == "X"


def test_lowest_numeric_qualifying_color_is_selected() -> None:
    """Handle C2 before a separately confined higher-numbered color."""

    board = _board_from_values(
        (
            ("C2", "C4", "C2", "C4"),
            ("C4", "C4", "C4", "C4"),
            ("C4", "C4", "C4", "C4"),
            ("C3", "C5", "C3", "C5"),
        )
    )

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(3, 1) == "C5"


def test_c2_is_selected_before_c10() -> None:
    """Sort color suffixes numerically rather than lexicographically."""

    board = _board_from_values(
        (
            ("C2", "C11", "C2", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C10", "C12", "C10", "C12"),
        )
    )

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(3, 1) == "C12"


def test_one_apply_handles_at_most_one_color() -> None:
    """Return after blocking the first qualifying color's line."""

    board = _board_from_values(
        (
            ("C2", "C11", "C2", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C11", "C11", "C11", "C11"),
            ("C10", "C12", "C10", "C12"),
        )
    )

    ColorConfinedToLineRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(0, 3) == "X"
    assert board.get(3, 1) == "C12"
    assert board.get(3, 3) == "C12"


def test_one_apply_can_block_multiple_cells_in_selected_line() -> None:
    """Apply every validated target for one color before returning True."""

    board = _row_case()

    ColorConfinedToLineRule().apply(board)

    assert board.is_blocked(0, 1)
    assert board.is_blocked(0, 3)


def test_no_change_for_first_confined_color_continues_to_next_color() -> None:
    """Skip satisfied C0 row/C1 column plans and find a mutable C2 row."""

    board = _board_from_values(
        (
            ("X", "C0", "X", "C0"),
            ("C1", "C2", "C3", "C2"),
            ("X", "C4", "C4", "C5"),
            ("C1", "C6", "C6", "C7"),
        )
    )

    changed = ColorConfinedToLineRule().apply(board)

    assert changed is True
    assert board.get(0, 1) == "C0"
    assert board.get(0, 3) == "C0"
    assert board.get(1, 0) == "X"
    assert board.get(1, 2) == "X"


def test_cat_in_planned_line_raises_board_state_error() -> None:
    """Reject a line whose existing K would need to become X."""

    board = _board_from_values((("C0", "K", "C0", "C2"), ("C1", "C2", "C1", "C2")))

    with pytest.raises(BoardStateError, match="existing cat"):
        ColorConfinedToLineRule().apply(board)


def test_cat_conflict_leaves_complete_board_unchanged() -> None:
    """Validate the whole plan before applying its first block_cell mutation."""

    board = _board_from_values((("C0", "K", "C0", "C2"), ("C1", "C2", "C1", "C2")))
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ColorConfinedToLineRule().apply(board)

    assert _values(board) == expected


def test_invalid_value_in_line_raises_without_partial_mutation() -> None:
    """Reject unsupported state after planning but before any earlier target write."""

    board = _board_from_values(
        (("C0", "C1", "C0", "INVALID"), ("C2", "C1", "C2", "C3"))
    )
    expected = _values(board)

    with pytest.raises(BoardStateError, match="invalid value"):
        ColorConfinedToLineRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_board() -> None:
    """Apply row confinement on non-square matrix geometry."""

    board = _board_from_values(
        (
            ("C0", "C1", "C0", "C2", "C3"),
            ("C4", "C1", "C4", "C2", "C3"),
            ("C5", "C6", "C5", "C6", "C7"),
        )
    )

    changed = ColorConfinedToLineRule().apply(board)

    assert changed is True
    assert tuple(board.get(0, column) for column in range(5)) == (
        "C0",
        "X",
        "C0",
        "X",
        "X",
    )
