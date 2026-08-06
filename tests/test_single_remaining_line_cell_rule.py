"""Tests for the Cats single remaining row-or-column cell deduction."""

import inspect

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import SingleRemainingLineCellRule
from logicforge.plugins.cats import single_remaining_line_cell_rule as rule_module
from logicforge.plugins.cats.board_actions import place_cat as real_place_cat
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
            sample_inner_fraction=0.65,
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
    """Capture a test-only immutable view for all-or-nothing assertions."""

    return tuple(tuple(row) for row in board.cells)


def _row_case() -> Board:
    """Return a first row with one C0 and three blocked coordinates."""

    return _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C1", "C2", "C3", "C0"),
            ("C1", "C2", "C3", "C0"),
            ("C4", "C4", "C5", "C5"),
        )
    )


def _column_case() -> Board:
    """Return a first column with one C0 and two blocked coordinates."""

    return _board_from_values(
        (
            ("X", "C1", "C2"),
            ("X", "C3", "C4"),
            ("C0", "C5", "C6"),
        )
    )


def _two_independent_rows() -> Board:
    """Return two forced rows far enough apart for consecutive placements."""

    return _board_from_values(
        (
            ("X", "X", "C0", "X", "X"),
            ("X", "X", "X", "X", "X"),
            ("C1", "C2", "C3", "C4", "C5"),
            ("X", "X", "X", "X", "X"),
            ("C6", "X", "X", "X", "X"),
        )
    )


def test_row_with_one_unknown_and_other_x_places_cat() -> None:
    """Place K in the sole unresolved coordinate of the first row."""

    board = _row_case()

    SingleRemainingLineCellRule().apply(board)

    assert tuple(board.get(0, column) for column in range(4)) == (
        "X",
        "X",
        "K",
        "X",
    )


def test_column_with_one_unknown_and_other_x_places_cat() -> None:
    """Place K in the sole unresolved coordinate of the first column."""

    board = _column_case()

    SingleRemainingLineCellRule().apply(board)

    assert board.get(2, 0) == "K"


def test_apply_returns_true_after_real_cat_placement() -> None:
    """Return the exact successful result delegated by place_cat."""

    assert SingleRemainingLineCellRule().apply(_row_case()) is True


def test_rule_uses_full_place_cat_propagation() -> None:
    """Apply color, row, column, and neighboring exclusions from one forced K."""

    board = _row_case()

    SingleRemainingLineCellRule().apply(board)

    assert (board.get(1, 3), board.get(2, 3)) == ("X", "X")
    assert tuple(board.get(row, 2) for row in range(1, 4)) == ("X", "X", "X")
    assert (board.get(1, 1), board.get(1, 3)) == ("X", "X")
    assert tuple(board.get(0, column) for column in range(4)) == (
        "X",
        "X",
        "K",
        "X",
    )


def test_forced_cell_color_may_still_exist_elsewhere() -> None:
    """Use line occupancy rather than global color candidate count."""

    board = _row_case()
    assert tuple(
        board.get(row, column) for row, column in ((0, 2), (1, 3), (2, 3))
    ) == ("C0", "C0", "C0")

    SingleRemainingLineCellRule().apply(board)

    assert board.get(0, 2) == "K"


def test_two_same_color_cells_in_line_do_not_qualify() -> None:
    """Reject a line with two possibilities even when their color IDs match."""

    board = _board_from_values(
        (
            ("X", "C0", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("C5", "C6", "C7", "C8"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is False


def test_two_different_color_cells_in_line_do_not_qualify() -> None:
    """Reject a line with two possibilities regardless of their equality."""

    board = _board_from_values(
        (
            ("X", "C0", "C1", "X"),
            ("C2", "C3", "C4", "C5"),
            ("C6", "C7", "C8", "C9"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is False


def test_all_x_line_does_not_qualify() -> None:
    """Require exactly one unresolved coordinate rather than zero."""

    board = _board_from_values(
        (
            ("X", "X", "X", "X"),
            ("C0", "C1", "C2", "C3"),
            ("C4", "C5", "C6", "C7"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is False


def test_line_containing_cat_is_skipped() -> None:
    """Treat any confirmed K as proof that the line needs no second cat."""

    board = _board_from_values(
        (
            ("K", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("C5", "C6", "C7", "C8"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is False


def test_cat_x_unknown_x_line_does_not_place_second_cat() -> None:
    """Leave the remaining C0 unchanged when its row already contains K."""

    board = _board_from_values(
        (
            ("K", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("C5", "C6", "C7", "C8"),
        )
    )

    SingleRemainingLineCellRule().apply(board)

    assert board.get(0, 0) == "K"
    assert board.get(0, 2) == "C0"


def test_skipped_cat_row_allows_later_qualifying_row() -> None:
    """Continue row scanning after K and place in the next forced row."""

    board = _board_from_values(
        (
            ("K", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("X", "X", "C5", "X"),
            ("C6", "C7", "C8", "C9"),
        )
    )

    changed = SingleRemainingLineCellRule().apply(board)

    assert changed is True
    assert board.get(2, 2) == "K"


def test_skipped_cat_column_allows_later_qualifying_column() -> None:
    """Continue column scanning after K and place in the next forced column."""

    board = _board_from_values(
        (
            ("K", "X", "C3", "C4"),
            ("X", "X", "C5", "C6"),
            ("C0", "C2", "C7", "C8"),
        )
    )

    changed = SingleRemainingLineCellRule().apply(board)

    assert changed is True
    assert board.get(2, 1) == "K"


def test_row_is_handled_before_simultaneously_qualifying_column() -> None:
    """Prefer a forced row over a lower-priority forced column."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("X", "C1", "C2", "C3"),
            ("C5", "C4", "C6", "C7"),
        )
    )

    SingleRemainingLineCellRule().apply(board)

    assert board.get(0, 2) == "K"
    assert board.get(2, 0) == "C5"


def test_lower_row_index_is_handled_first() -> None:
    """Select row zero before a separately forced row at index two."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("X", "C5", "X", "X"),
        )
    )

    SingleRemainingLineCellRule().apply(board)

    assert board.get(0, 2) == "K"
    assert board.get(2, 1) == "C5"


def test_lower_column_index_is_handled_first() -> None:
    """Select column zero before a separately forced column at index two."""

    board = _board_from_values(
        (
            ("X", "C1", "X", "C2"),
            ("X", "C3", "X", "C4"),
            ("C0", "C5", "C6", "C7"),
        )
    )

    SingleRemainingLineCellRule().apply(board)

    assert board.get(2, 0) == "K"


def test_one_apply_places_at_most_one_cat() -> None:
    """Return immediately after the first forced line placement."""

    board = _two_independent_rows()

    SingleRemainingLineCellRule().apply(board)

    assert sum(value == "K" for row in board.cells for value in row) == 1
    assert board.get(4, 0) == "C6"


def test_second_apply_can_place_next_independent_cat() -> None:
    """Allow a later invocation to consume another still-forced line."""

    board = _two_independent_rows()
    rule = SingleRemainingLineCellRule()

    first_changed = rule.apply(board)
    second_changed = rule.apply(board)

    assert first_changed is True
    assert second_changed is True
    assert (board.get(0, 2), board.get(4, 0)) == ("K", "K")


def test_no_qualifying_line_returns_false() -> None:
    """Return False when every row and column retains multiple possibilities."""

    board = _board_from_values((("C0", "C1"), ("C2", "C3")))

    assert SingleRemainingLineCellRule().apply(board) is False


def test_false_result_leaves_complete_board_unchanged() -> None:
    """Perform no mutation when no line has exactly one possible cell."""

    board = _board_from_values((("C0", "C1"), ("C2", "C3")))
    expected = _values(board)

    SingleRemainingLineCellRule().apply(board)

    assert _values(board) == expected


def test_invalid_value_in_row_raises_board_state_error() -> None:
    """Report an unsupported row value with exact coordinates and content."""

    board = _board_from_values((("INVALID", "X"), ("C0", "C1")))

    with pytest.raises(BoardStateError, match=r"INVALID.*\(0, 0\)"):
        SingleRemainingLineCellRule().apply(board)


def test_invalid_value_in_column_raises_board_state_error() -> None:
    """Reject an unsupported value before it can participate in column logic."""

    board = _board_from_values((("K", "C0"), ("INVALID", "C1")))

    with pytest.raises(BoardStateError, match=r"INVALID.*\(1, 0\)"):
        SingleRemainingLineCellRule().apply(board)


def test_invalid_value_error_occurs_before_any_mutation() -> None:
    """Validate all values before an earlier forced row can place its cat."""

    board = _board_from_values((("X", "C0", "X"), ("C1", "C2", "INVALID")))
    expected = _values(board)

    with pytest.raises(BoardStateError):
        SingleRemainingLineCellRule().apply(board)

    assert _values(board) == expected


def test_place_cat_conflict_propagates_board_state_error() -> None:
    """Surface an existing K in the forced candidate's column."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("C5", "C6", "K", "C7"),
        )
    )

    with pytest.raises(BoardStateError, match="existing cat"):
        SingleRemainingLineCellRule().apply(board)


def test_place_cat_conflict_leaves_complete_board_unchanged() -> None:
    """Rely on atomic place_cat planning instead of implementing rollback."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C1", "C2", "C3", "C4"),
            ("C5", "C6", "K", "C7"),
        )
    )
    expected = _values(board)

    with pytest.raises(BoardStateError):
        SingleRemainingLineCellRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_three_by_five_board() -> None:
    """Apply a row deduction without assuming square or 8x8 geometry."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X", "X"),
            ("C1", "C2", "C3", "C4", "C5"),
            ("C6", "C7", "C8", "C9", "C10"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is True
    assert board.get(0, 2) == "K"


def test_rule_supports_rectangular_five_by_three_board() -> None:
    """Apply a column deduction on tall rectangular geometry."""

    board = _board_from_values(
        (
            ("X", "C1", "C2"),
            ("X", "C3", "C4"),
            ("X", "C5", "C6"),
            ("X", "C7", "C8"),
            ("C0", "C9", "C10"),
        )
    )

    assert SingleRemainingLineCellRule().apply(board) is True
    assert board.get(4, 0) == "K"


def test_rule_uses_place_cat_and_not_block_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate the forced placement and all propagation only to place_cat."""

    board = _row_case()
    calls: list[tuple[int, int]] = []

    def tracking_place_cat(target: Board, row: int, column: int) -> bool:
        calls.append((row, column))
        return real_place_cat(target, row, column)

    monkeypatch.setattr(rule_module, "place_cat", tracking_place_cat)

    SingleRemainingLineCellRule().apply(board)

    assert calls == [(0, 2)]
    assert "block_cell" not in inspect.getsource(rule_module)


def test_rule_does_not_call_board_set_cat_directly() -> None:
    """Keep confirmed-cat writes encapsulated by the shared Cats action."""

    assert "board.set_cat(" not in inspect.getsource(rule_module)


def test_rule_does_not_assign_board_cells_directly() -> None:
    """Read the sole Board matrix without bypassing its mutation operations."""

    assert "board.cells[" not in inspect.getsource(rule_module)
