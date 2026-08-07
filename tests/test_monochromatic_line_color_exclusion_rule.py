"""Tests for excluding a monochromatic line's color outside that line."""

import inspect

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import MonochromaticLineColorExclusionRule
from logicforge.plugins.cats import (
    monochromatic_line_color_exclusion_rule as rule_module,
)
from logicforge.plugins.cats.board_actions import block_cell as real_block_cell
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
    """Capture a test-only immutable view for no-mutation assertions."""

    return tuple(tuple(row) for row in board.cells)


def _vertical_case() -> Board:
    """Return the documented board with a fully monochromatic first column."""

    return _board_from_values(
        (
            ("C0", "C0", "C0", "C1"),
            ("C0", "C2", "C3", "C4"),
            ("C0", "C5", "C6", "C7"),
            ("C0", "C8", "C9", "C10"),
        )
    )


def _horizontal_case() -> Board:
    """Return a board with a fully monochromatic first row."""

    return _board_from_values(
        (
            ("C0", "C0", "C0", "C0"),
            ("C0", "C2", "C5", "C8"),
            ("C0", "C3", "C6", "C9"),
            ("C1", "C4", "C7", "C10"),
        )
    )


def _single_remaining_line_case() -> Board:
    """Return the documented row whose only non-X possibility is C0."""

    return _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C3", "C0"),
            ("C2", "C3", "C1", "C0"),
        )
    )


def test_full_monochromatic_column_blocks_color_outside_column() -> None:
    """Block both C0 cells outside the documented first column."""

    board = _vertical_case()

    changed = MonochromaticLineColorExclusionRule().apply(board)

    assert changed is True
    assert board.get(0, 1) == "X"
    assert board.get(0, 2) == "X"


def test_full_monochromatic_row_blocks_color_outside_row() -> None:
    """Reserve row zero for C0 and exclude its other two candidates."""

    board = _horizontal_case()

    MonochromaticLineColorExclusionRule().apply(board)

    assert board.get(1, 0) == "X"
    assert board.get(2, 0) == "X"


def test_studied_color_inside_column_remains_unchanged() -> None:
    """Preserve every C0 that made the selected column monochromatic."""

    board = _vertical_case()

    MonochromaticLineColorExclusionRule().apply(board)

    assert tuple(board.get(row, 0) for row in range(4)) == ("C0",) * 4


def test_other_colors_outside_line_remain_unchanged() -> None:
    """Plan only exact C0 matches and preserve unrelated outside colors."""

    board = _vertical_case()

    MonochromaticLineColorExclusionRule().apply(board)

    assert board.get(1, 1) == "C2"
    assert board.get(3, 3) == "C10"


def test_existing_x_inside_line_is_ignored() -> None:
    """Treat X as an absent possibility rather than a second line value."""

    board = _single_remaining_line_case()

    changed = MonochromaticLineColorExclusionRule().apply(board)

    assert changed is True
    assert tuple(board.get(0, column) for column in (0, 1, 3)) == ("X",) * 3


def test_one_c0_among_x_values_qualifies_as_monochromatic_line() -> None:
    """Use row zero even though only one unresolved coordinate remains there."""

    board = _single_remaining_line_case()

    MonochromaticLineColorExclusionRule().apply(board)

    assert tuple(
        board.get(row, column) for row, column in ((1, 0), (2, 3), (3, 3))
    ) == (
        "X",
        "X",
        "X",
    )


def test_line_with_multiple_active_colors_does_not_qualify() -> None:
    """Reject rows and columns whose unresolved values are not monochromatic."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))

    assert MonochromaticLineColorExclusionRule().apply(board) is False


def test_line_containing_cat_is_skipped() -> None:
    """Do not infer a color from a line after K replaced its original ID."""

    board = _board_from_values(
        (
            ("K", "X", "C0", "X"),
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C3", "C0"),
            ("C2", "C3", "C1", "C0"),
        )
    )

    assert MonochromaticLineColorExclusionRule().apply(board) is False


def test_cat_line_causes_no_color_exclusion_outside_line() -> None:
    """Leave every outside C0 untouched when the apparent source row has K."""

    board = _board_from_values(
        (
            ("K", "X", "C0", "X"),
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C3", "C0"),
            ("C2", "C3", "C1", "C0"),
        )
    )

    MonochromaticLineColorExclusionRule().apply(board)

    assert tuple(
        board.get(row, column) for row, column in ((1, 0), (2, 3), (3, 3))
    ) == (
        "C0",
        "C0",
        "C0",
    )


def test_all_x_line_does_not_qualify() -> None:
    """Require at least one current C<n> possibility in a candidate line."""

    board = _board_from_values((("X", "X"), ("X", "X")))

    assert MonochromaticLineColorExclusionRule().apply(board) is False


def test_single_color_cell_with_other_x_values_qualifies() -> None:
    """Block the same color outside a one-C line."""

    board = _board_from_values(
        (("X", "C0", "X"), ("C1", "C2", "C3"), ("C2", "C3", "C0"))
    )

    assert MonochromaticLineColorExclusionRule().apply(board) is True
    assert board.get(2, 2) == "X"


def test_apply_returns_true_after_real_change() -> None:
    """Report the actual C<n>-to-X mutation performed by the action."""

    assert MonochromaticLineColorExclusionRule().apply(_vertical_case()) is True


def test_apply_returns_false_without_useful_line() -> None:
    """Return False after checking every mixed row and column."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))

    assert MonochromaticLineColorExclusionRule().apply(board) is False


def test_false_result_leaves_complete_board_unchanged() -> None:
    """Perform no mutation when candidate analysis finds no useful plan."""

    board = _board_from_values((("C0", "C1"), ("C1", "C0")))
    expected = _values(board)

    MonochromaticLineColorExclusionRule().apply(board)

    assert _values(board) == expected


def test_candidate_without_targets_does_not_stop_later_useful_line() -> None:
    """Skip satisfied C0 and continue to a useful higher-priority C1 row."""

    board = _board_from_values(
        (
            ("C0", "C0", "X", "X"),
            ("C1", "C1", "X", "X"),
            ("C1", "C2", "C3", "C4"),
        )
    )

    changed = MonochromaticLineColorExclusionRule().apply(board)

    assert changed is True
    assert board.get(2, 0) == "X"


def test_one_apply_handles_at_most_one_line() -> None:
    """Apply the C0 row plan while leaving a separately useful C1 plan alone."""

    board = _board_from_values(
        (
            ("C0", "C0", "X", "X"),
            ("C1", "C1", "X", "X"),
            ("C0", "C2", "C3", "C4"),
            ("C1", "C5", "C6", "C7"),
        )
    )

    MonochromaticLineColorExclusionRule().apply(board)

    assert board.get(2, 0) == "X"
    assert board.get(3, 0) == "C1"


def test_one_apply_can_block_multiple_cells() -> None:
    """Apply every same-color outside target from one selected line plan."""

    board = _vertical_case()

    MonochromaticLineColorExclusionRule().apply(board)

    assert (board.get(0, 1), board.get(0, 2)) == ("X", "X")


def test_c2_is_handled_before_c10() -> None:
    """Sort numeric suffixes instead of selecting C10 lexicographically."""

    board = _board_from_values(
        (
            ("C2", "C2", "X"),
            ("C2", "C3", "C4"),
            ("C10", "C10", "X"),
            ("C10", "C11", "C12"),
        )
    )

    MonochromaticLineColorExclusionRule().apply(board)

    assert board.get(1, 0) == "X"
    assert board.get(3, 0) == "C10"


def test_row_is_handled_before_column_for_same_color() -> None:
    """Prefer the C2 row when both axes form useful monochromatic lines."""

    board = _board_from_values(
        (("C2", "C2", "C2"), ("C2", "C3", "C4"), ("C2", "C5", "C6"))
    )

    MonochromaticLineColorExclusionRule().apply(board)

    assert tuple(board.get(0, column) for column in range(3)) == ("C2",) * 3
    assert (board.get(1, 0), board.get(2, 0)) == ("X", "X")


def test_lower_line_index_is_handled_first() -> None:
    """Choose row zero before the other useful C2 row at index two."""

    board = _board_from_values(
        (("C2", "C2", "X"), ("C3", "C4", "C5"), ("C2", "C2", "X"))
    )

    MonochromaticLineColorExclusionRule().apply(board)

    assert (board.get(0, 0), board.get(0, 1)) == ("C2", "C2")
    assert (board.get(2, 0), board.get(2, 1)) == ("X", "X")


def test_existing_x_outside_line_is_not_counted_as_change() -> None:
    """Do not rewrite X or report success without an exact unresolved target."""

    board = _board_from_values((("C0", "X"), ("X", "X")))

    changed = MonochromaticLineColorExclusionRule().apply(board)

    assert changed is False
    assert tuple(board.get(1, column) for column in range(2)) == ("X", "X")


def test_invalid_line_value_raises_board_state_error_with_details() -> None:
    """Report both coordinates and unsupported value during line analysis."""

    board = _board_from_values((("C0", "C0", "X"), ("C0", "C1", "INVALID")))

    with pytest.raises(BoardStateError, match=r"INVALID.*\(1, 2\)"):
        MonochromaticLineColorExclusionRule().apply(board)


def test_board_state_error_leaves_board_without_partial_changes() -> None:
    """Finish candidate validation before mutating an earlier useful plan."""

    board = _board_from_values((("C0", "C0", "X"), ("C0", "C1", "INVALID")))
    expected = _values(board)

    with pytest.raises(BoardStateError):
        MonochromaticLineColorExclusionRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_board() -> None:
    """Apply a monochromatic column deduction on three-by-five geometry."""

    board = _board_from_values(
        (
            ("C0", "C0", "C1", "C2", "C3"),
            ("C0", "C4", "C5", "C6", "C7"),
            ("C0", "C8", "C9", "C10", "C11"),
        )
    )

    changed = MonochromaticLineColorExclusionRule().apply(board)

    assert changed is True
    assert board.get(0, 1) == "X"
    assert tuple(board.get(row, 0) for row in range(3)) == ("C0",) * 3


def test_rule_uses_only_block_cell_for_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observe every planned write passing through the shared Cats action."""

    board = _vertical_case()
    calls: list[tuple[int, int]] = []

    def tracking_block_cell(target: Board, row: int, column: int) -> bool:
        calls.append((row, column))
        return real_block_cell(target, row, column)

    monkeypatch.setattr(rule_module, "block_cell", tracking_block_cell)

    MonochromaticLineColorExclusionRule().apply(board)

    assert calls == [(0, 1), (0, 2)]
    assert "place_cat" not in inspect.getsource(rule_module)


def test_rule_does_not_write_board_cells_or_call_set_blocked_directly() -> None:
    """Keep Board writes encapsulated by block_cell rather than its internals."""

    source = inspect.getsource(rule_module)

    assert "board.set_blocked(" not in source
    assert "board.cells[" not in source
