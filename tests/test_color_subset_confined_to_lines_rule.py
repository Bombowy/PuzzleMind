"""Tests for the Cats N-colors-in-N-lines subset deduction."""

import inspect

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import ColorSubsetConfinedToLinesRule
from logicforge.plugins.cats import color_subset_confined_to_lines_rule as rule_module
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


def _two_colors_in_two_columns() -> Board:
    """Return the documented C0/C1 subset reserving columns zero and one."""

    return _board_from_values(
        (
            ("C0", "C0", "C2", "C3"),
            ("C0", "C4", "C2", "C3"),
            ("C1", "C1", "C4", "C5"),
            ("C1", "C5", "C6", "C6"),
        )
    )


def _two_colors_in_two_rows_with_x() -> Board:
    """Return the documented C2/C3 row subset with existing X evidence."""

    return _board_from_values(
        (
            ("X", "C2", "C3", "C8"),
            ("X", "C2", "C3", "C9"),
            ("C8", "C9", "C4", "C5"),
            ("C4", "C5", "C6", "C7"),
        )
    )


def _three_colors_in_three_rows() -> Board:
    """Return C0/C1/C2 spanning three rows with C3 row targets."""

    return _board_from_values(
        (
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C0", "C3"),
            ("C2", "C0", "C1", "C3"),
            ("C4", "C4", "C4", "C3"),
        )
    )


def _three_colors_in_three_columns() -> Board:
    """Return C0/C1/C2 spanning four rows but exactly three columns."""

    return _board_from_values(
        (
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C0", "C3"),
            ("C2", "C0", "C1", "C3"),
            ("C3", "C1", "C2", "C3"),
        )
    )


def _latin_three() -> Board:
    """Return three colors whose every pair spans three rows and columns."""

    return _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C2", "C0"),
            ("C2", "C0", "C1"),
        )
    )


def _same_subset_on_rows_and_columns() -> Board:
    """Return C0/C1 reserving both two rows and two columns with distinct targets."""

    return _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C0", "C3"),
            ("C4", "C5", "C6"),
        )
    )


def test_two_colors_in_two_columns_block_other_colors_in_columns() -> None:
    """Block C4 and C5 from columns reserved by C0 and C1."""

    board = _two_colors_in_two_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(1, 1), board.get(3, 1)) == ("X", "X")


def test_selected_color_cells_remain_unresolved() -> None:
    """Preserve every C0 and C1 candidate in the reserved columns."""

    board = _two_colors_in_two_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert tuple(
        board.get(row, column)
        for row, column in ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (3, 0))
    ) == ("C0", "C0", "C0", "C1", "C1", "C1")


def test_other_colors_outside_reserved_columns_remain_unchanged() -> None:
    """Limit mutations to the union of reserved lines."""

    board = _two_colors_in_two_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert tuple(
        board.get(row, column) for row, column in ((0, 2), (1, 2), (2, 3), (3, 3))
    ) == ("C2", "C2", "C5", "C6")


def test_two_colors_in_two_rows_block_other_colors_in_rows() -> None:
    """Reserve rows zero and one for C2/C3 and block their outside colors."""

    board = _two_colors_in_two_rows_with_x()

    changed = ColorSubsetConfinedToLinesRule().apply(board)

    assert changed is True
    assert (board.get(0, 3), board.get(1, 3)) == ("X", "X")


def test_existing_x_are_ignored_when_identifying_subset_lines() -> None:
    """Do not include X in colors or row-index evidence."""

    board = _two_colors_in_two_rows_with_x()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 0), board.get(1, 0)) == ("X", "X")
    assert tuple(
        board.get(row, column) for row, column in ((0, 1), (0, 2), (1, 1), (1, 2))
    ) == (
        "C2",
        "C3",
        "C2",
        "C3",
    )


def test_documented_existing_x_example_blocks_c8_and_c9() -> None:
    """Produce the exact two exclusions from the documented X-heavy board."""

    board = _two_colors_in_two_rows_with_x()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert board.get(0, 3) == "X"
    assert board.get(1, 3) == "X"


def test_existing_x_are_not_counted_as_real_change() -> None:
    """Return False when a qualifying subset has only selected colors and X."""

    board = _board_from_values(
        (
            ("C0", "C1", "X"),
            ("C1", "C0", "X"),
            ("X", "X", "C2"),
        )
    )
    expected = _values(board)

    changed = ColorSubsetConfinedToLinesRule().apply(board)

    assert changed is False
    assert _values(board) == expected


def test_two_color_subset_spanning_three_rows_does_not_qualify() -> None:
    """Reject row confinement when the union has more lines than colors."""

    assert ColorSubsetConfinedToLinesRule().apply(_latin_three()) is False


def test_two_color_subset_spanning_three_columns_does_not_qualify() -> None:
    """Reject column confinement when the union has more lines than colors."""

    assert ColorSubsetConfinedToLinesRule().apply(_latin_three()) is False


def test_candidate_outside_two_lines_prevents_subset_qualification() -> None:
    """Include every current candidate when collecting the subset's line union."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C2", "C0"),
            ("C0", "C1", "C2"),
        )
    )

    assert ColorSubsetConfinedToLinesRule().apply(board) is False


def test_three_colors_in_three_rows_block_other_colors() -> None:
    """Support N=3 by excluding C3 from all three reserved rows."""

    board = _three_colors_in_three_rows()

    changed = ColorSubsetConfinedToLinesRule().apply(board)

    assert changed is True
    assert tuple(board.get(row, 3) for row in range(3)) == ("X", "X", "X")


def test_three_colors_in_three_columns_block_other_colors() -> None:
    """Support N=3 column unions without requiring a rectangular candidate block."""

    board = _three_colors_in_three_columns()

    changed = ColorSubsetConfinedToLinesRule().apply(board)

    assert changed is True
    assert board.get(3, 0) == "X"


def test_single_color_in_single_line_is_not_handled() -> None:
    """Leave N=1 to ColorConfinedToLineRule rather than duplicating it here."""

    board = _board_from_values(
        (
            ("C0", "C0", "C1"),
            ("C1", "C2", "C2"),
            ("C1", "C2", "C1"),
        )
    )

    assert ColorSubsetConfinedToLinesRule().apply(board) is False


def test_no_useful_subset_returns_false() -> None:
    """Return False after exhausting all equality-sized line unions."""

    assert ColorSubsetConfinedToLinesRule().apply(_latin_three()) is False


def test_false_result_leaves_board_unchanged() -> None:
    """Perform no mutation when no subset supplies an exclusion."""

    board = _latin_three()
    expected = _values(board)

    ColorSubsetConfinedToLinesRule().apply(board)

    assert _values(board) == expected


def test_apply_returns_true_after_real_exclusion() -> None:
    """Report at least one actual C<n>-to-X transition."""

    assert ColorSubsetConfinedToLinesRule().apply(_two_colors_in_two_columns()) is True


def test_one_apply_handles_one_subset_and_one_axis() -> None:
    """Apply row targets and return before the same subset's column targets."""

    board = _same_subset_on_rows_and_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 2), board.get(1, 2)) == ("X", "X")
    assert (board.get(2, 0), board.get(2, 1)) == ("C4", "C5")


def test_one_apply_can_block_multiple_cells() -> None:
    """Apply every validated target belonging to the selected axis."""

    board = _two_colors_in_two_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert sum(value == "X" for row in board.cells for value in row) == 2


def test_subset_size_two_is_handled_before_size_three() -> None:
    """Prefer a useful C0/C1 pair over a separately useful C2/C3/C4 triple."""

    board = _board_from_values(
        (
            ("C0", "C1", "C6", "X", "X"),
            ("C1", "C0", "C7", "X", "X"),
            ("X", "X", "X", "X", "X"),
            ("C2", "C3", "C4", "C8", "X"),
            ("C3", "C4", "C2", "C8", "X"),
            ("C4", "C2", "C3", "C8", "X"),
        )
    )

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 2), board.get(1, 2)) == ("X", "X")
    assert tuple(board.get(row, 3) for row in range(3, 6)) == ("C8",) * 3


def test_lower_numbered_color_combination_is_handled_first() -> None:
    """Choose the useful C0/C1 pair before a useful C2/C3 pair."""

    board = _board_from_values(
        (
            ("C0", "C1", "C8", "X"),
            ("C1", "C0", "C9", "X"),
            ("X", "X", "X", "X"),
            ("C2", "C3", "C10", "X"),
            ("C3", "C2", "C11", "X"),
        )
    )

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 2), board.get(1, 2)) == ("X", "X")
    assert (board.get(3, 2), board.get(4, 2)) == ("C10", "C11")


def test_c2_combination_is_handled_before_c10_combination() -> None:
    """Sort logical identifiers by numeric suffix rather than lexicographically."""

    board = _board_from_values(
        (
            ("C2", "C3", "C20", "X"),
            ("C3", "C2", "C21", "X"),
            ("X", "X", "X", "X"),
            ("C10", "C11", "C22", "X"),
            ("C11", "C10", "C23", "X"),
        )
    )

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 2), board.get(1, 2)) == ("X", "X")
    assert (board.get(3, 2), board.get(4, 2)) == ("C22", "C23")


def test_rows_are_analyzed_before_columns_for_same_subset() -> None:
    """Use row targets first when both axes contain exactly N lines."""

    board = _same_subset_on_rows_and_columns()

    ColorSubsetConfinedToLinesRule().apply(board)

    assert (board.get(0, 2), board.get(1, 2)) == ("X", "X")
    assert (board.get(2, 0), board.get(2, 1)) == ("C4", "C5")


def test_satisfied_first_subset_continues_to_next_useful_subset() -> None:
    """Skip C0/C1 plans containing only selected colors and existing X."""

    board = _board_from_values(
        (
            ("C0", "C1", "X", "X", "X"),
            ("C1", "C0", "X", "X", "X"),
            ("X", "X", "C2", "C3", "C4"),
            ("X", "X", "C3", "C2", "C5"),
        )
    )

    changed = ColorSubsetConfinedToLinesRule().apply(board)

    assert changed is True
    assert (board.get(2, 4), board.get(3, 4)) == ("X", "X")


def test_cat_in_reserved_row_raises_board_state_error() -> None:
    """Reject a row union needed by unresolved selected colors when it contains K."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C0", "K"),
            ("C3", "C4", "C5"),
        )
    )

    with pytest.raises(BoardStateError, match=r"existing cat.*\(1, 2\)"):
        ColorSubsetConfinedToLinesRule().apply(board)


def test_cat_in_reserved_column_raises_board_state_error() -> None:
    """Reject a column union containing K after row confinement does not qualify."""

    board = _board_from_values(
        (
            ("C0", "C2", "C3"),
            ("C1", "C4", "C5"),
            ("C0", "C1", "C6"),
            ("K", "C1", "C7"),
        )
    )

    with pytest.raises(BoardStateError, match=r"existing cat.*\(3, 0\)"):
        ColorSubsetConfinedToLinesRule().apply(board)


def test_cat_conflict_leaves_board_without_partial_changes() -> None:
    """Validate every reserved cell before applying an earlier planned target."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C0", "K"),
            ("C3", "C4", "C5"),
        )
    )
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ColorSubsetConfinedToLinesRule().apply(board)

    assert _values(board) == expected


def test_invalid_board_value_raises_with_coordinates() -> None:
    """Report unsupported state before generating any color combinations."""

    board = _two_colors_in_two_columns()
    board.cells[3][3] = "INVALID"

    with pytest.raises(BoardStateError, match=r"INVALID.*\(3, 3\)"):
        ColorSubsetConfinedToLinesRule().apply(board)


def test_invalid_board_value_leaves_board_without_partial_changes() -> None:
    """Validate the complete Board before a useful subset can mutate targets."""

    board = _two_colors_in_two_columns()
    board.cells[3][3] = "INVALID"
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ColorSubsetConfinedToLinesRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_three_by_five_board() -> None:
    """Apply a two-row reservation on wide rectangular geometry."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2", "C3", "C4"),
            ("C1", "C0", "C5", "C6", "C7"),
            ("C8", "C9", "C10", "C11", "C12"),
        )
    )

    assert ColorSubsetConfinedToLinesRule().apply(board) is True
    assert (
        tuple(board.get(row, column) for row in range(2) for column in range(2, 5))
        == ("X",) * 6
    )


def test_rule_supports_rectangular_five_by_three_board() -> None:
    """Apply a two-column reservation on tall rectangular geometry."""

    board = _board_from_values(
        (
            ("C0", "C2", "C3"),
            ("C1", "C4", "C5"),
            ("C0", "C1", "C6"),
            ("C1", "C7", "C8"),
            ("C0", "C9", "C10"),
        )
    )

    assert ColorSubsetConfinedToLinesRule().apply(board) is True
    assert tuple(board.get(row, 1) for row in (0, 1, 3, 4)) == ("X",) * 4


def test_all_mutations_use_block_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observe every target flowing through the existing Cats block action."""

    board = _two_colors_in_two_columns()
    calls: list[tuple[int, int]] = []

    def tracking_block_cell(target: Board, row: int, column: int) -> bool:
        calls.append((row, column))
        return real_block_cell(target, row, column)

    monkeypatch.setattr(rule_module, "block_cell", tracking_block_cell)

    ColorSubsetConfinedToLinesRule().apply(board)

    assert calls == [(1, 1), (3, 1)]


def test_rule_does_not_use_place_cat() -> None:
    """Avoid simulating which selected color occupies which reserved line."""

    assert "place_cat" not in inspect.getsource(rule_module)


def test_rule_does_not_call_board_set_blocked_directly() -> None:
    """Keep all final-state writes encapsulated by block_cell."""

    assert "board.set_blocked(" not in inspect.getsource(rule_module)


def test_rule_does_not_assign_board_cells_directly() -> None:
    """Read the sole mutable matrix without bypassing Board mutation safety."""

    assert "board.cells[" not in inspect.getsource(rule_module)
