"""Tests for the Cats adjacent-color-pair perpendicular exclusion."""

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import AdjacentColorPairExclusionRule
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
    """Capture a test-only value view for all-or-nothing assertions."""

    return tuple(tuple(row) for row in board.cells)


def _vertical_pair_board() -> Board:
    """Return an interior vertical C1 pair with four mutable side targets."""

    return _board_from_values(
        (
            ("C0", "C1", "C2", "C3"),
            ("C4", "C1", "C5", "C6"),
            ("C7", "C8", "C9", "C10"),
        )
    )


def _horizontal_pair_board() -> Board:
    """Return an interior horizontal C5 pair with four mutable vertical targets."""

    return _board_from_values(
        (
            ("C0", "C1", "C2", "C3"),
            ("C4", "C5", "C5", "C6"),
            ("C7", "C8", "C9", "C10"),
        )
    )


def test_vertical_pair_blocks_left_and_right_of_both_cells() -> None:
    """Block all four perpendicular neighbors of an interior vertical pair."""

    board = _vertical_pair_board()

    AdjacentColorPairExclusionRule().apply(board)

    assert tuple(
        board.get(row, column) for row, column in ((0, 0), (0, 2), (1, 0), (1, 2))
    ) == (
        "X",
        "X",
        "X",
        "X",
    )


def test_horizontal_pair_blocks_above_and_below_both_cells() -> None:
    """Block all four perpendicular neighbors of an interior horizontal pair."""

    board = _horizontal_pair_board()

    AdjacentColorPairExclusionRule().apply(board)

    assert tuple(
        board.get(row, column) for row, column in ((0, 1), (0, 2), (2, 1), (2, 2))
    ) == (
        "X",
        "X",
        "X",
        "X",
    )


def test_studied_pair_remains_unresolved() -> None:
    """Preserve both C1 cells because the rule does not know which is the cat."""

    board = _vertical_pair_board()

    AdjacentColorPairExclusionRule().apply(board)

    assert board.get(0, 1) == "C1"
    assert board.get(1, 1) == "C1"


def test_apply_returns_true_after_real_exclusion() -> None:
    """Report True only after at least one C<n> to X transition."""

    assert AdjacentColorPairExclusionRule().apply(_vertical_pair_board()) is True


def test_interior_pair_can_block_four_cells() -> None:
    """Apply all four validated target mutations in one invocation."""

    board = _vertical_pair_board()

    AdjacentColorPairExclusionRule().apply(board)

    assert sum(value == "X" for row in board.cells for value in row) == 4


@pytest.mark.parametrize(
    ("values", "expected_blocked"),
    (
        ((("C1", "C2"), ("C1", "C3"), ("C4", "C5")), ((0, 1), (1, 1))),
        ((("C2", "C1"), ("C3", "C1"), ("C4", "C5")), ((0, 0), (1, 0))),
    ),
)
def test_vertical_pair_at_side_edge_clips_out_of_bounds_targets(
    values: tuple[tuple[str, ...], ...],
    expected_blocked: tuple[tuple[int, int], ...],
) -> None:
    """Clip missing left or right neighbors while applying existing targets."""

    board = _board_from_values(values)

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is True
    assert all(board.is_blocked(row, column) for row, column in expected_blocked)


@pytest.mark.parametrize(
    ("values", "expected_blocked"),
    (
        ((("C0", "C1", "C1"), ("C2", "C3", "C4")), ((1, 1), (1, 2))),
        ((("C2", "C3", "C4"), ("C0", "C1", "C1")), ((0, 1), (0, 2))),
    ),
)
def test_horizontal_pair_at_top_or_bottom_clips_out_of_bounds_targets(
    values: tuple[tuple[str, ...], ...],
    expected_blocked: tuple[tuple[int, int], ...],
) -> None:
    """Clip missing upper or lower neighbors while applying existing targets."""

    board = _board_from_values(values)

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is True
    assert all(board.is_blocked(row, column) for row, column in expected_blocked)


def test_singleton_color_is_skipped() -> None:
    """Ignore colors with fewer than exactly two current candidates."""

    board = _board_from_values((("C0", "X"), ("X", "X")))

    assert AdjacentColorPairExclusionRule().apply(board) is False


def test_color_with_three_candidates_is_skipped() -> None:
    """Ignore a color that still has more than two unresolved cells."""

    board = _board_from_values((("C0", "C0"), ("C0", "X")))

    assert AdjacentColorPairExclusionRule().apply(board) is False


def test_pair_separated_by_one_cell_causes_no_change() -> None:
    """Reject a Manhattan distance of two even along the same row."""

    board = _board_from_values((("C0", "C1", "C0"),))

    assert AdjacentColorPairExclusionRule().apply(board) is False


def test_diagonal_pair_causes_no_change() -> None:
    """Reject diagonally touching candidates because they do not share an edge."""

    board = _board_from_values((("C0", "C1"), ("C2", "C0")))

    assert AdjacentColorPairExclusionRule().apply(board) is False


def test_no_matching_pair_returns_false() -> None:
    """Return False when no color has exactly two edge-adjacent candidates."""

    board = _board_from_values((("C0", "C1", "C0"), ("C2", "C3", "C4")))

    assert AdjacentColorPairExclusionRule().apply(board) is False


def test_false_result_leaves_complete_board_unchanged() -> None:
    """Perform no write when all candidate pairs fail the shape requirement."""

    board = _board_from_values((("C0", "C1"), ("C2", "C0")))
    expected = _values(board)

    AdjacentColorPairExclusionRule().apply(board)

    assert _values(board) == expected


def test_k_and_x_are_not_grouped_as_color_candidates() -> None:
    """Do not interpret finalized values as members of a logical color class."""

    board = _board_from_values((("C0", "K"), ("X", "C1")))
    expected = _values(board)

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is False
    assert _values(board) == expected


def test_existing_x_target_remains_x() -> None:
    """Accept an already blocked target without rewriting it."""

    board = _board_from_values(
        (("X", "C1", "C2"), ("C3", "C1", "C4"), ("C5", "C6", "C7"))
    )

    AdjacentColorPairExclusionRule().apply(board)

    assert board.get(0, 0) == "X"


def test_partial_existing_x_still_blocks_remaining_targets() -> None:
    """Apply unresolved targets even when another target already satisfies the plan."""

    board = _board_from_values(
        (("X", "C1", "C2"), ("C3", "C1", "X"), ("C4", "C5", "C6"))
    )

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is True
    assert board.get(0, 2) == "X"
    assert board.get(1, 0) == "X"


def test_lowest_numeric_color_pair_is_selected() -> None:
    """Handle the lowest numerical color and leave a later pair untouched."""

    board = _board_from_values(
        (
            ("C0", "C2", "C3", "C4", "C5"),
            ("C6", "C2", "C7", "C8", "C9"),
            ("C11", "C12", "C13", "C14", "C15"),
            ("C16", "C17", "C18", "C3", "C19"),
            ("C20", "C21", "C22", "C3", "C23"),
        )
    )

    AdjacentColorPairExclusionRule().apply(board)

    assert board.get(0, 0) == "X"
    assert board.get(3, 2) == "C18"
    assert board.get(3, 4) == "C19"


def test_c2_is_selected_before_c10() -> None:
    """Sort color suffixes numerically instead of comparing identifiers as text."""

    board = _board_from_values(
        (
            ("C0", "C2", "C3", "C4", "C5"),
            ("C6", "C2", "C7", "C8", "C9"),
            ("C11", "C12", "C13", "C14", "C15"),
            ("C16", "C17", "C18", "C10", "C19"),
            ("C20", "C21", "C22", "C10", "C23"),
        )
    )

    AdjacentColorPairExclusionRule().apply(board)

    assert board.get(0, 0) == "X"
    assert board.get(3, 2) == "C18"
    assert board.get(3, 4) == "C19"


def test_one_apply_handles_at_most_one_color() -> None:
    """Return after applying the first pair even when another pair also qualifies."""

    board = _board_from_values(
        (
            ("C0", "C2", "C3", "C4", "C5"),
            ("C6", "C2", "C7", "C8", "C9"),
            ("C11", "C12", "C13", "C14", "C15"),
            ("C16", "C17", "C18", "C10", "C19"),
            ("C20", "C21", "C22", "C10", "C23"),
        )
    )

    AdjacentColorPairExclusionRule().apply(board)

    assert board.get(0, 0) == "X"
    assert board.get(0, 2) == "X"
    assert board.get(1, 0) == "X"
    assert board.get(1, 2) == "X"
    assert board.get(3, 2) == "C18"
    assert board.get(3, 4) == "C19"
    assert board.get(4, 2) == "C22"
    assert board.get(4, 4) == "C23"


def test_satisfied_first_pair_continues_to_next_color() -> None:
    """Skip a pair with only X targets and apply the next useful pair."""

    board = _board_from_values(
        (
            ("C0", "X", "C2", "C3"),
            ("C0", "X", "C2", "C4"),
            ("C5", "C6", "C7", "C8"),
        )
    )

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is True
    assert board.get(0, 3) == "X"
    assert board.get(1, 3) == "X"


def test_cat_in_target_raises_board_state_error() -> None:
    """Reject a perpendicular plan containing an existing cat."""

    board = _board_from_values(
        (("C2", "C1", "C3"), ("C4", "C1", "K"), ("C5", "C6", "C7"))
    )

    with pytest.raises(BoardStateError, match="existing cat"):
        AdjacentColorPairExclusionRule().apply(board)


def test_cat_conflict_leaves_complete_board_unchanged() -> None:
    """Validate every target before performing the first block_cell write."""

    board = _board_from_values(
        (("C2", "C1", "C3"), ("C4", "C1", "K"), ("C5", "C6", "C7"))
    )
    expected = _values(board)

    with pytest.raises(BoardStateError):
        AdjacentColorPairExclusionRule().apply(board)

    assert _values(board) == expected


def test_invalid_target_value_raises_without_partial_mutation() -> None:
    """Fail atomically when a later target contains an unsupported board value."""

    board = _board_from_values(
        (
            ("C2", "C1", "C3"),
            ("C4", "C1", "INVALID"),
            ("C5", "C6", "C7"),
        )
    )
    expected = _values(board)

    with pytest.raises(BoardStateError, match="invalid target value"):
        AdjacentColorPairExclusionRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_board() -> None:
    """Apply an interior horizontal pair on non-square board geometry."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2", "C3", "C4"),
            ("C5", "C6", "C6", "C7", "C8"),
            ("C9", "C10", "C11", "C12", "C13"),
        )
    )

    changed = AdjacentColorPairExclusionRule().apply(board)

    assert changed is True
    assert tuple(
        board.get(row, column) for row, column in ((0, 1), (0, 2), (2, 1), (2, 2))
    ) == (
        "X",
        "X",
        "X",
        "X",
    )
