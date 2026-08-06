"""Deterministic tests for atomic Cats-specific board actions."""

from collections.abc import Callable

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import block_cell, place_cat
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)

INITIAL_MATRIX = (
    ("C0", "C1", "C2", "C3"),
    ("C4", "C0", "C5", "C6"),
    ("C7", "C2", "C3", "C1"),
    ("C4", "C5", "C6", "C7"),
)


def _board() -> Board:
    """Build the documented 4x4 Cats example through the public vision contract."""

    centers = tuple((40.0 + index * 20.0, 120.0, 130.0) for index in range(8))
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=color_id,
            confidence=0.95,
            representative_lab=centers[int(color_id[1:])],
        )
        for row, values in enumerate(INITIAL_MATRIX)
        for column, color_id in enumerate(values)
    )
    result = ColorDetectionResult(
        observations=observations,
        color_count=8,
        color_matrix=INITIAL_MATRIX,
        mean_confidence=0.95,
        diagnostics=ColorDetectionDiagnostics(
            rows=4,
            columns=4,
            sample_inner_fraction=0.65,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * 16,
            within_cell_spreads=(1.0,) * 16,
            cluster_centers_lab=centers,
            minimum_intercluster_distance=20.0,
        ),
    )
    return Board(result)


def _matrix_values(board: Board) -> tuple[tuple[str, ...], ...]:
    """Freeze values only inside tests to assert all-or-nothing behavior."""

    return tuple(tuple(row) for row in board.cells)


def test_place_cat_changes_selected_color_to_cat() -> None:
    """Apply the requested C0 to K transition through the Board API."""

    board = _board()

    place_cat(board, 1, 1)

    assert board.get(1, 1) == "K"
    assert board.is_cat(1, 1)


def test_place_cat_returns_true_after_real_change() -> None:
    """Report that an unresolved field and its direct exclusions were changed."""

    board = _board()

    assert place_cat(board, 1, 1) is True


def test_repeated_place_cat_is_idempotent_and_returns_false() -> None:
    """Avoid re-running propagation when the selected cell is already K."""

    board = _board()
    place_cat(board, 1, 1)
    expected = _matrix_values(board)

    changed = place_cat(board, 1, 1)

    assert changed is False
    assert _matrix_values(board) == expected


def test_place_cat_blocks_every_other_cell_of_the_same_color() -> None:
    """Use the color ID captured before C0 is replaced with K."""

    board = _board()

    place_cat(board, 1, 1)

    assert board.get(0, 0) == "X"


def test_place_cat_blocks_every_other_unknown_in_the_same_row() -> None:
    """Apply the direct one-cat-per-row exclusion to all remaining row cells."""

    board = _board()

    place_cat(board, 1, 1)

    assert tuple(board.get(1, column) for column in range(4)) == (
        "X",
        "K",
        "X",
        "X",
    )


def test_place_cat_blocks_every_other_unknown_in_the_same_column() -> None:
    """Apply the direct one-cat-per-column exclusion to all remaining cells."""

    board = _board()

    place_cat(board, 1, 1)

    assert tuple(board.get(row, 1) for row in range(4)) == (
        "X",
        "K",
        "X",
        "X",
    )


def test_place_cat_blocks_all_eight_existing_neighbors() -> None:
    """Block orthogonal and diagonal neighbors around an interior cat."""

    board = _board()
    neighbors = (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    )

    place_cat(board, 1, 1)

    assert all(board.is_blocked(row, column) for row, column in neighbors)


def test_place_cat_in_corner_skips_out_of_bounds_neighbors() -> None:
    """Clip neighbor planning to the board instead of relying on negative indexes."""

    board = _board()

    assert place_cat(board, 0, 0)

    assert board.is_cat(0, 0)
    assert board.is_blocked(0, 1)
    assert board.is_blocked(1, 0)
    assert board.is_blocked(1, 1)


def test_place_cat_preserves_cells_unrelated_to_all_four_exclusions() -> None:
    """Leave cells outside color, row, column, and neighborhood unchanged."""

    board = _board()
    unrelated = {
        (0, 3): "C3",
        (2, 3): "C1",
        (3, 0): "C4",
        (3, 2): "C6",
        (3, 3): "C7",
    }

    place_cat(board, 1, 1)

    assert {
        coordinates: board.get(*coordinates) for coordinates in unrelated
    } == unrelated


def test_existing_blocked_cells_remain_blocked_during_propagation() -> None:
    """Treat pre-existing X values as satisfied exclusions rather than mutations."""

    board = _board()
    board.set_blocked(1, 0)

    place_cat(board, 1, 1)

    assert board.is_blocked(1, 0)


def test_place_cat_on_blocked_cell_raises_and_preserves_board() -> None:
    """Reject X to K before planning or applying any additional exclusion."""

    board = _board()
    board.set_blocked(1, 1)
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="already blocked"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_place_cat_rejects_invalid_target_without_mutation() -> None:
    """Fail before planning when direct list access corrupted the selected value."""

    board = _board()
    board.cells[1][1] = "INVALID"
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="invalid board value"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_existing_cat_in_same_row_rejects_complete_plan_atomically() -> None:
    """Detect a row conflict before setting the requested cat or any X values."""

    board = _board()
    board.set_cat(1, 3)
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="existing cat"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_existing_cat_in_same_column_rejects_complete_plan_atomically() -> None:
    """Detect a column conflict while preserving every original matrix value."""

    board = _board()
    board.set_cat(3, 1)
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="existing cat"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_existing_cat_of_same_color_prevents_second_cat_atomically() -> None:
    """Use prior color propagation to preserve one-cat-per-color invariants."""

    board = _board()
    place_cat(board, 0, 0)
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="already blocked"):
        place_cat(board, 1, 1)

    assert board.is_cat(0, 0)
    assert _matrix_values(board) == expected


def test_existing_diagonal_cat_rejects_complete_plan_atomically() -> None:
    """Detect an eight-neighbor conflict that is not a row or column conflict."""

    board = _board()
    board.set_cat(0, 2)
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="existing cat"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_invalid_value_in_exclusion_plan_rejects_before_any_mutation() -> None:
    """Validate every planned coordinate before setting the requested K."""

    board = _board()
    board.cells[1][3] = "INVALID"
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError, match="invalid board value"):
        place_cat(board, 1, 1)

    assert _matrix_values(board) == expected


def test_block_cell_changes_unknown_to_x_and_returns_true() -> None:
    """Provide future rules one non-propagating C<n> to X operation."""

    board = _board()

    changed = block_cell(board, 2, 3)

    assert changed is True
    assert board.is_blocked(2, 3)


def test_block_cell_on_x_is_idempotent_and_returns_false() -> None:
    """Report no change when a future rule repeats the same exclusion."""

    board = _board()
    block_cell(board, 2, 3)

    changed = block_cell(board, 2, 3)

    assert changed is False
    assert board.is_blocked(2, 3)


def test_block_cell_on_cat_raises_and_preserves_cat() -> None:
    """Keep K terminal when a later rule proposes an incompatible X."""

    board = _board()
    board.set_cat(2, 3)

    with pytest.raises(BoardStateError, match="confirmed cat"):
        block_cell(board, 2, 3)

    assert board.is_cat(2, 3)


def test_block_cell_rejects_invalid_value_without_mutation() -> None:
    """Reject unsupported state rather than treating it as unresolved."""

    board = _board()
    board.cells[2][3] = "INVALID"

    with pytest.raises(BoardStateError, match="invalid board value"):
        block_cell(board, 2, 3)

    assert board.get(2, 3) == "INVALID"


@pytest.mark.parametrize("action", (place_cat, block_cell))
def test_board_actions_preserve_index_errors_for_invalid_coordinates(
    action: Callable[[Board, int, int], bool],
) -> None:
    """Delegate coordinate validation to Board without translating IndexError."""

    board = _board()

    with pytest.raises(IndexError):
        action(board, -1, 0)
    with pytest.raises(IndexError):
        action(board, 0, 4)
