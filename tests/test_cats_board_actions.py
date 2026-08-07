"""Deterministic tests for atomic Cats-specific board actions."""

from collections.abc import Callable

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import block_cell, place_cat
from logicforge.plugins.cats import board_actions as board_actions_module
from logicforge.plugins.cats.board_actions import collect_cat_exclusion_coordinates
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


def _rectangular_board() -> Board:
    """Build a 3x5 board for geometry-independent action-plan tests."""

    values = (
        ("C0", "C1", "C2", "C3", "C4"),
        ("C5", "C0", "C6", "C7", "C8"),
        ("C9", "C2", "C3", "C4", "C5"),
    )
    centers = tuple((20.0 + index * 10.0, 120.0, 130.0) for index in range(10))
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=color_id,
            confidence=0.95,
            representative_lab=centers[int(color_id[1:])],
        )
        for row, row_values in enumerate(values)
        for column, color_id in enumerate(row_values)
    )
    result = ColorDetectionResult(
        observations=observations,
        color_count=10,
        color_matrix=values,
        mean_confidence=0.95,
        diagnostics=ColorDetectionDiagnostics(
            rows=3,
            columns=5,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * 15,
            within_cell_spreads=(1.0,) * 15,
            cluster_centers_lab=centers,
            minimum_intercluster_distance=10.0,
        ),
    )
    return Board(result)


def test_collect_cat_exclusions_does_not_mutate_board() -> None:
    """Keep the shared direct-consequence planner completely read-only."""

    board = _board()
    expected = _matrix_values(board)

    collect_cat_exclusion_coordinates(board, 1, 1)

    assert _matrix_values(board) == expected


def test_collect_cat_exclusions_contains_other_same_color_cells() -> None:
    """Include every other current coordinate carrying the target color ID."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)

    assert (0, 0) in coordinates


def test_collect_cat_exclusions_contains_complete_remaining_row() -> None:
    """Include the whole target row except for the hypothetical cat itself."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)

    assert {(1, 0), (1, 2), (1, 3)} <= set(coordinates)


def test_collect_cat_exclusions_contains_complete_remaining_column() -> None:
    """Include the whole target column except for the hypothetical cat itself."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)

    assert {(0, 1), (2, 1), (3, 1)} <= set(coordinates)


def test_collect_cat_exclusions_contains_all_eight_existing_neighbors() -> None:
    """Include every orthogonal and diagonal neighbor of an interior target."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)
    neighbors = {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }

    assert neighbors <= set(coordinates)


def test_collect_cat_exclusions_omits_target() -> None:
    """Never include the coordinate that would receive the hypothetical K."""

    assert (1, 1) not in collect_cat_exclusion_coordinates(_board(), 1, 1)


def test_collect_cat_exclusions_deduplicates_overlapping_reasons() -> None:
    """Represent a coordinate once even when color, row, column, or neighbor overlap."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)

    assert len(coordinates) == len(set(coordinates))


def test_collect_cat_exclusions_are_sorted_row_major() -> None:
    """Provide stable ordering for both real placement and hypothetical analysis."""

    coordinates = collect_cat_exclusion_coordinates(_board(), 1, 1)

    assert coordinates == tuple(sorted(coordinates))


def test_collect_cat_exclusions_clip_board_edges() -> None:
    """Omit nonexistent neighbors without allowing negative index wraparound."""

    board = _board()
    coordinates = collect_cat_exclusion_coordinates(board, 0, 0)

    assert all(
        0 <= row < len(board.cells) and 0 <= column < len(board.cells[row])
        for row, column in coordinates
    )
    assert (-1, 0) not in coordinates
    assert (0, -1) not in coordinates


def test_collect_cat_exclusions_support_rectangular_board() -> None:
    """Derive row and column consequences without assuming square geometry."""

    board = _rectangular_board()
    coordinates = collect_cat_exclusion_coordinates(board, 1, 1)

    assert {(1, 0), (1, 2), (1, 3), (1, 4)} <= set(coordinates)
    assert {(0, 1), (2, 1)} <= set(coordinates)


def test_collect_cat_exclusions_include_existing_x_and_k_coordinates() -> None:
    """Plan geometry independently from terminal values later validated by callers."""

    board = _board()
    board.set_blocked(1, 0)
    board.set_cat(3, 1)

    coordinates = collect_cat_exclusion_coordinates(board, 1, 1)

    assert (1, 0) in coordinates
    assert (3, 1) in coordinates


def test_collect_cat_exclusions_preserve_unknown_cat_and_blocked_values() -> None:
    """Do not alter any state category while constructing a hypothetical plan."""

    board = _board()
    board.set_blocked(1, 0)
    board.set_cat(3, 1)
    expected = _matrix_values(board)

    collect_cat_exclusion_coordinates(board, 1, 1)

    assert _matrix_values(board) == expected


@pytest.mark.parametrize("terminal_value", ("K", "X", "INVALID"))
def test_collect_cat_exclusions_reject_invalid_target_without_mutation(
    terminal_value: str,
) -> None:
    """Require an unresolved C<n> target and preserve every invalid input state."""

    board = _board()
    board.cells[1][1] = terminal_value
    expected = _matrix_values(board)

    with pytest.raises(BoardStateError):
        collect_cat_exclusion_coordinates(board, 1, 1)

    assert _matrix_values(board) == expected


def test_place_cat_uses_shared_exclusion_coordinate_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep real placement and lookahead on the same direct-consequence plan."""

    board = _board()
    calls: list[tuple[int, int]] = []

    def fake_collector(
        target_board: Board,
        row: int,
        column: int,
    ) -> tuple[tuple[int, int], ...]:
        """Record delegation and expose one safely blockable coordinate."""

        assert target_board is board
        calls.append((row, column))
        return ((0, 1),)

    monkeypatch.setattr(
        board_actions_module,
        "collect_cat_exclusion_coordinates",
        fake_collector,
    )

    assert place_cat(board, 1, 1) is True
    assert calls == [(1, 1)]
    assert board.is_blocked(0, 1)
    assert board.is_unknown(1, 0)


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
