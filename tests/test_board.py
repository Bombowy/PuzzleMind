"""Tests for the single mutable solver-facing board matrix."""

import pytest

from logicforge.core import BoardStateError
from logicforge.core.board import Board
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)


def _color_result() -> ColorDetectionResult:
    """Build a small valid vision result without invoking OpenCV infrastructure."""

    matrix = (("C0", "C1", "C0"), ("C1", "C0", "C1"))
    centers = ((120.0, 130.0, 140.0), (180.0, 110.0, 90.0))
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=color_id,
            confidence=0.95,
            representative_lab=centers[int(color_id[1:])],
        )
        for row, values in enumerate(matrix)
        for column, color_id in enumerate(values)
    )
    diagnostics = ColorDetectionDiagnostics(
        rows=2,
        columns=3,
        cluster_distance_threshold=18.0,
        sample_pixel_counts=(100,) * 6,
        within_cell_spreads=(1.0,) * 6,
        cluster_centers_lab=centers,
        minimum_intercluster_distance=70.0,
    )
    return ColorDetectionResult(
        observations=observations,
        color_count=2,
        color_matrix=matrix,
        mean_confidence=0.95,
        diagnostics=diagnostics,
    )


def test_constructor_copies_color_matrix_into_nested_mutable_lists() -> None:
    """Own one mutable matrix without retaining an editable vision-state alias."""

    color_result = _color_result()

    board = Board(color_result)

    assert board.cells == [["C0", "C1", "C0"], ["C1", "C0", "C1"]]
    assert isinstance(board.cells, list)
    assert all(isinstance(row, list) for row in board.cells)
    board.cells[0][0] = "K"
    assert color_result.color_matrix[0][0] == "C0"


def test_get_and_state_queries_read_the_same_matrix() -> None:
    """Interpret only C<n>, K, and X values from the sole matrix."""

    board = Board(_color_result())

    assert board.get(0, 1) == "C1"
    assert board.is_unknown(0, 1)
    assert not board.is_cat(0, 1)
    assert not board.is_blocked(0, 1)


def test_set_cat_mutates_requested_cell_in_place() -> None:
    """Replace an unresolved color with K without creating another board state."""

    board = Board(_color_result())
    cells_identity = id(board.cells)
    rows_identity = tuple(id(row) for row in board.cells)

    board.set_cat(0, 0)

    assert board.get(0, 0) == "K"
    assert board.is_cat(0, 0)
    assert not board.is_unknown(0, 0)
    assert id(board.cells) == cells_identity
    assert tuple(id(row) for row in board.cells) == rows_identity


def test_set_blocked_mutates_requested_cell_in_place() -> None:
    """Replace an unresolved color with X in the existing nested list."""

    board = Board(_color_result())

    board.set_blocked(0, 0)

    assert board.get(0, 0) == "X"
    assert board.is_blocked(0, 0)
    assert not board.is_unknown(0, 0)


def test_setting_cat_twice_is_idempotent() -> None:
    """Permit K to K without replacing the matrix or raising an error."""

    board = Board(_color_result())
    cells_identity = id(board.cells)
    row_identity = id(board.cells[0])

    board.set_cat(0, 2)
    board.set_cat(0, 2)

    assert board.get(0, 2) == "K"
    assert id(board.cells) == cells_identity
    assert id(board.cells[0]) == row_identity


def test_setting_blocked_twice_is_idempotent() -> None:
    """Permit X to X without replacing the matrix or raising an error."""

    board = Board(_color_result())

    board.set_blocked(1, 1)
    board.set_blocked(1, 1)

    assert board.get(1, 1) == "X"


def test_confirmed_cat_cannot_become_blocked() -> None:
    """Reject K to X and preserve the confirmed cat after the exception."""

    board = Board(_color_result())

    board.set_cat(0, 2)
    with pytest.raises(BoardStateError, match="already a confirmed cat"):
        board.set_blocked(0, 2)

    assert board.get(0, 2) == "K"
    assert board.is_cat(0, 2)


def test_blocked_cell_cannot_become_cat() -> None:
    """Reject X to K and preserve the exclusion after the exception."""

    board = Board(_color_result())

    board.set_blocked(1, 1)
    with pytest.raises(BoardStateError, match="already blocked"):
        board.set_cat(1, 1)

    assert board.get(1, 1) == "X"
    assert board.is_blocked(1, 1)


def test_conflict_in_one_cell_does_not_change_other_cells() -> None:
    """Keep the complete single matrix unchanged when one transition conflicts."""

    board = Board(_color_result())
    board.set_cat(0, 0)
    expected_cells = [["K", "C1", "C0"], ["C1", "C0", "C1"]]

    with pytest.raises(BoardStateError):
        board.set_blocked(0, 0)

    assert board.cells == expected_cells


def test_mutation_rejects_an_invalid_existing_board_value() -> None:
    """Fail safely if external direct list access introduced an unsupported value."""

    board = Board(_color_result())
    board.cells[0][0] = "INVALID"

    with pytest.raises(BoardStateError, match="invalid board value"):
        board.set_cat(0, 0)

    assert board.get(0, 0) == "INVALID"


def test_all_accessors_reject_invalid_coordinates() -> None:
    """Prevent Python negative indexing and silent access outside board geometry."""

    board = Board(_color_result())

    for operation in (
        board.get,
        board.set_cat,
        board.set_blocked,
        board.is_unknown,
        board.is_cat,
        board.is_blocked,
    ):
        try:
            operation(-1, 0)
        except IndexError as error:
            assert "row -1" in str(error)
        else:
            raise AssertionError("Negative rows must be rejected.")

        try:
            operation(0, 3)
        except IndexError as error:
            assert "column 3" in str(error)
        else:
            raise AssertionError("Out-of-range columns must be rejected.")
