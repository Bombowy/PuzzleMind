"""Fail-closed Cats input-geometry and complete-solution validation."""

from collections import Counter
from itertools import combinations

from logicforge.application.cats.click_plan import collect_cat_coordinates
from logicforge.application.cats.models import (
    CatsBoardInput,
    CatsSolvedBoard,
    CatsSolveStatus,
)


class CatsSolutionValidationError(RuntimeError):
    """Report an unsafe, partial, or internally inconsistent Cats solution."""


class CatsBoardGeometryMismatchError(CatsSolutionValidationError):
    """Reject Cats vision geometry before constructing or solving a Board."""


def validate_complete_cats_solution(solved: CatsSolvedBoard) -> None:
    """Require a complete one-cat-per-row, column, and original-color solution."""

    if solved.status is not CatsSolveStatus.COMPLETE:
        raise CatsSolutionValidationError(
            f"Cats solution status is {solved.status}, not COMPLETE."
        )

    board = solved.logical_board
    grid = solved.board_input.grid
    color_result = solved.board_input.color_result
    matrix = color_result.color_matrix
    if not (
        len(board.cells) == grid.rows == len(matrix)
        and all(len(row) == grid.columns for row in board.cells)
        and all(len(row) == grid.columns for row in matrix)
    ):
        raise CatsSolutionValidationError(
            "Logical board, color matrix, and grid dimensions are inconsistent."
        )
    if not grid.rows == grid.columns == color_result.color_count:
        raise CatsSolutionValidationError(
            "Rows, columns, and color_count must be equal for a complete Cats solution."
        )

    unsupported = tuple(
        (row, column, value)
        for row, values in enumerate(board.cells)
        for column, value in enumerate(values)
        if value not in {"K", "X"}
    )
    if unsupported:
        row, column, value = unsupported[0]
        if value.startswith("C") and value[1:].isdigit():
            raise CatsSolutionValidationError(
                f"Unresolved Cats cell remains at ({row}, {column}): {value}."
            )
        raise CatsSolutionValidationError(
            f"Unsupported Cats board value at ({row}, {column}): {value!r}."
        )

    cats = collect_cat_coordinates(board)
    for row in range(grid.rows):
        row_count = sum(cat_row == row for cat_row, _ in cats)
        if row_count != 1:
            raise CatsSolutionValidationError(
                f"Row {row} contains {row_count} cats instead of exactly one."
            )
    for column in range(grid.columns):
        column_count = sum(cat_column == column for _, cat_column in cats)
        if column_count != 1:
            raise CatsSolutionValidationError(
                f"Column {column} contains {column_count} cats instead of exactly one."
            )

    original_ids = tuple(color_id for row in matrix for color_id in row)
    expected_color_ids = set(original_ids)
    cat_color_ids = tuple(matrix[row][column] for row, column in cats)
    color_counts = Counter(cat_color_ids)
    for color_id in sorted(expected_color_ids):
        if color_counts[color_id] != 1:
            raise CatsSolutionValidationError(
                f"Original color {color_id} has {color_counts[color_id]} cats "
                "instead of exactly one."
            )
    if len(set(cat_color_ids)) != color_result.color_count:
        raise CatsSolutionValidationError(
            "The number of cat colors does not equal color_count."
        )

    for first, second in combinations(cats, 2):
        if max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1:
            raise CatsSolutionValidationError(
                f"Cats at {first} and {second} touch orthogonally or diagonally."
            )

    planned_coordinates = tuple(
        (target.row, target.column) for target in solved.click_plan
    )
    if len(planned_coordinates) != len(set(planned_coordinates)):
        raise CatsSolutionValidationError("Cat click plan contains duplicate targets.")
    existing_coordinates = tuple(
        (cat.row, cat.column) for cat in solved.board_input.existing_cat_detection.cats
    )
    if len(existing_coordinates) != len(set(existing_coordinates)):
        raise CatsSolutionValidationError(
            "Existing cat evidence contains duplicate coordinates."
        )
    for row, column in existing_coordinates:
        if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
            raise CatsSolutionValidationError(
                f"Existing cat coordinate ({row}, {column}) is outside the grid."
            )
        if (row, column) not in cats:
            raise CatsSolutionValidationError(
                f"Existing cat coordinate ({row}, {column}) is not K on final Board."
            )
    expected_new_cats = tuple(
        coordinate for coordinate in cats if coordinate not in set(existing_coordinates)
    )
    if planned_coordinates != expected_new_cats:
        raise CatsSolutionValidationError(
            "Cat click plan does not exactly match row-major new K coordinates."
        )
    if not len(cats) == grid.rows == grid.columns == color_result.color_count:
        raise CatsSolutionValidationError(
            "Cat count must equal rows, columns, and color_count."
        )


def validate_cats_board_input_geometry(board_input: CatsBoardInput) -> None:
    """Require a square Cats grid and one color per row before logical solving."""

    grid = board_input.grid
    color_result = board_input.color_result
    matrix = color_result.color_matrix
    if grid.rows != grid.columns or grid.rows != color_result.color_count:
        raise CatsBoardGeometryMismatchError(
            "Cats board geometry mismatch: "
            f"grid={grid.rows}x{grid.columns}, "
            f"colors={color_result.color_count}."
        )
    if len(matrix) != grid.rows or any(len(row) != grid.columns for row in matrix):
        matrix_widths = tuple(len(row) for row in matrix)
        raise CatsBoardGeometryMismatchError(
            "Cats color_matrix geometry mismatch: "
            f"grid={grid.rows}x{grid.columns}, "
            f"matrix_rows={len(matrix)}, matrix_widths={matrix_widths}."
        )
