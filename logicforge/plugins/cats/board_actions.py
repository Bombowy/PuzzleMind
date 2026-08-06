"""Cats-specific atomic mutations over the single mutable logical board."""

from logicforge.core.board import Board, BoardStateError

type CellCoordinates = tuple[int, int]


def place_cat(board: Board, row: int, column: int) -> bool:
    """Place one cat and apply every immediate Cats exclusion atomically.

    The function plans color, row, column, and eight-neighbor exclusions before
    performing a write. Any existing cat or invalid value in that plan raises
    ``BoardStateError`` while the original matrix is untouched. A repeated request
    for the same confirmed cat is an idempotent ``False`` result.
    """

    current = board.get(row, column)
    if board.is_cat(row, column):
        return False
    if board.is_blocked(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) is already blocked and cannot become a cat."
        )
    if not board.is_unknown(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) contains invalid board value: {current!r}."
        )

    blocked_coordinates = collect_cat_exclusion_coordinates(
        board,
        row,
        column,
    )
    _validate_no_cat_conflicts(
        board,
        blocked_coordinates,
        cat_row=row,
        cat_column=column,
    )

    board.set_cat(row, column)
    for blocked_row, blocked_column in blocked_coordinates:
        if board.is_unknown(blocked_row, blocked_column):
            board.set_blocked(blocked_row, blocked_column)
    return True


def block_cell(board: Board, row: int, column: int) -> bool:
    """Block one unresolved cell without triggering any Cats propagation."""

    current = board.get(row, column)
    if board.is_blocked(row, column):
        return False
    if board.is_cat(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) is already a confirmed cat "
            "and cannot be blocked."
        )
    if not board.is_unknown(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) contains invalid board value: {current!r}."
        )
    board.set_blocked(row, column)
    return True


def collect_cat_exclusion_coordinates(
    board: Board,
    row: int,
    column: int,
) -> tuple[CellCoordinates, ...]:
    """Return the direct exclusion plan for an unresolved hypothetical cat.

    The pure helper centralizes the exact color, row, column, and eight-neighbor
    geometry used by ``place_cat``. It validates the target, reads the current
    board without mutating it, excludes the target itself, removes duplicate
    coordinates, and returns the plan in deterministic row-major order.
    """

    color_id = board.get(row, column)
    if board.is_cat(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) is already a confirmed cat and has no "
            "hypothetical exclusion plan."
        )
    if board.is_blocked(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) is already blocked and cannot become a cat."
        )
    if not board.is_unknown(row, column):
        raise BoardStateError(
            f"Cell ({row}, {column}) contains invalid board value: {color_id!r}."
        )

    coordinates: set[CellCoordinates] = set()
    target = (row, column)

    for candidate_row, values in enumerate(board.cells):
        for candidate_column in range(len(values)):
            if (candidate_row, candidate_column) != target and board.get(
                candidate_row, candidate_column
            ) == color_id:
                coordinates.add((candidate_row, candidate_column))

    for candidate_column in range(len(board.cells[row])):
        if (row, candidate_column) != target:
            coordinates.add((row, candidate_column))

    for candidate_row, values in enumerate(board.cells):
        if column < len(values) and (candidate_row, column) != target:
            coordinates.add((candidate_row, column))

    for row_offset in (-1, 0, 1):
        neighbor_row = row + row_offset
        if not 0 <= neighbor_row < len(board.cells):
            continue
        for column_offset in (-1, 0, 1):
            neighbor_column = column + column_offset
            if (neighbor_row, neighbor_column) != target and 0 <= neighbor_column < len(
                board.cells[neighbor_row]
            ):
                coordinates.add((neighbor_row, neighbor_column))

    return tuple(sorted(coordinates))


def _validate_no_cat_conflicts(
    board: Board,
    blocked_coordinates: tuple[CellCoordinates, ...],
    *,
    cat_row: int,
    cat_column: int,
) -> None:
    """Reject the complete plan before mutation if any exclusion is contradictory."""

    for row, column in blocked_coordinates:
        if board.is_cat(row, column):
            raise BoardStateError(
                f"Cannot place cat at ({cat_row}, {cat_column}): existing cat at "
                f"({row}, {column}) would have to be blocked."
            )
        if not board.is_unknown(row, column) and not board.is_blocked(row, column):
            raise BoardStateError(
                f"Cannot place cat at ({cat_row}, {cat_column}): cell "
                f"({row}, {column}) contains invalid board value "
                f"{board.get(row, column)!r}."
            )
