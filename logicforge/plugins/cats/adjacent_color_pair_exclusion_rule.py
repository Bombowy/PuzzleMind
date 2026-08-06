"""Cats deduction for exactly two orthogonally adjacent color candidates."""

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import block_cell

type CellCoordinates = tuple[int, int]


class AdjacentColorPairExclusionRule:
    """Block cells that neighbor every possible cat in an adjacent color pair.

    The rule is stateless and operates on the Board's sole mutable matrix. Each
    invocation handles at most one numerically ordered color, validates its whole
    perpendicular exclusion plan, and only then delegates writes to ``block_cell``.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first adjacent-pair exclusion that produces a real change."""

        candidates_by_color = _group_unknown_coordinates(board)
        ordered_color_ids = sorted(
            candidates_by_color,
            key=lambda color_id: int(color_id[1:]),
        )
        for color_id in ordered_color_ids:
            coordinates = candidates_by_color[color_id]
            if len(coordinates) != 2:
                continue

            first, second = coordinates
            if not _are_orthogonally_adjacent(first, second):
                continue

            targets = _collect_perpendicular_targets(board, first, second)
            coordinates_to_block = _validate_and_collect_changes(
                board,
                targets,
                color_id,
            )
            if not coordinates_to_block:
                continue

            for row, column in coordinates_to_block:
                block_cell(board, row, column)
            return True
        return False


def _group_unknown_coordinates(
    board: Board,
) -> dict[str, list[CellCoordinates]]:
    """Group only current unresolved ``C<n>`` cells in row-major order."""

    candidates_by_color: dict[str, list[CellCoordinates]] = {}
    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            if not board.is_unknown(row, column):
                continue
            color_id = board.get(row, column)
            candidates_by_color.setdefault(color_id, []).append((row, column))
    return candidates_by_color


def _are_orthogonally_adjacent(
    first: CellCoordinates,
    second: CellCoordinates,
) -> bool:
    """Return whether two cells share an edge according to Manhattan distance."""

    first_row, first_column = first
    second_row, second_column = second
    return abs(first_row - second_row) + abs(first_column - second_column) == 1


def _collect_perpendicular_targets(
    board: Board,
    first: CellCoordinates,
    second: CellCoordinates,
) -> tuple[CellCoordinates, ...]:
    """Build an ordered, de-duplicated plan on both sides of the adjacent pair."""

    first_row, first_column = first
    second_row, second_column = second
    if first_column == second_column:
        proposed = (
            (first_row, first_column - 1),
            (first_row, first_column + 1),
            (second_row, second_column - 1),
            (second_row, second_column + 1),
        )
    else:
        proposed = (
            (first_row - 1, first_column),
            (first_row + 1, first_column),
            (second_row - 1, second_column),
            (second_row + 1, second_column),
        )

    return tuple(
        sorted(
            {
                (row, column)
                for row, column in proposed
                if _coordinate_exists(board, row, column)
            }
        )
    )


def _coordinate_exists(board: Board, row: int, column: int) -> bool:
    """Check bounds without asking Board to resolve an intentionally clipped cell."""

    return 0 <= row < len(board.cells) and 0 <= column < len(board.cells[row])


def _validate_and_collect_changes(
    board: Board,
    targets: tuple[CellCoordinates, ...],
    color_id: str,
) -> tuple[CellCoordinates, ...]:
    """Validate the complete plan before returning unresolved cells to mutate."""

    coordinates_to_block: list[CellCoordinates] = []
    for row, column in targets:
        if board.is_cat(row, column):
            raise BoardStateError(
                f"Adjacent pair for {color_id} would have to block existing cat "
                f"at ({row}, {column})."
            )
        if board.is_blocked(row, column):
            continue
        if not board.is_unknown(row, column):
            raise BoardStateError(
                f"Adjacent pair for {color_id} has invalid target value "
                f"{board.get(row, column)!r} at ({row}, {column})."
            )
        coordinates_to_block.append((row, column))
    return tuple(coordinates_to_block)
