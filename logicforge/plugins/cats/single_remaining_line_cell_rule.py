"""Cats deduction for a row or column with one remaining possible cell."""

from collections.abc import Iterable

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import place_cat

type CellCoordinates = tuple[int, int]


class SingleRemainingLineCellRule:
    """Place one cat when a cat-free line contains one C<n> and otherwise X.

    The rule is stateless and uses the Board's sole mutable matrix. Rows precede
    columns, lower line indices precede higher ones, and one invocation delegates
    at most one forced placement to the atomic Cats ``place_cat`` action.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first deterministic single-remaining-line deduction."""

        _validate_supported_values(board)

        for row, values in enumerate(board.cells):
            candidate = _find_single_candidate(
                board,
                ((row, column) for column in range(len(values))),
            )
            if candidate is not None:
                return place_cat(board, *candidate)

        column_count = max((len(values) for values in board.cells), default=0)
        for column in range(column_count):
            candidate = _find_single_candidate(
                board,
                (
                    (row, column)
                    for row, values in enumerate(board.cells)
                    if column < len(values)
                ),
            )
            if candidate is not None:
                return place_cat(board, *candidate)
        return False


def _validate_supported_values(board: Board) -> None:
    """Reject unsupported cell state before any line can trigger a mutation."""

    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            value = board.get(row, column)
            if (
                board.is_unknown(row, column)
                or board.is_blocked(row, column)
                or board.is_cat(row, column)
            ):
                continue
            raise BoardStateError(
                f"Invalid board value {value!r} at ({row}, {column}) while "
                "analyzing remaining Cats line cells."
            )


def _find_single_candidate(
    board: Board,
    coordinates: Iterable[CellCoordinates],
) -> CellCoordinates | None:
    """Return the sole unresolved coordinate only when every other cell is X."""

    candidate: CellCoordinates | None = None
    for row, column in coordinates:
        if board.is_cat(row, column):
            return None
        if board.is_blocked(row, column):
            continue
        if candidate is not None:
            return None
        candidate = (row, column)
    return candidate
