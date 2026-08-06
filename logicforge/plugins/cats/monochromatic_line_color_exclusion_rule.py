"""Cats deduction for a line reserved by one remaining color."""

from collections.abc import Iterable

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import block_cell

type CellCoordinates = tuple[int, int]
type LineCandidate = tuple[int, int, int, str]

_ROW_AXIS = 0
_COLUMN_AXIS = 1


class MonochromaticLineColorExclusionRule:
    """Block one line's sole remaining color everywhere outside that line.

    The rule is stateless and works directly on the Board's sole mutable matrix.
    One call handles at most one row or column, but may delegate several planned
    row-major mutations to ``block_cell``.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first useful monochromatic line in deterministic order."""

        candidates = _collect_line_candidates(board)
        for _, axis_order, line_index, color_id in sorted(candidates):
            targets = _collect_exclusion_plan(
                board,
                axis_order=axis_order,
                line_index=line_index,
                color_id=color_id,
            )
            if not targets:
                continue

            for row, column in targets:
                block_cell(board, row, column)
            return True
        return False


def _collect_line_candidates(board: Board) -> tuple[LineCandidate, ...]:
    """Validate every line and return all qualifying row/column candidates."""

    candidates: list[LineCandidate] = []
    for row, values in enumerate(board.cells):
        candidate = _analyze_line(
            board,
            ((row, column) for column in range(len(values))),
        )
        if candidate is not None:
            candidates.append((int(candidate[1:]), _ROW_AXIS, row, candidate))

    column_count = max((len(values) for values in board.cells), default=0)
    for column in range(column_count):
        candidate = _analyze_line(
            board,
            (
                (row, column)
                for row, values in enumerate(board.cells)
                if column < len(values)
            ),
        )
        if candidate is not None:
            candidates.append((int(candidate[1:]), _COLUMN_AXIS, column, candidate))
    return tuple(candidates)


def _analyze_line(
    board: Board,
    coordinates: Iterable[CellCoordinates],
) -> str | None:
    """Return the sole active color, skipping X and disqualifying lines with K."""

    color_ids: set[str] = set()
    contains_cat = False
    for row, column in coordinates:
        value = board.get(row, column)
        if board.is_blocked(row, column):
            continue
        if board.is_cat(row, column):
            contains_cat = True
            continue
        if not board.is_unknown(row, column):
            raise BoardStateError(
                f"Invalid board value {value!r} at ({row}, {column}) while "
                "analyzing a Cats line."
            )
        color_ids.add(value)

    if contains_cat or len(color_ids) != 1:
        return None
    return next(iter(color_ids))


def _collect_exclusion_plan(
    board: Board,
    *,
    axis_order: int,
    line_index: int,
    color_id: str,
) -> tuple[CellCoordinates, ...]:
    """Plan only same-color unresolved cells outside the selected line."""

    targets: list[CellCoordinates] = []
    for row, values in enumerate(board.cells):
        for column, value in enumerate(values):
            inside_selected_line = (
                row == line_index if axis_order == _ROW_AXIS else column == line_index
            )
            if not inside_selected_line and value == color_id:
                targets.append((row, column))
    return tuple(targets)
