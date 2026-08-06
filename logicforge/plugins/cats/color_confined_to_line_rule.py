"""Cats deduction for colors confined to one row or one column."""

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import block_cell

type CellCoordinates = tuple[int, int]


class ColorConfinedToLineRule:
    """Block other colors when one color's candidates share a single line.

    The rule is stateless and independent from the future generic Rule Engine.
    Each call handles at most one color and one line, while one validated line may
    produce several atomic ``block_cell`` mutations on the sole Board matrix.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first numerically ordered row or column confinement move."""

        candidates_by_color = _group_unknown_coordinates(board)
        ordered_color_ids = sorted(
            candidates_by_color,
            key=lambda color_id: int(color_id[1:]),
        )
        for color_id in ordered_color_ids:
            coordinates = candidates_by_color[color_id]
            if len(coordinates) < 2:
                continue

            rows = {row for row, _ in coordinates}
            columns = {column for _, column in coordinates}
            if len(rows) == 1:
                row = next(iter(rows))
                targets = _validated_targets(
                    board,
                    _collect_row_plan(board, row, color_id),
                    color_id,
                )
                if targets:
                    _block_targets(board, targets)
                    return True

            if len(columns) == 1:
                column = next(iter(columns))
                targets = _validated_targets(
                    board,
                    _collect_column_plan(board, column, color_id),
                    color_id,
                )
                if targets:
                    _block_targets(board, targets)
                    return True
        return False


def _group_unknown_coordinates(
    board: Board,
) -> dict[str, list[CellCoordinates]]:
    """Group only current unresolved C<n> coordinates by logical color ID."""

    candidates_by_color: dict[str, list[CellCoordinates]] = {}
    for row, values in enumerate(board.cells):
        for column, value in enumerate(values):
            if board.is_unknown(row, column):
                candidates_by_color.setdefault(value, []).append((row, column))
    return candidates_by_color


def _collect_row_plan(
    board: Board,
    row: int,
    color_id: str,
) -> tuple[CellCoordinates, ...]:
    """Collect every row coordinate not occupied by the confined color."""

    return tuple(
        (row, column)
        for column in range(len(board.cells[row]))
        if board.get(row, column) != color_id
    )


def _collect_column_plan(
    board: Board,
    column: int,
    color_id: str,
) -> tuple[CellCoordinates, ...]:
    """Collect every existing column coordinate outside the confined color."""

    return tuple(
        (row, column)
        for row, values in enumerate(board.cells)
        if column < len(values) and board.get(row, column) != color_id
    )


def _validated_targets(
    board: Board,
    line_plan: tuple[CellCoordinates, ...],
    color_id: str,
) -> tuple[CellCoordinates, ...]:
    """Validate the complete line before returning only unresolved block targets."""

    targets: list[CellCoordinates] = []
    for row, column in line_plan:
        if board.is_cat(row, column):
            raise BoardStateError(
                f"Color {color_id} is confined to a line containing an existing "
                f"cat at ({row}, {column}) that would have to be blocked."
            )
        if board.is_blocked(row, column):
            continue
        if not board.is_unknown(row, column):
            raise BoardStateError(
                f"Color {color_id} is confined to a line containing invalid value "
                f"{board.get(row, column)!r} at ({row}, {column})."
            )
        targets.append((row, column))
    return tuple(targets)


def _block_targets(board: Board, targets: tuple[CellCoordinates, ...]) -> None:
    """Apply one fully validated line plan through the shared Cats action."""

    for row, column in targets:
        block_cell(board, row, column)
