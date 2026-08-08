"""Cats deduction for a color with exactly one unresolved cell."""

from logicforge.core.board import Board
from logicforge.plugins.cats.board_actions import place_cat


class SingleRemainingColorCellRule:
    """Place one forced cat in the numerically lowest singleton color class.

    The rule is deliberately stateless and runs in the Cats fixed-point rule loop.
    One call scans only current ``C<n>`` entries and delegates at most one real
    mutation to the atomic Cats ``place_cat`` action.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first deterministic singleton-color move, if one exists."""

        candidates_by_color: dict[str, list[tuple[int, int]]] = {}
        for row, values in enumerate(board.cells):
            for column in range(len(values)):
                if not board.is_unknown(row, column):
                    continue
                color_id = board.get(row, column)
                candidates_by_color.setdefault(color_id, []).append((row, column))

        ordered_color_ids = sorted(
            candidates_by_color,
            key=lambda color_id: int(color_id[1:]),
        )
        for color_id in ordered_color_ids:
            coordinates = candidates_by_color[color_id]
            if len(coordinates) != 1:
                continue
            row, column = coordinates[0]
            return place_cat(board, row, column)
        return False
