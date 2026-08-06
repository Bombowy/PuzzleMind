"""Cats deduction for N unresolved colors confined to exactly N lines."""

from collections.abc import Callable
from itertools import combinations

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import block_cell

type CellCoordinates = tuple[int, int]
type ColorSubset = tuple[str, ...]
type CandidatesByColor = dict[str, tuple[CellCoordinates, ...]]
type LineMembership = Callable[[int, int], bool]


class ColorSubsetConfinedToLinesRule:
    """Reserve N rows or columns for N unresolved colors and block outsiders.

    The rule is stateless and evaluates subset sizes from two upward. One call
    handles at most one numerically ordered color subset and one axis, while all
    resulting row-major mutations are delegated to the shared ``block_cell``
    action after complete plan validation.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Apply the first useful deterministic color-subset confinement."""

        _validate_board_values(board)
        candidates_by_color = _group_unknown_coordinates(board)
        ordered_color_ids = tuple(
            sorted(
                candidates_by_color,
                key=lambda color_id: int(color_id[1:]),
            )
        )

        for subset_size in range(2, len(ordered_color_ids)):
            for color_ids in combinations(ordered_color_ids, subset_size):
                selected_colors = frozenset(color_ids)

                rows = _collect_rows_for_colors(candidates_by_color, color_ids)
                if len(rows) == subset_size:
                    targets = _validated_row_targets(
                        board,
                        rows,
                        selected_colors,
                    )
                    if targets:
                        _block_targets(board, targets)
                        return True

                columns = _collect_columns_for_colors(
                    candidates_by_color,
                    color_ids,
                )
                if len(columns) == subset_size:
                    targets = _validated_column_targets(
                        board,
                        columns,
                        selected_colors,
                    )
                    if targets:
                        _block_targets(board, targets)
                        return True
        return False


def _validate_board_values(board: Board) -> None:
    """Reject unsupported state before subset planning can mutate any cell."""

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
                "analyzing Cats color subsets."
            )


def _group_unknown_coordinates(board: Board) -> CandidatesByColor:
    """Group only current C<n> coordinates into immutable row-major tuples."""

    mutable_candidates: dict[str, list[CellCoordinates]] = {}
    for row, values in enumerate(board.cells):
        for column, value in enumerate(values):
            if board.is_unknown(row, column):
                mutable_candidates.setdefault(value, []).append((row, column))
    return {
        color_id: tuple(coordinates)
        for color_id, coordinates in mutable_candidates.items()
    }


def _collect_rows_for_colors(
    candidates_by_color: CandidatesByColor,
    color_ids: ColorSubset,
) -> tuple[int, ...]:
    """Return sorted row indices containing any selected color candidate."""

    return tuple(
        sorted(
            {row for color_id in color_ids for row, _ in candidates_by_color[color_id]}
        )
    )


def _collect_columns_for_colors(
    candidates_by_color: CandidatesByColor,
    color_ids: ColorSubset,
) -> tuple[int, ...]:
    """Return sorted column indices containing any selected color candidate."""

    return tuple(
        sorted(
            {
                column
                for color_id in color_ids
                for _, column in candidates_by_color[color_id]
            }
        )
    )


def _validated_row_targets(
    board: Board,
    rows: tuple[int, ...],
    selected_colors: frozenset[str],
) -> tuple[CellCoordinates, ...]:
    """Validate complete reserved rows and return other unresolved colors."""

    reserved_rows = frozenset(rows)
    return _validated_targets(
        board,
        selected_colors,
        lambda row, column: row in reserved_rows,
    )


def _validated_column_targets(
    board: Board,
    columns: tuple[int, ...],
    selected_colors: frozenset[str],
) -> tuple[CellCoordinates, ...]:
    """Validate complete reserved columns and return other unresolved colors."""

    reserved_columns = frozenset(columns)
    return _validated_targets(
        board,
        selected_colors,
        lambda row, column: column in reserved_columns,
    )


def _validated_targets(
    board: Board,
    selected_colors: frozenset[str],
    is_reserved: LineMembership,
) -> tuple[CellCoordinates, ...]:
    """Validate a full line union before exposing row-major mutation targets."""

    targets: list[CellCoordinates] = []
    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            if not is_reserved(row, column):
                continue
            value = board.get(row, column)
            if board.is_cat(row, column):
                raise BoardStateError(
                    f"Reserved Cats color-subset line contains existing cat at "
                    f"({row}, {column})."
                )
            if board.is_blocked(row, column):
                continue
            if not board.is_unknown(row, column):
                raise BoardStateError(
                    f"Reserved Cats color-subset line contains invalid value "
                    f"{value!r} at ({row}, {column})."
                )
            if value not in selected_colors:
                targets.append((row, column))
    return tuple(targets)


def _block_targets(board: Board, targets: tuple[CellCoordinates, ...]) -> None:
    """Apply one completely validated subset plan through the Cats action."""

    for row, column in targets:
        block_cell(board, row, column)
