"""One-step failed-candidate deduction for the Cats puzzle."""

from dataclasses import dataclass

from logicforge.core.board import Board, BoardStateError
from logicforge.plugins.cats.board_actions import (
    block_cell,
    collect_cat_exclusion_coordinates,
)

type CellCoordinates = tuple[int, int]
type CandidatesByColor = dict[str, tuple[CellCoordinates, ...]]


@dataclass(frozen=True, slots=True)
class _BoardAnalysis:
    """Hold immutable coordinate indexes derived from one read-only board scan."""

    unknown_coordinates: tuple[CellCoordinates, ...]
    candidates_by_color: CandidatesByColor
    unknown_by_row: tuple[tuple[CellCoordinates, ...], ...]
    unknown_by_column: tuple[tuple[CellCoordinates, ...], ...]
    cat_coordinates: tuple[CellCoordinates, ...]
    rows_with_cat: frozenset[int]
    columns_with_cat: frozenset[int]


class ImpossibleCatCandidateRule:
    """Block the first candidate whose immediate cat consequences contradict.

    This stateless rule performs only one-step lookahead. It never places a
    hypothetical cat or X on ``Board``: direct ``place_cat`` consequences are
    represented as a coordinate set and checked against current color and line
    viability. At most one proven-impossible candidate is then blocked through
    the shared ``block_cell`` action.
    """

    __slots__ = ()

    def apply(self, board: Board) -> bool:
        """Block the first row-major C<n> candidate with an immediate conflict."""

        analysis = _analyze_board(board)
        if not analysis.unknown_coordinates:
            return False
        _validate_existing_cat_state(analysis)

        for row, column in analysis.unknown_coordinates:
            if _candidate_causes_contradiction(
                board,
                row,
                column,
                analysis,
            ):
                return block_cell(board, row, column)
        return False


def _analyze_board(board: Board) -> _BoardAnalysis:
    """Validate values and build all candidate indexes in one matrix scan."""

    row_count = len(board.cells)
    column_count = len(board.cells[0]) if board.cells else 0
    unknown_coordinates: list[CellCoordinates] = []
    mutable_candidates_by_color: dict[str, list[CellCoordinates]] = {}
    mutable_unknown_by_row: list[list[CellCoordinates]] = [[] for _ in range(row_count)]
    mutable_unknown_by_column: list[list[CellCoordinates]] = [
        [] for _ in range(column_count)
    ]
    cat_coordinates: list[CellCoordinates] = []
    rows_with_cat: set[int] = set()
    columns_with_cat: set[int] = set()

    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            value = board.get(row, column)
            coordinates = (row, column)
            if board.is_unknown(row, column):
                unknown_coordinates.append(coordinates)
                mutable_candidates_by_color.setdefault(value, []).append(coordinates)
                mutable_unknown_by_row[row].append(coordinates)
                mutable_unknown_by_column[column].append(coordinates)
                continue
            if board.is_cat(row, column):
                cat_coordinates.append(coordinates)
                rows_with_cat.add(row)
                columns_with_cat.add(column)
                continue
            if board.is_blocked(row, column):
                continue
            raise BoardStateError(
                f"Invalid board value {value!r} at ({row}, {column}) while "
                "analyzing impossible Cats candidates."
            )

    return _BoardAnalysis(
        unknown_coordinates=tuple(unknown_coordinates),
        candidates_by_color={
            color_id: tuple(coordinates)
            for color_id, coordinates in mutable_candidates_by_color.items()
        },
        unknown_by_row=tuple(
            tuple(coordinates) for coordinates in mutable_unknown_by_row
        ),
        unknown_by_column=tuple(
            tuple(coordinates) for coordinates in mutable_unknown_by_column
        ),
        cat_coordinates=tuple(cat_coordinates),
        rows_with_cat=frozenset(rows_with_cat),
        columns_with_cat=frozenset(columns_with_cat),
    )


def _validate_existing_cat_state(analysis: _BoardAnalysis) -> None:
    """Reject pre-existing direct contradictions before testing candidates."""

    for row, coordinates in enumerate(analysis.unknown_by_row):
        if row not in analysis.rows_with_cat and not coordinates:
            raise BoardStateError(
                f"Row {row} has neither a confirmed cat nor an unresolved "
                "Cats candidate."
            )

    for column, coordinates in enumerate(analysis.unknown_by_column):
        if column not in analysis.columns_with_cat and not coordinates:
            raise BoardStateError(
                f"Column {column} has neither a confirmed cat nor an unresolved "
                "Cats candidate."
            )

    cat_count_by_row = [0] * len(analysis.unknown_by_row)
    cat_count_by_column = [0] * len(analysis.unknown_by_column)
    for row, column in analysis.cat_coordinates:
        cat_count_by_row[row] += 1
        cat_count_by_column[column] += 1

    for row, count in enumerate(cat_count_by_row):
        if count > 1:
            raise BoardStateError(f"Row {row} contains more than one confirmed cat.")
    for column, count in enumerate(cat_count_by_column):
        if count > 1:
            raise BoardStateError(
                f"Column {column} contains more than one confirmed cat."
            )

    for index, first in enumerate(analysis.cat_coordinates):
        for second in analysis.cat_coordinates[index + 1 :]:
            if _coordinates_touch(first, second):
                raise BoardStateError(
                    f"Confirmed cats at {first} and {second} touch each other."
                )


def _candidate_causes_contradiction(
    board: Board,
    row: int,
    column: int,
    analysis: _BoardAnalysis,
) -> bool:
    """Check immediate cat consequences without mutating the logical board."""

    target_color = board.get(row, column)
    hypothetical_x = frozenset(collect_cat_exclusion_coordinates(board, row, column))

    if any(coordinates in hypothetical_x for coordinates in analysis.cat_coordinates):
        return True

    ordered_color_ids = sorted(
        analysis.candidates_by_color,
        key=lambda color_id: int(color_id[1:]),
    )
    for color_id in ordered_color_ids:
        if color_id == target_color:
            continue
        if _eliminates_every_candidate(
            analysis.candidates_by_color[color_id],
            hypothetical_x,
        ):
            return True

    for candidate_row, coordinates in enumerate(analysis.unknown_by_row):
        if candidate_row == row or candidate_row in analysis.rows_with_cat:
            continue
        if _eliminates_every_candidate(coordinates, hypothetical_x):
            return True

    for candidate_column, coordinates in enumerate(analysis.unknown_by_column):
        if candidate_column == column or candidate_column in analysis.columns_with_cat:
            continue
        if _eliminates_every_candidate(coordinates, hypothetical_x):
            return True
    return False


def _eliminates_every_candidate(
    coordinates: tuple[CellCoordinates, ...],
    hypothetical_x: frozenset[CellCoordinates],
) -> bool:
    """Return whether the hypothetical plan removes a non-empty candidate set."""

    return bool(coordinates) and all(
        coordinates_item in hypothetical_x for coordinates_item in coordinates
    )


def _coordinates_touch(first: CellCoordinates, second: CellCoordinates) -> bool:
    """Return whether two different coordinates touch orthogonally or diagonally."""

    first_row, first_column = first
    second_row, second_column = second
    return (
        max(
            abs(first_row - second_row),
            abs(first_column - second_column),
        )
        == 1
    )
