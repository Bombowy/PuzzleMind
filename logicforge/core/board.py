"""Single mutable logical board consumed directly by future solver rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from logicforge.vision.color_detector import ColorDetectionResult

CAT_VALUE: Final = "K"
BLOCKED_VALUE: Final = "X"


class BoardStateError(RuntimeError):
    """Raised when a requested mutation contradicts a finalized board cell."""


def _is_color_id(value: str) -> bool:
    """Return whether a value is an unresolved logical color identifier."""

    return value.startswith("C") and value[1:].isdigit()


class Board:
    """Own exactly one mutable row-major matrix of logical cell strings.

    Construction copies ``ColorDetectionResult.color_matrix`` into independent
    nested lists. From that point onward the solver-facing board mutates this one
    matrix directly: ``C<n>`` means unresolved color, ``K`` means cat, and ``X``
    means excluded. No parallel state matrix or immutable board snapshot exists.
    """

    __slots__ = ("cells",)

    cells: list[list[str]]

    def __init__(self, color_detection: ColorDetectionResult) -> None:
        """Deep-copy detected logical color IDs into the sole mutable matrix."""

        self.cells = [list(row) for row in color_detection.color_matrix]

    def get(self, row: int, column: int) -> str:
        """Return one value after validating zero-based matrix coordinates."""

        self._validate_coordinates(row, column)
        return self.cells[row][column]

    def set_cat(self, row: int, column: int) -> None:
        """Mark one cell as a confirmed cat by mutating the owned matrix in place."""

        self._set(row, column, CAT_VALUE)

    def set_blocked(self, row: int, column: int) -> None:
        """Mark one cell as excluded by mutating the owned matrix in place."""

        self._set(row, column, BLOCKED_VALUE)

    def is_unknown(self, row: int, column: int) -> bool:
        """Return whether a cell still carries its detected logical color ID."""

        return _is_color_id(self.get(row, column))

    def is_cat(self, row: int, column: int) -> bool:
        """Return whether a cell is currently marked as a confirmed cat."""

        return self.get(row, column) == CAT_VALUE

    def is_blocked(self, row: int, column: int) -> bool:
        """Return whether a cell is currently marked as excluded."""

        return self.get(row, column) == BLOCKED_VALUE

    def _set(self, row: int, column: int, value: str) -> None:
        """Apply an idempotent final state or reject a contradictory transition."""

        self._validate_coordinates(row, column)
        current = self.cells[row][column]
        if current == value:
            return
        if current == CAT_VALUE:
            raise BoardStateError(
                f"Cell ({row}, {column}) is already a confirmed cat "
                "and cannot be blocked."
            )
        if current == BLOCKED_VALUE:
            raise BoardStateError(
                f"Cell ({row}, {column}) is already blocked and cannot become a cat."
            )
        if not _is_color_id(current):
            raise BoardStateError(
                f"Cell ({row}, {column}) contains invalid board value: {current!r}."
            )
        self.cells[row][column] = value

    def _validate_coordinates(self, row: int, column: int) -> None:
        """Reject negative and out-of-range access with an actionable error."""

        if row < 0 or row >= len(self.cells):
            raise IndexError(
                f"Board row {row} is outside the valid range 0..{len(self.cells) - 1}."
            )
        column_count = len(self.cells[row])
        if column < 0 or column >= column_count:
            raise IndexError(
                f"Board column {column} is outside the valid range "
                f"0..{column_count - 1} for row {row}."
            )
