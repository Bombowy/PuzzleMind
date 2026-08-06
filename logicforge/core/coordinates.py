"""Value objects for addressing logical cells independently of screen pixels."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Coordinates:
    """Identify one zero-based location on a logical puzzle board.

    Keeping logical coordinates separate from screen coordinates prevents the
    domain model from depending on capture resolution or automation details.

    TODO: Add construction-time bounds validation once board dimensions and the
    error model for invalid parser output have been finalized.
    """

    row: int
    column: int
