"""Immutable logical cell snapshots."""

from dataclasses import dataclass

from logicforge.core.candidate import Candidate
from logicforge.core.coordinates import Coordinates
from logicforge.core.enums import CellState


@dataclass(frozen=True, slots=True)
class Cell:
    """Capture the complete puzzle-neutral state of one board position.

    Cells are immutable so solver transitions can retain prior snapshots for
    debugging and human-readable explanations without defensive copying.

    TODO: Add validated transition factories when the solver state model defines
    which state changes are legal and how contradictions are represented.
    """

    coordinates: Coordinates
    state: CellState = CellState.UNKNOWN
    value: str | None = None
    candidates: tuple[Candidate, ...] = ()
    region_ids: tuple[str, ...] = ()
