"""Root aggregate for immutable puzzle-board snapshots."""

from dataclasses import dataclass

from logicforge.core.cell import Cell
from logicforge.core.metadata import Metadata
from logicforge.core.region import Region


@dataclass(frozen=True, slots=True)
class Board:
    """Represent a complete puzzle-neutral board at one moment in time.

    Board intentionally provides no lookup or mutation behavior in v0.1. Future
    APIs must preserve immutable snapshots and enforce aggregate invariants in one
    place rather than scattering them across solvers and plugins.

    TODO: Add validated construction and indexed cell access in v0.3 after the
    parser contract defines how incomplete and uncertain observations are stored.
    """

    width: int
    height: int
    cells: tuple[Cell, ...] = ()
    regions: tuple[Region, ...] = ()
    puzzle_type: str | None = None
    metadata: Metadata = ()
