"""Framework-independent domain models used across LogicForge."""

from logicforge.core.board import Board, BoardStateError
from logicforge.core.candidate import Candidate
from logicforge.core.cell import Cell
from logicforge.core.coordinates import Coordinates
from logicforge.core.region import Region

__all__ = ["Board", "BoardStateError", "Candidate", "Cell", "Coordinates", "Region"]
