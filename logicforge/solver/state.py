"""Lifecycle metadata referencing the one mutable solver board."""

from dataclasses import dataclass
from enum import StrEnum, auto

from logicforge.core.board import Board
from logicforge.core.metadata import Metadata
from logicforge.rules.base_rule import RuleOutcome


class SolverStatus(StrEnum):
    """Describe the lifecycle of a deterministic deduction run.

    TODO: Add explicit cancellation and invalid-input statuses when application
    orchestration and parser diagnostics define their failure taxonomy.
    """

    READY = auto()
    RUNNING = auto()
    SOLVED = auto()
    STALLED = auto()
    CONTRADICTION = auto()


@dataclass(frozen=True, slots=True)
class SolverState:
    """Track solver lifecycle without duplicating the mutable board matrix.

    The frozen wrapper does not make ``board`` immutable and never copies its
    cells. Applied outcomes may describe in-place changes for later explanations.

    TODO: Define lifecycle metadata without introducing board snapshots when the
    concrete solver orchestration is implemented.
    """

    board: Board
    status: SolverStatus = SolverStatus.READY
    iteration: int = 0
    applied_outcomes: tuple[RuleOutcome, ...] = ()
    metadata: Metadata = ()
