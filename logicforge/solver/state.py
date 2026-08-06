"""Immutable records representing solver progress."""

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
    """Capture a complete, replayable solver snapshot after one engine iteration.

    Storing applied outcomes alongside the board makes state transitions auditable
    and gives the explainability layer stable provenance without solver coupling.

    TODO: Add parent-state identifiers and compact persistence once real puzzle
    fixtures establish memory and replay requirements.
    """

    board: Board
    status: SolverStatus = SolverStatus.READY
    iteration: int = 0
    applied_outcomes: tuple[RuleOutcome, ...] = ()
    metadata: Metadata = ()
