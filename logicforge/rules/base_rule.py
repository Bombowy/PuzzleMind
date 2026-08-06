"""Base contracts and data records for deterministic puzzle rules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.core.board import Board
from logicforge.core.coordinates import Coordinates
from logicforge.core.enums import DeductionKind
from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class RuleContext:
    """Provide immutable input and orchestration metadata to a rule evaluation.

    TODO: Add cancellation, tracing, and plugin capability information when the
    v0.4 engine lifecycle has stable operational requirements.
    """

    board: Board
    iteration: int
    metadata: Metadata = ()


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """Describe a proposed state transition without applying it to the board.

    Separating proposals from mutation lets the engine validate conflicts, retain
    provenance, and produce explanations before a new snapshot is committed.

    TODO: Replace the opaque payload with typed transition commands in v0.4 once
    assignment, elimination, and contradiction semantics are fully specified.
    """

    rule_id: str
    kind: DeductionKind
    affected_coordinates: tuple[Coordinates, ...]
    summary: str
    payload: Metadata = ()


class BaseRule(ABC):
    """Define one stateless, deterministic, and independently testable rule.

    Implementations must inspect only the supplied context and return proposed
    outcomes. Side effects, I/O, automation, and direct board mutation are banned.
    """

    identifier: str
    description: str

    @abstractmethod
    def evaluate(self, context: RuleContext) -> tuple[RuleOutcome, ...]:
        """Return every deduction this rule can justify for the current snapshot.

        TODO: Implement concrete plugin rules in milestone-specific modules after
        the engine defines ordering, conflict resolution, and outcome validation.
        """

        raise NotImplementedError
