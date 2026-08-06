"""Rule-engine orchestration port."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.core.board import Board
from logicforge.rules.base_rule import BaseRule, RuleOutcome


@dataclass(frozen=True, slots=True)
class RuleEngineResult:
    """Capture one deterministic evaluation pass and its proposed outcomes.

    TODO: Add timing, skipped-rule reasons, conflicts, and trace identifiers when
    the explainability and observability contracts are introduced.
    """

    outcomes: tuple[RuleOutcome, ...]
    evaluated_rule_ids: tuple[str, ...]


class RuleEngine(ABC):
    """Coordinate rule evaluation without owning puzzle-specific knowledge.

    The engine is an application service boundary. A future implementation will
    enforce deterministic ordering and validate proposals before propagation.
    """

    @abstractmethod
    def evaluate(
        self,
        board: Board,
        rules: tuple[BaseRule, ...],
        *,
        iteration: int,
    ) -> RuleEngineResult:
        """Evaluate a complete ordered rule set against one board snapshot.

        TODO: Implement deterministic scheduling, isolation of rule failures,
        outcome validation, and conflict reporting in the v0.4 milestone.
        """

        raise NotImplementedError
