"""Boundary for validating and applying rule outcomes to solver snapshots."""

from abc import ABC, abstractmethod

from logicforge.rules.base_rule import RuleOutcome
from logicforge.solver.state import SolverState


class PropagationStrategy(ABC):
    """Define atomic conversion of rule proposals into a new immutable state.

    Propagation is isolated from rule discovery so invariants, conflicts, and
    snapshot creation have one implementation point shared by every plugin.
    """

    @abstractmethod
    def propagate(
        self,
        state: SolverState,
        outcomes: tuple[RuleOutcome, ...],
    ) -> SolverState:
        """Validate outcomes and return the next immutable solver state.

        TODO: Implement typed transition validation, atomic conflict handling, and
        structural sharing in v0.4 after the board aggregate is finalized.
        """

        raise NotImplementedError
