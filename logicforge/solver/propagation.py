"""Boundary for validating outcomes applied to the shared mutable board."""

from abc import ABC, abstractmethod

from logicforge.rules.base_rule import RuleOutcome
from logicforge.solver.state import SolverState


class PropagationStrategy(ABC):
    """Define validated application of rule proposals to the existing board.

    Propagation is isolated from rule discovery so invariants, conflicts, and
    ordered in-place mutations have one implementation point shared by plugins.
    """

    @abstractmethod
    def propagate(
        self,
        state: SolverState,
        outcomes: tuple[RuleOutcome, ...],
    ) -> SolverState:
        """Validate outcomes, mutate the shared board, and return lifecycle state.

        TODO: Implement typed transition validation and conflict handling without
        copying the board matrix in v0.4.
        """

        raise NotImplementedError
