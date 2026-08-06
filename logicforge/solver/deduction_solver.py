"""Top-level deterministic solver use-case contract."""

from abc import ABC, abstractmethod

from logicforge.core.board import Board
from logicforge.rules.base_rule import BaseRule
from logicforge.solver.state import SolverState


class DeductionSolver(ABC):
    """Orchestrate engine passes and propagation until a terminal state is reached.

    This boundary owns the use-case lifecycle, while rules discover deductions and
    propagation owns transitions. It must never depend on vision or automation.
    """

    @abstractmethod
    def solve(self, board: Board, rules: tuple[BaseRule, ...]) -> SolverState:
        """Run deterministic deduction for a board and an explicit ordered rule set.

        TODO: Implement bounded iteration, cycle detection, cancellation, and
        terminal-state classification after the v0.4 engine is available.
        """

        raise NotImplementedError
