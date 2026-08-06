"""Minimal fixed-point execution loop for the current Cats apply-style rules."""

from typing import Protocol

from logicforge.core.board import Board
from logicforge.plugins.cats.adjacent_color_pair_exclusion_rule import (
    AdjacentColorPairExclusionRule,
)
from logicforge.plugins.cats.color_confined_to_line_rule import (
    ColorConfinedToLineRule,
)
from logicforge.plugins.cats.monochromatic_line_color_exclusion_rule import (
    MonochromaticLineColorExclusionRule,
)
from logicforge.plugins.cats.single_remaining_color_cell_rule import (
    SingleRemainingColorCellRule,
)
from logicforge.plugins.cats.single_remaining_line_cell_rule import (
    SingleRemainingLineCellRule,
)


class CatsApplyRule(Protocol):
    """Describe the narrow mutation contract shared by current Cats rules."""

    def apply(self, board: Board) -> bool:
        """Mutate the board once and report whether a real change occurred."""


DEFAULT_CATS_RULES: tuple[CatsApplyRule, ...] = (
    SingleRemainingColorCellRule(),
    SingleRemainingLineCellRule(),
    MonochromaticLineColorExclusionRule(),
    AdjacentColorPairExclusionRule(),
    ColorConfinedToLineRule(),
)


def apply_cats_rules_until_stalled(
    board: Board,
    *,
    rules: tuple[CatsApplyRule, ...] | None = None,
) -> int:
    """Apply ordered Cats rules until one complete pass makes no mutation.

    Every successful rule application restarts evaluation from the first rule so
    higher-priority deductions immediately observe the latest state. The returned
    integer counts successful ``apply`` calls, regardless of how many cells each
    call changed. Exceptions deliberately propagate and stop evaluation.
    """

    ordered_rules = DEFAULT_CATS_RULES if rules is None else rules
    successful_applications = 0

    while True:
        for rule in ordered_rules:
            if not rule.apply(board):
                continue
            successful_applications += 1
            break
        else:
            return successful_applications
