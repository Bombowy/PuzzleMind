"""Reserved architecture for the planned Cats puzzle plugin."""

from logicforge.plugins.cats.adjacent_color_pair_exclusion_rule import (
    AdjacentColorPairExclusionRule,
)
from logicforge.plugins.cats.board_actions import block_cell, place_cat
from logicforge.plugins.cats.color_confined_to_line_rule import (
    ColorConfinedToLineRule,
)
from logicforge.plugins.cats.parser import CatsParser
from logicforge.plugins.cats.rule_loop import apply_cats_rules_until_stalled
from logicforge.plugins.cats.rules import CatsRuleCatalog
from logicforge.plugins.cats.single_remaining_color_cell_rule import (
    SingleRemainingColorCellRule,
)

__all__ = [
    "AdjacentColorPairExclusionRule",
    "CatsParser",
    "CatsRuleCatalog",
    "ColorConfinedToLineRule",
    "SingleRemainingColorCellRule",
    "apply_cats_rules_until_stalled",
    "block_cell",
    "place_cat",
]
