"""Reserved architecture for the planned Cats puzzle plugin."""

from logicforge.plugins.cats.board_actions import block_cell, place_cat
from logicforge.plugins.cats.color_confined_to_line_rule import (
    ColorConfinedToLineRule,
)
from logicforge.plugins.cats.parser import CatsParser
from logicforge.plugins.cats.rules import CatsRuleCatalog
from logicforge.plugins.cats.single_remaining_color_cell_rule import (
    SingleRemainingColorCellRule,
)

__all__ = [
    "CatsParser",
    "CatsRuleCatalog",
    "ColorConfinedToLineRule",
    "SingleRemainingColorCellRule",
    "block_cell",
    "place_cat",
]
