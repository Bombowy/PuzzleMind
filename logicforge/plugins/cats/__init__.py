"""Reserved architecture for the planned Cats puzzle plugin."""

from logicforge.plugins.cats.board_actions import block_cell, place_cat
from logicforge.plugins.cats.parser import CatsParser
from logicforge.plugins.cats.rules import CatsRuleCatalog

__all__ = ["CatsParser", "CatsRuleCatalog", "block_cell", "place_cat"]
