"""Public Cats rules, actions, exact search, and vision contracts."""

from logicforge.plugins.cats.adjacent_color_pair_exclusion_rule import (
    AdjacentColorPairExclusionRule,
)
from logicforge.plugins.cats.board_actions import block_cell, place_cat
from logicforge.plugins.cats.color_confined_to_line_rule import (
    ColorConfinedToLineRule,
)
from logicforge.plugins.cats.color_subset_confined_to_lines_rule import (
    ColorSubsetConfinedToLinesRule,
)
from logicforge.plugins.cats.exact_search import (
    CatsExactSearchError,
    CatsExactSearchResult,
    CatsExactSearchStatus,
    apply_unique_cats_exact_solution,
    solve_cats_exact,
)
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatCellDiagnostic,
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
    CatsExistingCatDetector,
    CatsExistingCatDiagnostics,
    CatsExistingCatObservation,
)
from logicforge.plugins.cats.impossible_cat_candidate_rule import (
    ImpossibleCatCandidateRule,
)
from logicforge.plugins.cats.monochromatic_line_color_exclusion_rule import (
    MonochromaticLineColorExclusionRule,
)
from logicforge.plugins.cats.rule_loop import apply_cats_rules_until_stalled
from logicforge.plugins.cats.screen_state import (
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDetector,
    CatsScreenStateDiagnostics,
)
from logicforge.plugins.cats.single_remaining_color_cell_rule import (
    SingleRemainingColorCellRule,
)
from logicforge.plugins.cats.single_remaining_line_cell_rule import (
    SingleRemainingLineCellRule,
)
from logicforge.plugins.cats.tile_grid import (
    CatsTileComponentDiagnostic,
    CatsTileGridDetection,
    CatsTileGridDetectionError,
    CatsTileGridDetector,
    CatsTileGridDiagnostics,
)

__all__ = [
    "AdjacentColorPairExclusionRule",
    "CatsExactSearchError",
    "CatsExactSearchResult",
    "CatsExactSearchStatus",
    "CatsExistingCatCellDiagnostic",
    "CatsExistingCatDetection",
    "CatsExistingCatDetectionError",
    "CatsExistingCatDetector",
    "CatsExistingCatDiagnostics",
    "CatsExistingCatObservation",
    "CatsScreenPoint",
    "CatsScreenRect",
    "CatsScreenState",
    "CatsScreenStateDetection",
    "CatsScreenStateDetector",
    "CatsScreenStateDiagnostics",
    "CatsTileComponentDiagnostic",
    "CatsTileGridDetection",
    "CatsTileGridDetectionError",
    "CatsTileGridDetector",
    "CatsTileGridDiagnostics",
    "ColorConfinedToLineRule",
    "ColorSubsetConfinedToLinesRule",
    "ImpossibleCatCandidateRule",
    "MonochromaticLineColorExclusionRule",
    "SingleRemainingColorCellRule",
    "SingleRemainingLineCellRule",
    "apply_cats_rules_until_stalled",
    "apply_unique_cats_exact_solution",
    "block_cell",
    "place_cat",
    "solve_cats_exact",
]
