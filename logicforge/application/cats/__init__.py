"""Public application API for Cats analysis, solving, validation, and autoplay."""

from logicforge.application.cats.analysis import analyze_captured_cats_board
from logicforge.application.cats.autoplay import (
    CatsAutomationError,
    CatsAutomationPhase,
    CatsAutomationTimeoutError,
    CatsAutoplayRunner,
    CatsAutoplaySettings,
    CatsAutoplaySummary,
)
from logicforge.application.cats.click_plan import (
    CatClickExecutionError,
    CatClickPlanError,
    build_cat_click_plan,
    collect_cat_coordinates,
    create_cat_click_target,
    execute_cat_click_plan,
    get_grid_cell,
)
from logicforge.application.cats.models import (
    CatClickTarget,
    CatsBoardInput,
    CatsSolvedBoard,
    CatsSolveStatus,
)
from logicforge.application.cats.presentation import (
    format_matrix,
    print_cat_click_plan,
    print_solve_information,
)
from logicforge.application.cats.solving import (
    classify_result,
    count_unresolved_cells,
    solve_analyzed_cats_board,
    solve_captured_cats_board,
)
from logicforge.application.cats.validation import (
    CatsBoardGeometryMismatchError,
    CatsSolutionValidationError,
    validate_cats_board_input_geometry,
    validate_complete_cats_solution,
)

__all__ = [
    "CatClickExecutionError",
    "CatClickPlanError",
    "CatClickTarget",
    "CatsAutomationError",
    "CatsAutomationPhase",
    "CatsAutomationTimeoutError",
    "CatsAutoplayRunner",
    "CatsAutoplaySettings",
    "CatsAutoplaySummary",
    "CatsBoardGeometryMismatchError",
    "CatsBoardInput",
    "CatsSolutionValidationError",
    "CatsSolveStatus",
    "CatsSolvedBoard",
    "analyze_captured_cats_board",
    "build_cat_click_plan",
    "classify_result",
    "collect_cat_coordinates",
    "count_unresolved_cells",
    "create_cat_click_target",
    "execute_cat_click_plan",
    "format_matrix",
    "get_grid_cell",
    "print_cat_click_plan",
    "print_solve_information",
    "solve_analyzed_cats_board",
    "solve_captured_cats_board",
    "validate_cats_board_input_geometry",
    "validate_complete_cats_solution",
]
