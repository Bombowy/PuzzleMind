"""Immutable application models shared by Cats solve and autoplay use cases."""

from dataclasses import dataclass, field
from enum import StrEnum

from logicforge.core import Board
from logicforge.plugins.cats.exact_search import CatsExactSearchResult
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDiagnostics,
)
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.color_detector import ColorDetectionResult
from logicforge.vision.grid_detector import GridDetection


class CatsSolveStatus(StrEnum):
    """Expose the existing Cats solve outcomes as a closed typed set."""

    COMPLETE = "COMPLETE"
    STALLED = "STALLED"
    UNSAT = "UNSAT"
    AMBIGUOUS = "AMBIGUOUS"
    SEARCH_LIMIT = "SEARCH_LIMIT"


@dataclass(frozen=True, slots=True)
class CatClickTarget:
    """Map one logical cat to screenshot and virtual-desktop coordinates."""

    row: int
    column: int
    screenshot_x: int
    screenshot_y: int
    desktop_x: int
    desktop_y: int


@dataclass(frozen=True, slots=True)
class CatsBoardInput:
    """Retain immutable vision output required to solve one captured Cats board."""

    detected_board: BoardDetection
    grid: GridDetection
    color_result: ColorDetectionResult
    existing_cat_detection: CatsExistingCatDetection = field(
        default_factory=lambda: CatsExistingCatDetection(
            cats=(),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        )
    )


@dataclass(frozen=True, slots=True)
class CatsSolvedBoard:
    """Retain one logical solve result and its deterministic click plan."""

    board_input: CatsBoardInput
    logical_board: Board
    successful_applications: int
    click_plan: tuple[CatClickTarget, ...]
    status: CatsSolveStatus
    exact_search_result: CatsExactSearchResult | None = None
