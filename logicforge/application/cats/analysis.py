"""Cats screenshot analysis orchestrated exclusively through backend-neutral ports."""

from collections.abc import Callable

from logicforge.application.cats.models import CatsBoardInput
from logicforge.plugins.cats.existing_cat import CatsExistingCatDetector
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetectionError,
    CatsTileGridDetector,
)
from logicforge.vision.board_detector import BoardDetector
from logicforge.vision.color_detector import ColorDetector
from logicforge.vision.grid_detector import GridDetector
from logicforge.vision.screenshot import Screenshot

type FallbackGeometryDetectorFactory = Callable[[], tuple[BoardDetector, GridDetector]]


def analyze_captured_cats_board(
    screenshot: Screenshot,
    *,
    tile_grid_detector: CatsTileGridDetector,
    fallback_geometry_detectors: FallbackGeometryDetectorFactory,
    color_detector: ColorDetector,
    existing_cat_detector: CatsExistingCatDetector,
) -> CatsBoardInput:
    """Fit Cats tiles first, then classify colors and existing cats exactly once."""

    try:
        tile_grid = tile_grid_detector.detect(screenshot)
    except CatsTileGridDetectionError:
        fallback_board_detector, fallback_grid_detector = fallback_geometry_detectors()
        detected_board = fallback_board_detector.detect(screenshot)
        grid = fallback_grid_detector.detect(screenshot, detected_board)
    else:
        detected_board = tile_grid.board
        grid = tile_grid.grid
    color_result = color_detector.detect(screenshot, grid)
    existing_cat_detection = existing_cat_detector.detect(
        screenshot,
        grid,
        color_result,
    )
    return CatsBoardInput(
        detected_board=detected_board,
        grid=grid,
        color_result=color_result,
        existing_cat_detection=existing_cat_detection,
    )
