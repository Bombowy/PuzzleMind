"""Cats application screenshot-analysis orchestration tests."""

from types import SimpleNamespace
from typing import cast

import pytest

from cats_solve_test_support import (
    _board_detection,
    _CaptureService,
    _color_result,
    _grid_detection,
    _screenshot,
)
from logicforge.application.cats.analysis import analyze_captured_cats_board
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDetector,
    CatsExistingCatDiagnostics,
)
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetectionError,
    CatsTileGridDetector,
    CatsTileGridDiagnostics,
)
from logicforge.vision.board_detector import BoardDetection, BoardDetector
from logicforge.vision.color_detector import ColorDetectionResult, ColorDetector
from logicforge.vision.grid_detector import GridDetection, GridDetector
from logicforge.vision.screenshot import Screenshot


def test_analyze_captured_board_runs_each_fallback_stage_once_without_solving() -> None:
    """Run rejected tile, fallback geometry, color, and cat stages exactly once."""

    calls = {"tile": 0, "fallback": 0, "board": 0, "grid": 0, "color": 0, "cat": 0}

    class RejectingTileDetector:
        def detect(self, screenshot: Screenshot) -> None:
            assert screenshot is _CaptureService.screenshot
            calls["tile"] += 1
            raise CatsTileGridDetectionError(
                "synthetic tile rejection",
                cast(CatsTileGridDiagnostics, object()),
            )

    class CountingBoardDetector:
        def detect(self, screenshot: Screenshot) -> BoardDetection:
            assert screenshot is _CaptureService.screenshot
            calls["board"] += 1
            return _board_detection()

    class CountingGridDetector:
        def detect(
            self,
            screenshot: Screenshot,
            detected_board: BoardDetection,
        ) -> GridDetection:
            assert screenshot is _CaptureService.screenshot
            assert detected_board == _board_detection()
            calls["grid"] += 1
            return _grid_detection()

    class CountingColorDetector:
        def detect(
            self,
            screenshot: Screenshot,
            grid: GridDetection,
        ) -> ColorDetectionResult:
            assert screenshot is _CaptureService.screenshot
            assert grid == _grid_detection()
            calls["color"] += 1
            return _color_result()

    class CountingExistingCatDetector:
        def detect(
            self,
            screenshot: Screenshot,
            grid: GridDetection,
            colors: ColorDetectionResult,
        ) -> CatsExistingCatDetection:
            assert screenshot is _CaptureService.screenshot
            assert grid == _grid_detection()
            assert colors == _color_result()
            calls["cat"] += 1
            return CatsExistingCatDetection(
                cats=(),
                diagnostics=CatsExistingCatDiagnostics(cells=()),
            )

    def fallback() -> tuple[BoardDetector, GridDetector]:
        calls["fallback"] += 1
        return (
            cast(BoardDetector, CountingBoardDetector()),
            cast(GridDetector, CountingGridDetector()),
        )

    result = analyze_captured_cats_board(
        _CaptureService.screenshot,
        tile_grid_detector=cast(CatsTileGridDetector, RejectingTileDetector()),
        fallback_geometry_detectors=fallback,
        color_detector=cast(ColorDetector, CountingColorDetector()),
        existing_cat_detector=cast(
            CatsExistingCatDetector,
            CountingExistingCatDetector(),
        ),
    )

    assert calls == {
        "tile": 1,
        "fallback": 1,
        "board": 1,
        "grid": 1,
        "color": 1,
        "cat": 1,
    }
    assert result.detected_board == _board_detection()
    assert result.grid == _grid_detection()
    assert result.color_result == _color_result()


def test_analyze_captured_board_uses_tile_grid_primary_without_fallback() -> None:
    """Consume board and grid from one Cats lattice fit before color detection."""

    calls = {"tile": 0, "color": 0, "cat": 0}

    class TileDetector:
        def detect(self, screenshot: Screenshot) -> SimpleNamespace:
            calls["tile"] += 1
            return SimpleNamespace(board=_board_detection(), grid=_grid_detection())

    class ColorDetectorFake:
        def detect(
            self,
            screenshot: Screenshot,
            grid: GridDetection,
        ) -> ColorDetectionResult:
            calls["color"] += 1
            assert grid == _grid_detection()
            return _color_result()

    class ExistingCatDetectorFake:
        def detect(
            self,
            screenshot: Screenshot,
            grid: GridDetection,
            colors: ColorDetectionResult,
        ) -> CatsExistingCatDetection:
            calls["cat"] += 1
            return CatsExistingCatDetection(
                cats=(),
                diagnostics=CatsExistingCatDiagnostics(cells=()),
            )

    def forbidden_fallback() -> tuple[BoardDetector, GridDetector]:
        pytest.fail("generic contour geometry must not run after tile-grid success")

    result = analyze_captured_cats_board(
        _screenshot(),
        tile_grid_detector=cast(CatsTileGridDetector, TileDetector()),
        fallback_geometry_detectors=forbidden_fallback,
        color_detector=cast(ColorDetector, ColorDetectorFake()),
        existing_cat_detector=cast(
            CatsExistingCatDetector,
            ExistingCatDetectorFake(),
        ),
    )

    assert calls == {"tile": 1, "color": 1, "cat": 1}
    assert result.detected_board == _board_detection()
    assert result.grid == _grid_detection()
