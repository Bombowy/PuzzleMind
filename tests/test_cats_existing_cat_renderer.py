"""Explicit existing-cat debug overlay rendering regressions."""

from pathlib import Path

import numpy as np

from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_existing_cat_renderer import (
    OpenCvCatsExistingCatDebugRenderer,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.plugins.cats.existing_cat import CatsExistingCatDetection
from logicforge.plugins.cats.tile_grid import CatsTileGridDetection
from synthetic_cats_tile_grids import SyntheticCatsTileGrid, synthetic_cats_tile_grid


def _render_input() -> tuple[
    SyntheticCatsTileGrid,
    CatsTileGridDetection,
    CatsExistingCatDetection,
]:
    fixture = synthetic_cats_tile_grid(
        rows=6,
        columns=6,
        omitted_slots=frozenset({(1, 0)}),
        cat_sprite_slots=frozenset({(1, 0)}),
    )
    tile_grid = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    colors = OpenCvColorDetector().detect(fixture.screenshot, tile_grid.grid)
    existing = OpenCvCatsExistingCatDetector().detect(
        fixture.screenshot,
        tile_grid.grid,
        colors,
    )
    return fixture, tile_grid, existing


def test_renderer_returns_changed_copy_and_preserves_screenshot() -> None:
    fixture, tile_grid, existing = _render_input()
    before = fixture.screenshot.image.copy()
    overlay = OpenCvCatsExistingCatDebugRenderer().render(
        fixture.screenshot,
        tile_grid.grid,
        existing,
    )
    assert overlay.shape == before.shape
    assert not np.array_equal(overlay, before)
    assert np.array_equal(fixture.screenshot.image, before)


def test_renderer_persists_only_under_explicit_debug(
    tmp_path: Path,
) -> None:
    fixture, tile_grid, existing = _render_input()
    destination = tmp_path / "existing.png"
    renderer = OpenCvCatsExistingCatDebugRenderer()
    assert (
        renderer.save_debug_overlay(
            fixture.screenshot,
            tile_grid.grid,
            existing,
            destination,
            debug=False,
        )
        is None
    )
    assert not destination.exists()
    assert (
        renderer.save_debug_overlay(
            fixture.screenshot,
            tile_grid.grid,
            existing,
            destination,
            debug=True,
        )
        == destination.resolve()
    )
    assert destination.is_file()
