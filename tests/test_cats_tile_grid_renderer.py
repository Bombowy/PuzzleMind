"""Tests for Cats tile-grid debug rendering and explicit persistence."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_renderer import (
    CatsTileGridDebugRenderError,
    OpenCvCatsTileGridDebugRenderer,
)
from synthetic_cats_tile_grids import synthetic_cats_tile_grid


def test_overlay_draws_candidates_lattice_board_cells_and_metrics() -> None:
    """Make the fitted evidence visibly different from unchanged source pixels."""

    fixture = synthetic_cats_tile_grid(rows=5, columns=5)
    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    overlay = OpenCvCatsTileGridDebugRenderer().render(
        fixture.screenshot,
        detection,
    )

    assert overlay.shape == fixture.screenshot.image.shape
    assert not np.array_equal(overlay, fixture.screenshot.image)
    assert tuple(overlay[detection.board.y, detection.board.x]) != tuple(
        fixture.screenshot.image[detection.board.y, detection.board.x]
    )
    first_cell = detection.grid.cells[0]
    assert tuple(overlay[first_cell.center_y, first_cell.center_x]) != tuple(
        fixture.screenshot.image[first_cell.center_y, first_cell.center_x]
    )


def test_renderer_preserves_immutable_screenshot() -> None:
    """Draw exclusively on a copied array."""

    fixture = synthetic_cats_tile_grid(rows=8, columns=8)
    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    before = fixture.screenshot.image.copy()

    OpenCvCatsTileGridDebugRenderer().render(fixture.screenshot, detection)

    assert np.array_equal(fixture.screenshot.image, before)


def test_debug_false_does_not_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep persistence explicitly opt-in."""

    fixture = synthetic_cats_tile_grid(rows=5, columns=5)
    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    writes: list[str] = []

    def write(path: str, image: np.ndarray) -> bool:
        del image
        writes.append(path)
        return True

    monkeypatch.setattr(cv2, "imwrite", write)

    result = OpenCvCatsTileGridDebugRenderer().save_debug_overlay(
        fixture.screenshot,
        detection,
        tmp_path / "unused.png",
        debug=False,
    )

    assert result is None
    assert writes == []


def test_debug_true_writes_exactly_one_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Encode the composed overlay once and expose its absolute path."""

    fixture = synthetic_cats_tile_grid(rows=5, columns=5)
    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    writes: list[str] = []

    def write(path: str, image: np.ndarray) -> bool:
        del image
        writes.append(path)
        return True

    monkeypatch.setattr(cv2, "imwrite", write)
    destination = tmp_path / "cats-grid.png"

    result = OpenCvCatsTileGridDebugRenderer().save_debug_overlay(
        fixture.screenshot,
        detection,
        destination,
        debug=True,
    )

    assert result == destination.resolve()
    assert writes == [str(destination.resolve())]


def test_failed_encoding_raises_typed_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not report a nonexistent artifact after OpenCV declines encoding."""

    fixture = synthetic_cats_tile_grid(rows=5, columns=5)
    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: False)

    with pytest.raises(CatsTileGridDebugRenderError):
        OpenCvCatsTileGridDebugRenderer().save_debug_overlay(
            fixture.screenshot,
            detection,
            tmp_path / "failed.png",
            debug=True,
        )
