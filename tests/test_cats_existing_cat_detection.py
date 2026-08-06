"""Classical per-CellBounds existing-cat detection regressions."""

from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from logicforge.config.settings import CatsExistingCatDetectionSettings
from logicforge.core import Board
from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.plugins.cats.board_actions import place_cat
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
)
from logicforge.plugins.cats.tile_grid import CatsTileGridDetection
from logicforge.vision.color_detector import ColorDetectionResult
from logicforge.vision.screenshot import Screenshot
from synthetic_cats_tile_grids import SyntheticCatsTileGrid, synthetic_cats_tile_grid


def _pipeline(
    *,
    cats: frozenset[tuple[int, int]] = frozenset(),
    screenshot_width: int = 916,
    screenshot_height: int = 1032,
    pitch_x: int = 53,
    pitch_y: int = 53,
    tile_width: int = 47,
    tile_height: int = 47,
    omit_cat_tiles: bool = True,
) -> tuple[
    SyntheticCatsTileGrid,
    CatsTileGridDetection,
    ColorDetectionResult,
    CatsExistingCatDetection,
]:
    fixture = synthetic_cats_tile_grid(
        rows=6,
        columns=6,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        board_y=max(80, screenshot_height // 4),
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        tile_width=tile_width,
        tile_height=tile_height,
        omitted_slots=cats if omit_cat_tiles else frozenset(),
        cat_sprite_slots=cats,
    )
    tile_grid = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    colors = OpenCvColorDetector().detect(fixture.screenshot, tile_grid.grid)
    existing = OpenCvCatsExistingCatDetector().detect(
        fixture.screenshot,
        tile_grid.grid,
        colors,
    )
    return fixture, tile_grid, colors, existing


def _with_symbol(
    kind: str, color: tuple[int, int, int]
) -> tuple[Screenshot, CatsTileGridDetection, ColorDetectionResult]:
    fixture = synthetic_cats_tile_grid(rows=6, columns=6)
    tile_grid = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    colors = OpenCvColorDetector().detect(fixture.screenshot, tile_grid.grid)
    image = fixture.screenshot.image.copy()
    center_x, center_y = fixture.tile_centers[2][3]
    pitch_x = fixture.vertical_lines[1] - fixture.vertical_lines[0]
    pitch_y = fixture.horizontal_lines[1] - fixture.horizontal_lines[0]
    if kind == "x":
        extent = round(min(pitch_x, pitch_y) * 0.28)
        thickness = max(1, round(min(pitch_x, pitch_y) * 0.055))
        cv2.line(
            image,
            (center_x - extent, center_y - extent),
            (center_x + extent, center_y + extent),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            (center_x + extent, center_y - extent),
            (center_x - extent, center_y + extent),
            color,
            thickness,
            cv2.LINE_AA,
        )
    elif kind == "small":
        cv2.circle(
            image,
            (center_x, center_y),
            max(2, round(min(pitch_x, pitch_y) * 0.09)),
            color,
            cv2.FILLED,
        )
    elif kind == "edge":
        left = fixture.vertical_lines[3] + round(pitch_x * 0.08)
        top = fixture.horizontal_lines[2] + round(pitch_y * 0.08)
        cv2.rectangle(
            image,
            (left, top),
            (left + round(pitch_x * 0.35), top + round(pitch_y * 0.82)),
            color,
            cv2.FILLED,
        )
    elif kind == "border":
        left = fixture.vertical_lines[3] + 1
        right = fixture.vertical_lines[4] - 2
        top = fixture.horizontal_lines[2] + 1
        bottom = fixture.horizontal_lines[3] - 2
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            color,
            max(1, round(min(pitch_x, pitch_y) * 0.055)),
        )
    screenshot = Screenshot(
        image=image,
        width=fixture.screenshot.width,
        height=fixture.screenshot.height,
        timestamp=datetime(2026, 8, 7, tzinfo=UTC),
    )
    return screenshot, tile_grid, colors


def test_clean_board_has_no_existing_cats() -> None:
    _, _, _, existing = _pipeline()
    assert existing.cats == ()


def test_one_center_cat_like_sprite_has_exact_coordinate() -> None:
    _, grid, colors, existing = _pipeline(cats=frozenset({(1, 0)}))
    assert (grid.grid.rows, grid.grid.columns) == (6, 6)
    assert grid.diagnostics.missing_slot_coordinates == ((1, 0),)
    assert colors.color_count == 6
    assert tuple((cat.row, cat.column) for cat in existing.cats) == ((1, 0),)


def test_full_started_board_pipeline_places_existing_cat_on_single_board() -> None:
    """Regress the live-like 6x6/35-component case through logical initialization."""

    _, tile_grid, colors, existing = _pipeline(cats=frozenset({(1, 0)}))
    board = Board(colors)
    for cat in existing.cats:
        place_cat(board, cat.row, cat.column)

    assert (tile_grid.grid.rows, tile_grid.grid.columns) == (6, 6)
    assert len(tile_grid.grid.cells) == 36
    assert tile_grid.diagnostics.selected_tile_count == 35
    assert tile_grid.diagnostics.missing_slot_coordinates == ((1, 0),)
    assert colors.color_count == 6
    assert board.get(1, 0) == "K"
    assert all(board.get(1, column) == "X" for column in range(1, 6))
    assert all(board.get(row, 0) == "X" for row in range(6) if row != 1)


def test_resized_board_preserves_logical_coordinate() -> None:
    _, _, _, existing = _pipeline(
        cats=frozenset({(1, 0)}),
        screenshot_width=1200,
        screenshot_height=1100,
        pitch_x=68,
        pitch_y=68,
        tile_width=60,
        tile_height=60,
    )
    assert tuple((cat.row, cat.column) for cat in existing.cats) == ((1, 0),)


def test_rectangular_cell_bounds_use_percentage_roi() -> None:
    _, grid, _, existing = _pipeline(
        cats=frozenset({(1, 0)}),
        screenshot_width=1200,
        screenshot_height=1000,
        pitch_x=76,
        pitch_y=62,
        tile_width=68,
        tile_height=54,
    )
    cell = grid.grid.cells[6]
    diagnostic = existing.diagnostics.cells[6]
    assert cell.width != cell.height
    assert diagnostic.roi_width == cell.width - 2 * round(cell.width * 0.08)
    assert diagnostic.roi_height == cell.height - 2 * round(cell.height * 0.06)
    assert tuple((cat.row, cat.column) for cat in existing.cats) == ((1, 0),)


@pytest.mark.parametrize("color", ((0, 0, 0), (255, 255, 255)))
def test_central_x_is_not_a_cat(color: tuple[int, int, int]) -> None:
    screenshot, tile_grid, colors = _with_symbol("x", color)
    existing = OpenCvCatsExistingCatDetector().detect(
        screenshot, tile_grid.grid, colors
    )
    assert existing.cats == ()
    diagnostic = existing.diagnostics.cells[2 * 6 + 3]
    assert (
        diagnostic.foreground_ratio < 0.26 or diagnostic.largest_component_ratio < 0.24
    )


def test_yellow_selection_border_is_not_a_cat() -> None:
    screenshot, tile_grid, colors = _with_symbol("border", (20, 230, 245))
    existing = OpenCvCatsExistingCatDetector().detect(
        screenshot, tile_grid.grid, colors
    )
    assert existing.cats == ()


def test_small_dark_icon_is_not_a_cat() -> None:
    screenshot, tile_grid, colors = _with_symbol("small", (15, 15, 15))
    existing = OpenCvCatsExistingCatDetector().detect(
        screenshot, tile_grid.grid, colors
    )
    assert existing.cats == ()


def test_large_edge_localized_foreground_is_not_a_cat() -> None:
    screenshot, tile_grid, colors = _with_symbol("edge", (15, 15, 15))
    existing = OpenCvCatsExistingCatDetector().detect(
        screenshot, tile_grid.grid, colors
    )
    assert existing.cats == ()
    assert existing.diagnostics.cells[15].center_offset_ratio > 0.18


def test_large_central_multicolor_foreground_is_a_cat() -> None:
    _, _, _, existing = _pipeline(cats=frozenset({(3, 2)}))
    assert tuple((cat.row, cat.column) for cat in existing.cats) == ((3, 2),)


def test_multiple_valid_existing_cats_are_all_detected() -> None:
    coordinates = frozenset({(0, 1), (2, 3), (4, 5)})
    _, _, _, existing = _pipeline(cats=coordinates)
    assert tuple((cat.row, cat.column) for cat in existing.cats) == tuple(
        sorted(coordinates)
    )


@pytest.mark.parametrize(
    "coordinates,reason",
    (
        (frozenset({(0, 0), (0, 3)}), "row"),
        (frozenset({(0, 0), (3, 0)}), "column"),
        (frozenset({(0, 0), (2, 4)}), "original color"),
        (frozenset({(0, 0), (1, 1)}), "touch"),
    ),
)
def test_contradictory_existing_cats_fail_closed(
    coordinates: frozenset[tuple[int, int]],
    reason: str,
) -> None:
    with pytest.raises(CatsExistingCatDetectionError) as raised:
        _pipeline(
            cats=coordinates,
            omit_cat_tiles=reason not in {"row", "column"},
        )
    assert reason in str(raised.value)


def test_screenshot_is_immutable_and_detection_is_deterministic() -> None:
    fixture = synthetic_cats_tile_grid(
        rows=6,
        columns=6,
        omitted_slots=frozenset({(1, 0)}),
        cat_sprite_slots=frozenset({(1, 0)}),
    )
    before = fixture.screenshot.image.copy()
    tile_grid = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    colors = OpenCvColorDetector().detect(fixture.screenshot, tile_grid.grid)
    detector = OpenCvCatsExistingCatDetector()
    first = detector.detect(fixture.screenshot, tile_grid.grid, colors)
    second = detector.detect(fixture.screenshot, tile_grid.grid, colors)
    assert first == second
    assert np.array_equal(fixture.screenshot.image, before)
    assert not fixture.screenshot.image.flags.writeable


def test_malformed_grid_color_shape_raises_typed_error() -> None:
    fixture6 = synthetic_cats_tile_grid(rows=6, columns=6)
    fixture5 = synthetic_cats_tile_grid(rows=5, columns=5)
    grid6 = OpenCvCatsTileGridDetector().detect(fixture6.screenshot).grid
    grid5 = OpenCvCatsTileGridDetector().detect(fixture5.screenshot).grid
    colors5 = OpenCvColorDetector().detect(fixture5.screenshot, grid5)
    with pytest.raises(CatsExistingCatDetectionError):
        OpenCvCatsExistingCatDetector().detect(fixture6.screenshot, grid6, colors5)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cat_roi_horizontal_inset_ratio", 0.5),
        ("cat_roi_vertical_inset_ratio", -0.01),
        ("cat_foreground_lab_distance_threshold", 0.0),
        ("cat_mask_kernel_relative_size", 0.0),
        ("cat_minimum_foreground_ratio", 1.1),
        ("cat_minimum_largest_component_ratio", float("nan")),
        ("cat_minimum_component_width_ratio", 0.0),
        ("cat_minimum_component_height_ratio", -0.1),
        ("cat_maximum_center_offset_ratio", 1.1),
        ("cat_minimum_score", 0.0),
    ),
)
def test_existing_cat_settings_reject_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        CatsExistingCatDetectionSettings(**{field: value})  # type: ignore[arg-type]


def test_existing_cat_score_uses_documented_weighted_formula() -> None:
    _, _, _, existing = _pipeline(cats=frozenset({(1, 0)}))
    diagnostic = existing.diagnostics.cells[6]
    expected = (
        0.25 * diagnostic.foreground_ratio
        + 0.25 * diagnostic.largest_component_ratio
        + 0.20 * diagnostic.component_width_ratio
        + 0.20 * diagnostic.component_height_ratio
        + 0.10 * (1.0 - diagnostic.center_offset_ratio)
    )
    assert diagnostic.score == pytest.approx(expected)
