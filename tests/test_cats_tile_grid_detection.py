"""Tests for Cats tile-grid-first geometry and existing color integration."""

from datetime import UTC, datetime
from itertools import pairwise

import cv2
import numpy as np
import pytest

from logicforge.config.settings import CatsTileGridDetectionSettings
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.plugins.cats.tile_grid import CatsTileGridDetectionError
from logicforge.vision.screenshot import Screenshot
from synthetic_cats_tile_grids import (
    CATS_TILE_PALETTE,
    synthetic_cats_tile_grid,
)


@pytest.mark.parametrize("size", (5, 8, 9, 10))
def test_detects_complete_square_tile_lattices_without_outer_border(size: int) -> None:
    """Infer all cells from separated colored components for supported dimensions."""

    fixture = synthetic_cats_tile_grid(rows=size, columns=size)

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (result.grid.rows, result.grid.columns) == (size, size)
    assert len(result.grid.cells) == size * size
    assert len(result.grid.horizontal_lines) == size + 1
    assert len(result.grid.vertical_lines) == size + 1
    assert result.diagnostics.selected_tile_count == size * size
    assert result.diagnostics.occupancy_ratio == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("pastel_column", "pastel_row"),
    ((True, False), (False, True)),
)
def test_pastel_outer_tiles_remain_part_of_full_lattice(
    pastel_column: bool,
    pastel_row: bool,
) -> None:
    """Use LAB chroma alongside saturation for low-saturation edge tiles."""

    fixture = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        pastel_outer_column=pastel_column,
        pastel_outer_row=pastel_row,
    )

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (result.grid.rows, result.grid.columns) == (9, 9)
    assert result.diagnostics.selected_tile_count == 81


def test_board_bounds_are_half_pitch_extrapolation_not_tile_contour() -> None:
    """Include logical gaps around tile interiors in the public board geometry."""

    fixture = synthetic_cats_tile_grid(rows=9, columns=9)

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert result.board.x == fixture.board_x
    assert result.board.y == fixture.board_y
    assert result.board.width == fixture.board_width
    assert result.board.height == fixture.board_height
    first_component = next(
        component for component in result.diagnostics.components if component.accepted
    )
    assert result.board.x < first_component.x
    assert result.board.y < first_component.y


def test_contour_seed_can_truncate_pastel_edge_while_tile_lattice_stays_full() -> None:
    """Document the live architectural regression without removing generic vision."""

    fixture = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        pastel_outer_column=True,
    )

    contour_board = OpenCvBoardDetector().detect(fixture.screenshot)
    tile_result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert contour_board.width < fixture.board_width
    assert tile_result.board.width == fixture.board_width
    assert (tile_result.grid.rows, tile_result.grid.columns) == (9, 9)


def test_cells_tile_lattice_without_gaps_or_overlaps_and_track_centers() -> None:
    """Build row-major half-open cells while retaining fitted tile centers."""

    fixture = synthetic_cats_tile_grid(rows=9, columns=9)
    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    for row in range(9):
        row_cells = result.grid.cells[row * 9 : (row + 1) * 9]
        assert row_cells[0].x == result.board.x
        assert row_cells[-1].x + row_cells[-1].width == (
            result.board.x + result.board.width
        )
        assert all(
            first.x + first.width == second.x for first, second in pairwise(row_cells)
        )
        for column, cell in enumerate(row_cells):
            expected_x, expected_y = fixture.tile_centers[row][column]
            assert abs(cell.center_x - expected_x) <= 1
            assert abs(cell.center_y - expected_y) <= 1
            assert cell.width > 0
            assert cell.height > 0


def test_detection_is_deterministic_and_preserves_screenshot() -> None:
    """Return equal immutable records without modifying caller-owned pixels."""

    fixture = synthetic_cats_tile_grid(rows=10, columns=10)
    before = fixture.screenshot.image.copy()
    detector = OpenCvCatsTileGridDetector()

    assert detector.detect(fixture.screenshot) == detector.detect(fixture.screenshot)
    assert np.array_equal(fixture.screenshot.image, before)


def test_advertisement_and_ui_components_do_not_replace_supported_lattice() -> None:
    """Prefer a complete 2D family over isolated colorful rectangles and badges."""

    fixture = synthetic_cats_tile_grid(
        rows=8,
        columns=8,
        include_advertisement=True,
        include_ui=True,
    )

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (result.grid.rows, result.grid.columns) == (8, 8)
    assert result.board.x >= fixture.board_x
    assert result.diagnostics.component_count > 64


def test_advertisement_rectangles_alone_do_not_form_a_grid() -> None:
    """Reject colorful content that lacks repeated support on both axes."""

    image = np.full((720, 1100, 3), (230, 232, 235), dtype=np.uint8)
    for index in range(12):
        x = 20 + (index * 83) % 720
        y = 35 + (index * 127) % 590
        cv2.rectangle(
            image,
            (x, y),
            (x + 38 + index % 4 * 7, y + 27 + index % 3 * 9),
            CATS_TILE_PALETTE[index % len(CATS_TILE_PALETTE)],
            cv2.FILLED,
        )
    screenshot = Screenshot(image, 1100, 720, datetime(2026, 8, 6, tzinfo=UTC))

    with pytest.raises(CatsTileGridDetectionError):
        OpenCvCatsTileGridDetector().detect(screenshot)


def test_three_colored_buttons_do_not_form_a_board() -> None:
    """Require configured minimum row and column counts, not component color alone."""

    image = np.full((640, 480, 3), 235, dtype=np.uint8)
    for index, color in enumerate(CATS_TILE_PALETTE[:3]):
        cv2.rectangle(
            image, (95, 120 + index * 105), (385, 180 + index * 105), color, -1
        )
    screenshot = Screenshot(image, 480, 640, datetime(2026, 8, 6, tzinfo=UTC))

    with pytest.raises(CatsTileGridDetectionError):
        OpenCvCatsTileGridDetector().detect(screenshot)


def test_low_occupancy_fails_closed_without_interpolating_missing_tiles() -> None:
    """Reject a lattice whose real row support is far below the configured ratio."""

    fixture = synthetic_cats_tile_grid(
        rows=5,
        columns=5,
        omitted_slots=frozenset({(1, 0), (1, 1), (1, 2)}),
    )

    with pytest.raises(CatsTileGridDetectionError) as raised:
        OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert any(
        "support" in reason for reason in raised.value.diagnostics.rejection_reasons
    )


@pytest.mark.parametrize(
    "missing",
    ((0, 0), (0, 5), (5, 0), (5, 5), (2, 3)),
)
def test_one_missing_cartesian_slot_preserves_maximal_supported_six_by_six(
    missing: tuple[int, int],
) -> None:
    """Keep outer and interior axis centers when both axes have real support."""

    fixture = synthetic_cats_tile_grid(
        rows=6,
        columns=6,
        omitted_slots=frozenset({missing}),
    )

    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (detection.grid.rows, detection.grid.columns) == (6, 6)
    assert len(detection.grid.cells) == 36
    assert detection.diagnostics.selected_tile_count == 35
    assert detection.diagnostics.missing_slot_coordinates == (missing,)
    assert min(detection.diagnostics.row_support_ratios) == pytest.approx(5 / 6)
    assert min(detection.diagnostics.column_support_ratios) == pytest.approx(5 / 6)


def test_three_missing_cat_slots_keep_independently_supported_axes() -> None:
    """Several unique row/column omissions retain the maximal image-backed lattice."""

    missing = frozenset({(0, 1), (2, 3), (4, 5)})
    fixture = synthetic_cats_tile_grid(
        rows=6,
        columns=6,
        omitted_slots=missing,
    )

    detection = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (detection.grid.rows, detection.grid.columns) == (6, 6)
    assert len(detection.grid.cells) == 36
    assert detection.diagnostics.selected_tile_count == 33
    assert detection.diagnostics.missing_slot_coordinates == tuple(sorted(missing))


def test_irregular_row_and_column_pitch_are_rejected() -> None:
    """Fail closed when a mosaic cannot supply regular independent center pitches."""

    offsets = tuple(
        (row, column, column * column * 3, row * row * 3)
        for row in range(5)
        for column in range(5)
    )
    fixture = synthetic_cats_tile_grid(
        rows=5,
        columns=5,
        screenshot_width=1000,
        board_x=300,
        center_offsets=offsets,
    )

    with pytest.raises(CatsTileGridDetectionError):
        OpenCvCatsTileGridDetector().detect(fixture.screenshot)


def test_unstable_tile_sizes_do_not_form_a_size_family() -> None:
    """Reject components whose width and height variation exceeds the hard limit."""

    sizes = tuple(
        (row, column, 20 + 8 * column, 20 + 7 * row)
        for row in range(4)
        for column in range(4)
    )
    fixture = synthetic_cats_tile_grid(
        rows=4,
        columns=4,
        size_overrides=sizes,
    )

    with pytest.raises(CatsTileGridDetectionError):
        OpenCvCatsTileGridDetector().detect(fixture.screenshot)


@pytest.mark.parametrize(
    ("width", "height", "board_x"),
    ((1600, 900, 850), (600, 900, 110), (916, 1032, 330), (916, 1032, 20)),
)
def test_window_shape_and_horizontal_board_position_are_scale_independent(
    width: int,
    height: int,
    board_x: int,
) -> None:
    """Detect portrait, landscape, left-shifted, and right-shifted lattices."""

    fixture = synthetic_cats_tile_grid(
        rows=8,
        columns=8,
        screenshot_width=width,
        screenshot_height=height,
        board_x=board_x,
        board_y=190,
        include_advertisement=board_x >= 250,
    )

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (result.grid.rows, result.grid.columns) == (8, 8)


def test_rectangular_six_by_nine_lattice_has_no_square_assumption() -> None:
    """Keep geometry fitting independent of the final Cats puzzle invariant."""

    fixture = synthetic_cats_tile_grid(rows=6, columns=9)

    result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    assert (result.grid.rows, result.grid.columns) == (6, 9)
    assert len(result.grid.cells) == 54


def test_existing_color_detector_sees_exactly_nine_repeated_classes() -> None:
    """Give unchanged LAB clustering safe central samples from fitted cells."""

    fixture = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        palette=CATS_TILE_PALETTE[:9],
        include_ui=False,
    )
    tile_result = OpenCvCatsTileGridDetector().detect(fixture.screenshot)

    color_result = OpenCvColorDetector().detect(fixture.screenshot, tile_result.grid)

    assert color_result.color_count == 9
    assert len(color_result.color_matrix) == 9
    assert all(len(row) == 9 for row in color_result.color_matrix)
    for first_row in range(9):
        for first_column in range(9):
            for second_row in range(9):
                for second_column in range(9):
                    same_source = (
                        fixture.color_indices[first_row][first_column]
                        == fixture.color_indices[second_row][second_column]
                    )
                    same_class = (
                        color_result.color_matrix[first_row][first_column]
                        == color_result.color_matrix[second_row][second_column]
                    )
                    assert same_class is same_source


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tile_minimum_hsv_saturation", -1),
        ("tile_minimum_lab_chroma", float("nan")),
        ("tile_minimum_component_area_ratio", 0.0),
        ("tile_maximum_component_area_ratio", 1.1),
        ("tile_minimum_aspect_ratio", 1.1),
        ("tile_pitch_cv_maximum", 0.0),
        ("tile_grid_minimum_occupancy_ratio", 1.1),
        ("tile_grid_minimum_row_support_ratio", 0.0),
        ("tile_grid_minimum_column_support_ratio", float("nan")),
        ("tile_grid_minimum_rows", 1),
    ),
)
def test_settings_validation_rejects_invalid_values(field: str, value: object) -> None:
    """Keep every threshold finite and inside its documented semantic range."""

    with pytest.raises((TypeError, ValueError)):
        CatsTileGridDetectionSettings(**{field: value})  # type: ignore[arg-type]
