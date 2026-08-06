"""Deterministic tests for puzzle-neutral LAB cell-color classification."""

from datetime import UTC, datetime
from math import dist
from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.config.settings import ColorDetectionSettings
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detection_renderer import (
    ColorDebugRenderError,
    OpenCvColorDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.color_detector import ColorDetectionError
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot
from synthetic_cats_tile_grids import (
    CATS_TILE_PALETTE,
    synthetic_cats_tile_grid,
)


def _colored_grid(
    color_indices: tuple[tuple[int, ...], ...],
    *,
    palette: tuple[tuple[int, int, int], ...],
    cell_size: int = 40,
    draw_symbols: bool = False,
) -> tuple[Screenshot, BoardDetection, GridDetection]:
    """Build exact cell geometry over a synthetic BGR color board."""

    rows = len(color_indices)
    columns = len(color_indices[0])
    board_x, board_y = 20, 24
    board_width = columns * cell_size
    board_height = rows * cell_size
    image = np.full(
        (board_y + board_height + 20, board_x + board_width + 20, 3),
        25,
        dtype=np.uint8,
    )
    horizontal_lines = tuple(board_y + index * cell_size for index in range(rows + 1))
    vertical_lines = tuple(board_x + index * cell_size for index in range(columns + 1))
    cells: list[CellBounds] = []
    for row in range(rows):
        for column in range(columns):
            left = vertical_lines[column]
            top = horizontal_lines[row]
            right = vertical_lines[column + 1]
            bottom = horizontal_lines[row + 1]
            image[top:bottom, left:right] = palette[color_indices[row][column]]
            if draw_symbols and (row + column) % 3 == 0:
                cv2.circle(
                    image,
                    ((left + right) // 2, (top + bottom) // 2),
                    5,
                    (15, 15, 15),
                    cv2.FILLED,
                    cv2.LINE_AA,
                )
            cells.append(
                CellBounds(
                    row=row,
                    column=column,
                    x=left,
                    y=top,
                    width=cell_size,
                    height=cell_size,
                    center_x=(left + right) // 2,
                    center_y=(top + bottom) // 2,
                )
            )
    for y in horizontal_lines:
        cv2.line(image, (board_x, y), (board_x + board_width, y), (5, 5, 5), 2)
    for x in vertical_lines:
        cv2.line(image, (x, board_y), (x, board_y + board_height), (5, 5, 5), 2)
    screenshot = Screenshot(
        image=image,
        width=image.shape[1],
        height=image.shape[0],
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    board = BoardDetection(
        x=board_x,
        y=board_y,
        width=board_width,
        height=board_height,
        confidence=0.95,
    )
    grid = GridDetection(
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        rows=rows,
        columns=columns,
        cells=tuple(cells),
        confidence=0.95,
    )
    return screenshot, board, grid


def _four_color_indices(rows: int = 8, columns: int = 8) -> tuple[tuple[int, ...], ...]:
    """Create a stable repeated pattern whose equality relations are explicit."""

    return tuple(
        tuple((row + column) % 4 for column in range(columns)) for row in range(rows)
    )


PALETTE = (
    (65, 85, 220),
    (75, 195, 85),
    (65, 205, 225),
    (210, 100, 95),
)


def _with_center_rectangle(
    screenshot: Screenshot,
    grid: GridDetection,
    *,
    cell_index: int,
    color: tuple[int, int, int],
) -> Screenshot:
    """Cover a large central cell region while preserving all corner patches."""

    image = screenshot.image.copy()
    cell = grid.cells[cell_index]
    left = cell.x + round(cell.width * 0.35)
    right = cell.x + round(cell.width * 0.65)
    top = cell.y + round(cell.height * 0.35)
    bottom = cell.y + round(cell.height * 0.65)
    cv2.rectangle(image, (left, top), (right, bottom), color, cv2.FILLED)
    return Screenshot(image, screenshot.width, screenshot.height, screenshot.timestamp)


def test_detects_four_logical_classes_across_complete_8x8_board() -> None:
    """Return 64 observations, a complete matrix, and puzzle-neutral class IDs."""

    indices = _four_color_indices()
    screenshot, _, grid = _colored_grid(indices, palette=PALETTE)

    result = OpenCvColorDetector().detect(screenshot, grid)

    assert len(result.observations) == 64
    assert result.color_count == 4
    assert len(result.color_matrix) == 8
    assert all(len(row) == 8 for row in result.color_matrix)
    assert {observation.color_id for observation in result.observations} == {
        "C0",
        "C1",
        "C2",
        "C3",
    }
    assert 0.0 <= result.mean_confidence <= 1.0
    for first_row in range(8):
        for first_column in range(8):
            for second_row in range(8):
                for second_column in range(8):
                    same_input_color = (
                        indices[first_row][first_column]
                        == indices[second_row][second_column]
                    )
                    same_class = (
                        result.color_matrix[first_row][first_column]
                        == result.color_matrix[second_row][second_column]
                    )
                    assert same_class is same_input_color


def test_similar_shades_group_together_without_fixed_human_color_names() -> None:
    """Group small BGR variations through LAB distance rather than palette lookup."""

    indices = ((0, 1, 0, 1), (1, 0, 1, 0), (0, 1, 0, 1))
    palette = ((70, 180, 90), (74, 184, 94))
    screenshot, _, grid = _colored_grid(indices, palette=palette)

    result = OpenCvColorDetector().detect(screenshot, grid)

    assert result.color_count == 1
    assert result.color_matrix == tuple(tuple("C0" for _ in row) for row in indices)


def test_robust_sampling_ignores_minority_symbol_like_strokes() -> None:
    """Keep background classes stable when dark center strokes are minority outliers."""

    indices = _four_color_indices(4, 4)
    clean, _, clean_grid = _colored_grid(indices, palette=PALETTE)
    marked, _, marked_grid = _colored_grid(
        indices,
        palette=PALETTE,
        draw_symbols=True,
    )

    clean_result = OpenCvColorDetector().detect(clean, clean_grid)
    marked_result = OpenCvColorDetector().detect(marked, marked_grid)

    assert marked_result.color_matrix == clean_result.color_matrix
    assert marked_result.color_count == clean_result.color_count == 4


def test_uniform_tile_corner_sampling_returns_expected_lab_color() -> None:
    """Represent a clean tile from four equal inset patches."""

    screenshot, _, grid = _colored_grid(((0,),), palette=PALETTE)

    result = OpenCvColorDetector().detect(screenshot, grid)

    expected = cv2.cvtColor(
        np.asarray([[PALETTE[0]]], dtype=np.uint8),
        cv2.COLOR_BGR2LAB,
    )[0, 0]
    assert all(
        abs(actual - float(wanted)) <= 1.0
        for actual, wanted in zip(
            result.observations[0].representative_lab,
            expected,
            strict=True,
        )
    )
    assert result.diagnostics.sample_pixel_counts == (100,)


@pytest.mark.parametrize("symbol_color", ((0, 0, 0), (255, 255, 255)))
def test_large_central_black_or_white_symbol_preserves_background_class(
    symbol_color: tuple[int, int, int],
) -> None:
    """Ignore a central light or dark overlay without recognizing its semantics."""

    indices = _four_color_indices(4, 4)
    clean, _, grid = _colored_grid(indices, palette=PALETTE)
    marked = _with_center_rectangle(
        clean,
        grid,
        cell_index=5,
        color=symbol_color,
    )
    detector = OpenCvColorDetector()

    clean_result = detector.detect(clean, grid)
    marked_result = detector.detect(marked, grid)

    assert marked_result.color_matrix == clean_result.color_matrix
    assert marked_result.color_count == clean_result.color_count == 4
    assert (
        dist(
            marked_result.observations[5].representative_lab,
            clean_result.observations[5].representative_lab,
        )
        <= 1.0
    )


def test_large_central_x_preserves_background_class() -> None:
    """Keep an X-shaped central exclusion outside the four corner samples."""

    indices = _four_color_indices(4, 4)
    clean, _, grid = _colored_grid(indices, palette=PALETTE)
    image = clean.image.copy()
    cell = grid.cells[6]
    left = cell.x + round(cell.width * 0.40)
    right = cell.x + round(cell.width * 0.60)
    top = cell.y + round(cell.height * 0.40)
    bottom = cell.y + round(cell.height * 0.60)
    cv2.line(image, (left, top), (right, bottom), (20, 20, 20), 4, cv2.LINE_AA)
    cv2.line(image, (right, top), (left, bottom), (20, 20, 20), 4, cv2.LINE_AA)
    marked = Screenshot(image, clean.width, clean.height, clean.timestamp)
    detector = OpenCvColorDetector()

    assert (
        detector.detect(marked, grid).color_matrix
        == detector.detect(
            clean,
            grid,
        ).color_matrix
    )


def test_one_contaminated_corner_is_rejected_before_final_median() -> None:
    """Let the other three matching patches determine the cell background."""

    screenshot, _, grid = _colored_grid(((0, 1), (1, 0)), palette=PALETTE)
    detector = OpenCvColorDetector()
    clean_result = detector.detect(screenshot, grid)
    image = screenshot.image.copy()
    left, top, right, bottom = detector._corner_sample_bounds(grid.cells[0])[0]
    image[top:bottom, left:right] = PALETTE[1]
    contaminated = Screenshot(
        image,
        screenshot.width,
        screenshot.height,
        screenshot.timestamp,
    )

    result = detector.detect(contaminated, grid)

    assert result.color_matrix == clean_result.color_matrix
    assert (
        dist(
            result.observations[0].representative_lab,
            clean_result.observations[0].representative_lab,
        )
        <= 1.0
    )
    assert result.diagnostics.within_cell_spreads[0] > (
        clean_result.diagnostics.within_cell_spreads[0]
    )


def test_two_strongly_contaminated_corners_fail_closed() -> None:
    """Reject a two-versus-two corner split instead of inventing a confident color."""

    screenshot, _, grid = _colored_grid(((0,),), palette=PALETTE)
    detector = OpenCvColorDetector()
    image = screenshot.image.copy()
    bounds = detector._corner_sample_bounds(grid.cells[0])
    for index in (0, 3):
        left, top, right, bottom = bounds[index]
        image[top:bottom, left:right] = PALETTE[1]
    contaminated = Screenshot(
        image,
        screenshot.width,
        screenshot.height,
        screenshot.timestamp,
    )

    with pytest.raises(ColorDetectionError, match="LAB consensus") as raised:
        detector.detect(contaminated, grid)

    assert "cell (0, 0)" in str(raised.value)


def test_corner_patch_geometry_avoids_rounded_gaps_and_cell_edges() -> None:
    """Place every sample inside a rounded tile rather than its surrounding gap."""

    fixture = synthetic_cats_tile_grid(rows=5, columns=5, include_ui=False)
    tile_grid = OpenCvCatsTileGridDetector().detect(fixture.screenshot)
    detector = OpenCvColorDetector()
    cell = tile_grid.grid.cells[0]
    expected_color = CATS_TILE_PALETTE[fixture.color_indices[0][0]]

    bounds = detector._corner_sample_bounds(cell)

    assert len(bounds) == 4
    for left, top, right, bottom in bounds:
        assert cell.x < left < right < cell.x + cell.width
        assert cell.y < top < bottom < cell.y + cell.height
        patch = fixture.screenshot.image[top:bottom, left:right]
        assert np.all(patch == np.asarray(expected_color, dtype=np.uint8))


@pytest.mark.parametrize(("width", "height"), ((6, 6), (30, 50), (59, 59)))
def test_corner_patch_bounds_are_positive_for_small_and_rectangular_cells(
    width: int,
    height: int,
) -> None:
    """Keep relative half-open patches positive, ordered, and inside each cell."""

    cell = CellBounds(0, 0, 11, 13, width, height, 11 + width // 2, 13 + height // 2)

    bounds = OpenCvColorDetector()._corner_sample_bounds(cell)

    assert len(bounds) == 4
    assert all(
        cell.x <= left < right <= cell.x + cell.width
        and cell.y <= top < bottom <= cell.y + cell.height
        for left, top, right, bottom in bounds
    )


def test_default_59_pixel_cell_uses_seven_pixel_patches_at_six_pixel_offset() -> None:
    """Lock the requested scale-relative example without hardcoding detector pixels."""

    cell = CellBounds(0, 0, 100, 200, 59, 59, 129, 229)

    assert OpenCvColorDetector()._corner_sample_bounds(cell) == (
        (106, 206, 113, 213),
        (146, 206, 153, 213),
        (106, 246, 113, 253),
        (146, 246, 153, 253),
    )


def test_cat_like_center_sprite_preserves_nine_color_equality_pattern() -> None:
    """Keep an animated multicolor center sprite from creating a tenth class."""

    clean = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        palette=CATS_TILE_PALETTE[:9],
        include_ui=False,
    )
    marked = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        palette=CATS_TILE_PALETTE[:9],
        include_ui=False,
        cat_sprite_slots=frozenset({(4, 4)}),
    )
    tile_detector = OpenCvCatsTileGridDetector()
    clean_grid = tile_detector.detect(clean.screenshot).grid
    marked_grid = tile_detector.detect(marked.screenshot).grid
    detector = OpenCvColorDetector()

    clean_result = detector.detect(clean.screenshot, clean_grid)
    marked_result = detector.detect(marked.screenshot, marked_grid)

    assert (marked_grid.rows, marked_grid.columns) == (9, 9)
    assert len(marked_grid.cells) == 81
    assert marked_result.color_count == 9
    assert marked_result.color_matrix == clean_result.color_matrix
    assert marked_result.color_matrix[4][4] == marked_result.color_matrix[3][5]


def test_mapping_and_diagnostics_are_deterministic_across_repeated_runs() -> None:
    """Produce byte-for-byte equivalent immutable records for identical pixels."""

    screenshot, _, grid = _colored_grid(_four_color_indices(), palette=PALETTE)
    detector = OpenCvColorDetector()

    first = detector.detect(screenshot, grid)
    second = detector.detect(screenshot, grid)

    assert first == second
    assert len(first.diagnostics.sample_pixel_counts) == 64
    assert len(first.diagnostics.within_cell_spreads) == 64
    assert len(first.diagnostics.cluster_centers_lab) == 4
    assert first.diagnostics.minimum_intercluster_distance is not None


def test_supports_rectangular_grid_and_row_major_matrix_construction() -> None:
    """Remain puzzle-neutral instead of hardcoding the current 8x8 board size."""

    indices = ((0, 1, 2), (2, 1, 0))
    screenshot, _, grid = _colored_grid(indices, palette=PALETTE)

    result = OpenCvColorDetector().detect(screenshot, grid)

    assert len(result.observations) == 6
    assert tuple((item.row, item.column) for item in result.observations) == (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
    )
    assert result.color_matrix[0][0] == result.color_matrix[1][2]
    assert result.color_matrix[0][2] == result.color_matrix[1][0]


def test_rejects_cell_geometry_outside_screenshot() -> None:
    """Fail closed rather than relying on NumPy's silently clipped slicing."""

    screenshot, _, grid = _colored_grid(((0, 1), (1, 0)), palette=PALETTE)
    smaller_image = screenshot.image[:-25, :-25].copy()
    smaller = Screenshot(
        image=smaller_image,
        width=smaller_image.shape[1],
        height=smaller_image.shape[0],
        timestamp=screenshot.timestamp,
    )

    with pytest.raises(ColorDetectionError, match="exceeds screenshot") as raised:
        OpenCvColorDetector().detect(smaller, grid)

    assert raised.value.diagnostics.rejection_reasons


def test_rejects_cells_that_cannot_supply_minimum_corner_sample() -> None:
    """Explain insufficient total corner evidence instead of fabricating color."""

    screenshot, _, grid = _colored_grid(
        ((0, 1), (1, 0)),
        palette=PALETTE,
        cell_size=6,
    )
    detector = OpenCvColorDetector(ColorDetectionSettings(minimum_sample_pixels=25))

    with pytest.raises(ColorDetectionError, match="reliable corner samples") as raised:
        detector.detect(screenshot, grid)

    assert "minimum is 25" in str(raised.value)


def test_invalid_color_settings_fail_at_composition_time() -> None:
    """Reject unsafe sampling, clustering, and confidence settings immediately."""

    with pytest.raises(ValueError):
        ColorDetectionSettings(sample_inner_fraction=0.0)
    with pytest.raises(ValueError):
        ColorDetectionSettings(outlier_trim_fraction=0.5)
    with pytest.raises(ValueError):
        ColorDetectionSettings(cluster_distance_threshold=0.0)
    with pytest.raises(ValueError):
        ColorDetectionSettings(maximum_within_cell_spread=0.0)
    with pytest.raises(ValueError):
        ColorDetectionSettings(minimum_sample_pixels=0)
    with pytest.raises(ValueError):
        ColorDetectionSettings(homogeneity_confidence_weight=1.1)
    with pytest.raises(ValueError):
        ColorDetectionSettings(cluster_fit_confidence_weight=-0.1)
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_patch_fraction=0.0)
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_patch_fraction=0.5)
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_offset_fraction=-0.1)
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_offset_fraction=float("nan"))
    with pytest.raises(ValueError):
        ColorDetectionSettings(
            corner_sample_patch_fraction=0.30,
            corner_sample_offset_fraction=0.25,
        )
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_minimum_consistent_patches=1)
    with pytest.raises(ValueError):
        ColorDetectionSettings(corner_sample_minimum_consistent_patches=5)


def test_color_confidence_weights_must_sum_to_one() -> None:
    """Prevent silently underweighted or overweighted public confidence."""

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        ColorDetectionSettings(
            homogeneity_confidence_weight=0.4,
            cluster_fit_confidence_weight=0.4,
        )


def test_debug_false_creates_no_directory_or_file(tmp_path: Path) -> None:
    """Keep normal detection and non-debug rendering free of persistence effects."""

    screenshot, board, grid = _colored_grid(_four_color_indices(), palette=PALETTE)
    result = OpenCvColorDetector().detect(screenshot, grid)
    destination = tmp_path / "missing" / "color_detection.png"

    saved = OpenCvColorDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        board,
        grid,
        result,
        destination,
        debug=False,
    )

    assert saved is None
    assert not destination.parent.exists()


def test_debug_true_creates_readable_labeled_overlay(tmp_path: Path) -> None:
    """Write a readable OpenCV image containing labels and representative swatches."""

    screenshot, board, grid = _colored_grid(_four_color_indices(), palette=PALETTE)
    result = OpenCvColorDetector().detect(screenshot, grid)
    destination = tmp_path / "color_detection.png"

    saved = OpenCvColorDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        board,
        grid,
        result,
        destination,
        debug=True,
    )

    assert saved == destination.resolve()
    overlay = cv2.imread(str(destination))
    assert overlay is not None
    assert overlay.shape == screenshot.image.shape
    assert not np.array_equal(overlay, screenshot.image)


def test_renderer_can_draw_exact_corner_sample_regions() -> None:
    """Expose four inset patches only when explicitly requested by debug tooling."""

    screenshot, board, grid = _colored_grid(_four_color_indices(), palette=PALETTE)
    result = OpenCvColorDetector().detect(screenshot, grid)
    renderer = OpenCvColorDetectionDebugRenderer()
    before = screenshot.image.copy()

    ordinary = renderer.render(screenshot, board, grid, result)
    diagnostic = renderer.render(
        screenshot,
        board,
        grid,
        result,
        draw_sample_regions=True,
    )

    assert not np.array_equal(diagnostic, ordinary)
    assert np.array_equal(screenshot.image, before)


def test_renderer_rejects_mismatched_grid_and_result() -> None:
    """Avoid silently pairing labels with unrelated cell geometry."""

    screenshot, board, grid = _colored_grid(_four_color_indices(), palette=PALETTE)
    result = OpenCvColorDetector().detect(screenshot, grid)
    smaller_screenshot, _, smaller_grid = _colored_grid(((0,),), palette=PALETTE)

    with pytest.raises(ColorDebugRenderError, match="matching counts"):
        OpenCvColorDetectionDebugRenderer().render(
            smaller_screenshot,
            board,
            smaller_grid,
            result,
        )
