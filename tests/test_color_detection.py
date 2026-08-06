"""Deterministic tests for puzzle-neutral LAB cell-color classification."""

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.config.settings import ColorDetectionSettings
from logicforge.infrastructure.opencv_color_detection_renderer import (
    ColorDebugRenderError,
    OpenCvColorDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.color_detector import ColorDetectionError
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot


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


def test_rejects_cells_that_cannot_supply_minimum_central_sample() -> None:
    """Explain resolution failure instead of fabricating a representative color."""

    screenshot, _, grid = _colored_grid(
        ((0, 1), (1, 0)),
        palette=PALETTE,
        cell_size=6,
    )
    detector = OpenCvColorDetector(ColorDetectionSettings(minimum_sample_pixels=25))

    with pytest.raises(ColorDetectionError, match="larger cell samples") as raised:
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
