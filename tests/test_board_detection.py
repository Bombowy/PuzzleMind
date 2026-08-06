"""Deterministic tests for classical puzzle-board localization and debug output."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.opencv_board_detection_renderer import (
    OpenCvBoardDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.vision.board_detector import BoardDetectionError
from logicforge.vision.screenshot import Screenshot


def _screenshot_from_image(image: np.ndarray) -> Screenshot:
    """Wrap one synthetic BGR frame in the same immutable model used in production."""

    height, width = image.shape[:2]
    return Screenshot(
        image=image,
        width=width,
        height=height,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _blank_screenshot(width: int = 800, height: int = 600) -> Screenshot:
    """Create a texture-free frame that cannot contain a credible board."""

    return _screenshot_from_image(np.full((height, width, 3), 32, dtype=np.uint8))


def _board_screenshot(
    rectangles: tuple[tuple[int, int, int, int], ...],
    *,
    width: int = 800,
    height: int = 600,
) -> Screenshot:
    """Draw scale-independent board-like cards with borders and internal grid edges."""

    image = np.full((height, width, 3), 32, dtype=np.uint8)
    for x, y, board_width, board_height in rectangles:
        right = x + board_width
        bottom = y + board_height
        cv2.rectangle(image, (x, y), (right, bottom), (225, 225, 225), -1)
        cv2.rectangle(image, (x, y), (right, bottom), (245, 245, 245), 4)
        for fraction in (0.25, 0.5, 0.75):
            grid_x = x + round(board_width * fraction)
            grid_y = y + round(board_height * fraction)
            cv2.line(image, (grid_x, y), (grid_x, bottom), (80, 80, 80), 2)
            cv2.line(image, (x, grid_y), (right, grid_y), (80, 80, 80), 2)
    return _screenshot_from_image(image)


def _custom_grid_screenshot(
    *,
    rows: int,
    columns: int,
    board_width: int = 400,
    board_height: int = 320,
    horizontal_positions: tuple[float, ...] | None = None,
    vertical_positions: tuple[float, ...] | None = None,
    horizontal_coverage: float = 1.0,
    vertical_coverage: float = 1.0,
) -> Screenshot:
    """Draw one configurable grid without exposing detector implementation details."""

    image = np.full((600, 800, 3), 32, dtype=np.uint8)
    x = (800 - board_width) // 2
    y = (600 - board_height) // 2
    right = x + board_width
    bottom = y + board_height
    cv2.rectangle(image, (x, y), (right, bottom), (225, 225, 225), -1)
    cv2.rectangle(image, (x, y), (right, bottom), (245, 245, 245), 6)

    row_separators = horizontal_positions or tuple(
        index / rows for index in range(1, rows)
    )
    column_separators = vertical_positions or tuple(
        index / columns for index in range(1, columns)
    )
    horizontal_margin = round(board_width * (1.0 - horizontal_coverage) / 2.0)
    vertical_margin = round(board_height * (1.0 - vertical_coverage) / 2.0)
    for position in row_separators:
        separator_y = y + round(board_height * position)
        cv2.line(
            image,
            (x + horizontal_margin, separator_y),
            (right - horizontal_margin, separator_y),
            (70, 70, 70),
            3,
        )
    for position in column_separators:
        separator_x = x + round(board_width * position)
        cv2.line(
            image,
            (separator_x, y + vertical_margin),
            (separator_x, bottom - vertical_margin),
            (70, 70, 70),
            3,
        )
    return _screenshot_from_image(image)


def _advertisement_like_screenshot() -> Screenshot:
    """Create a geometry-plausible text/image card with no regular internal grid."""

    image = np.full((600, 800, 3), 32, dtype=np.uint8)
    x, y, width, height = 200, 140, 400, 320
    cv2.rectangle(image, (x, y), (x + width, y + height), (230, 230, 230), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (250, 250, 250), 8)
    cv2.circle(image, (295, 250), 62, (110, 110, 110), 8)
    cv2.circle(image, (295, 250), 28, (170, 170, 170), -1)
    cv2.rectangle(image, (390, 205), (550, 235), (85, 85, 85), 3)
    cv2.rectangle(image, (390, 255), (520, 275), (100, 100, 100), 3)
    cv2.line(image, (245, 365), (560, 365), (90, 90, 90), 4)
    cv2.line(image, (260, 395), (470, 395), (120, 120, 120), 4)
    cv2.line(image, (515, 320), (565, 410), (80, 80, 80), 7)
    cv2.putText(
        image,
        "SPECIAL",
        (375, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (65, 65, 65),
        2,
        cv2.LINE_AA,
    )
    return _screenshot_from_image(image)


def test_detects_a_synthetic_board_rectangle() -> None:
    """Select the outer rectangle rather than internal grid cells or background."""

    screenshot = _board_screenshot(((220, 130, 330, 330),))

    detection = OpenCvBoardDetector().detect(screenshot)

    assert detection.x == pytest.approx(220, abs=6)
    assert detection.y == pytest.approx(130, abs=6)
    # The scale-relative edge envelope may add a small, resolution-dependent
    # perimeter around the visible border while preserving the complete board.
    assert detection.width == pytest.approx(330, abs=12)
    assert detection.height == pytest.approx(330, abs=12)
    assert 0.0 <= detection.confidence <= 1.0


def test_rejects_a_frame_without_a_board() -> None:
    """Raise the dedicated error instead of manufacturing a fallback rectangle."""

    with pytest.raises(BoardDetectionError, match="No board candidate") as raised:
        OpenCvBoardDetector().detect(_blank_screenshot())

    assert raised.value.diagnostics.selected_candidate is None


def test_rejects_tiny_rectangles() -> None:
    """Prevent buttons, icons, and small UI cards from being reported as boards."""

    screenshot = _board_screenshot(((360, 260, 60, 60),))

    with pytest.raises(BoardDetectionError):
        OpenCvBoardDetector().detect(screenshot)


def test_rejects_implausible_aspect_ratios() -> None:
    """Reject wide advertisement-like rectangles despite their significant area."""

    screenshot = _board_screenshot(((100, 230, 600, 100),))

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    assert any(
        "implausible aspect ratio" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_candidate_selection_is_deterministic_for_an_exact_tie() -> None:
    """Resolve equal scores through stable geometry ordering across contour runs."""

    settings = BoardDetectionSettings(
        minimum_relative_area=0.03,
        expected_center_x=0.50,
    )
    screenshot = _board_screenshot(
        (
            (100, 200, 200, 200),
            (500, 200, 200, 200),
        )
    )
    detector = OpenCvBoardDetector(settings)

    detections = tuple(detector.detect(screenshot) for _ in range(3))

    assert detections[0] == detections[1] == detections[2]
    assert detections[0].x == pytest.approx(100, abs=6)


def test_all_candidate_confidence_values_stay_in_unit_interval() -> None:
    """Keep both selected and rejected diagnostic scores safe for downstream use."""

    analysis = OpenCvBoardDetector().analyze(_board_screenshot(((220, 130, 330, 330),)))

    assert 0.0 <= analysis.detection.confidence <= 1.0
    assert analysis.diagnostics.candidates
    assert all(
        0.0 <= candidate.confidence <= 1.0
        for candidate in analysis.diagnostics.candidates
    )


def test_debug_false_does_not_write_to_the_filesystem(tmp_path: Path) -> None:
    """Prove normal debug-helper calls have no directory or image side effects."""

    screenshot = _board_screenshot(((220, 130, 330, 330),))
    analysis = OpenCvBoardDetector().analyze(screenshot)
    output_path = tmp_path / "missing" / "board_detection.png"

    result = OpenCvBoardDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        analysis,
        output_path,
        debug=False,
    )

    assert result is None
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_debug_true_creates_a_readable_overlay(tmp_path: Path) -> None:
    """Encode an explicit debug PNG that OpenCV can load at the source resolution."""

    screenshot = _board_screenshot(((220, 130, 330, 330),))
    analysis = OpenCvBoardDetector().analyze(screenshot)
    output_path = tmp_path / "vision" / "board_detection.png"

    saved_path = OpenCvBoardDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        analysis,
        output_path,
        debug=True,
        draw_rejected_candidates=True,
    )
    loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert saved_path == output_path.resolve()
    assert output_path.is_file()
    assert loaded is not None
    assert loaded.shape == screenshot.image.shape


def test_accepts_a_regular_rectangular_grid() -> None:
    """Accept non-square boards when both separator axes remain regular."""

    analysis = OpenCvBoardDetector().analyze(
        _custom_grid_screenshot(
            rows=4,
            columns=7,
            board_width=420,
            board_height=280,
        )
    )
    candidate = analysis.diagnostics.selected_candidate

    assert candidate is not None
    assert candidate.estimated_rows == 4
    assert candidate.estimated_columns == 7
    assert candidate.grid_evidence_score >= 0.65


@pytest.mark.parametrize("rows", [3, 5, 9])
def test_accepts_regular_grids_with_different_row_counts(rows: int) -> None:
    """Derive row quantity from boundaries instead of assuming an 8-row puzzle."""

    analysis = OpenCvBoardDetector().analyze(
        _custom_grid_screenshot(rows=rows, columns=6)
    )
    candidate = analysis.diagnostics.selected_candidate

    assert candidate is not None
    assert candidate.estimated_rows == rows
    assert candidate.estimated_columns == 6


@pytest.mark.parametrize("columns", [3, 5, 10])
def test_accepts_regular_grids_with_different_column_counts(columns: int) -> None:
    """Derive column quantity without a fixed puzzle-size expectation."""

    analysis = OpenCvBoardDetector().analyze(
        _custom_grid_screenshot(rows=6, columns=columns)
    )
    candidate = analysis.diagnostics.selected_candidate

    assert candidate is not None
    assert candidate.estimated_rows == 6
    assert candidate.estimated_columns == columns


def test_rejects_a_large_plain_rectangle_without_grid_evidence() -> None:
    """Require separators even when a large double-edged card passes geometry."""

    screenshot = _custom_grid_screenshot(rows=1, columns=1)
    settings = BoardDetectionSettings(minimum_edge_density=0.005)

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector(settings).detect(screenshot)

    assert any(
        "insufficient horizontal grid lines" in candidate.rejection_reasons
        and "insufficient vertical grid lines" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_advertisement_that_previously_passed_geometry_confidence() -> None:
    """Regress the 0.630-class false positive through mandatory regular-grid checks."""

    settings = BoardDetectionSettings()
    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector(settings).detect(_advertisement_like_screenshot())

    geometrically_plausible = tuple(
        candidate
        for candidate in raised.value.diagnostics.candidates
        if candidate.geometry_score >= settings.minimum_confidence
        and candidate.edge_density >= settings.minimum_edge_density
        and candidate.relative_area >= settings.minimum_relative_area
        and candidate.relative_area <= settings.maximum_relative_area
        and candidate.aspect_ratio >= settings.minimum_aspect_ratio
        and candidate.aspect_ratio <= settings.maximum_aspect_ratio
        and candidate.rectangularity >= settings.minimum_rectangularity
    )
    assert geometrically_plausible
    assert any(
        "grid evidence below required threshold" in candidate.rejection_reasons
        for candidate in geometrically_plausible
    )
    assert all(not candidate.accepted for candidate in geometrically_plausible)


def test_rejected_advertisement_is_visible_in_failure_debug_rendering() -> None:
    """Render red rejected candidates when no successful board analysis exists."""

    screenshot = _advertisement_like_screenshot()
    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    overlay = OpenCvBoardDetectionDebugRenderer().render_rejected_candidates(
        screenshot,
        raised.value.diagnostics,
    )
    rejected_color_pixels = (
        (overlay[:, :, 0] <= 120)
        & (overlay[:, :, 1] <= 120)
        & (overlay[:, :, 2] >= 180)
    )

    assert np.any(rejected_color_pixels)


def test_rejects_an_irregular_decorative_line_pattern() -> None:
    """Reject full-length decorative divisions whose spacing is not grid-regular."""

    screenshot = _custom_grid_screenshot(
        rows=4,
        columns=4,
        horizontal_positions=(0.12, 0.47, 0.86),
        vertical_positions=(0.10, 0.38, 0.88),
    )

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    assert any(
        "irregular horizontal grid spacing" in candidate.rejection_reasons
        or "irregular vertical grid spacing" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_too_few_horizontal_grid_lines() -> None:
    """Reject two-row content despite sufficient regular vertical separators."""

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(_custom_grid_screenshot(rows=2, columns=6))

    assert any(
        "insufficient horizontal grid lines" in candidate.rejection_reasons
        and "too few estimated rows" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_too_few_vertical_grid_lines() -> None:
    """Reject two-column content despite sufficient regular horizontal separators."""

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(_custom_grid_screenshot(rows=6, columns=2))

    assert any(
        "insufficient vertical grid lines" in candidate.rejection_reasons
        and "too few estimated columns" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_irregular_horizontal_spacing() -> None:
    """Apply horizontal spacing variation as a hard acceptance condition."""

    screenshot = _custom_grid_screenshot(
        rows=4,
        columns=6,
        horizontal_positions=(0.12, 0.50, 0.87),
    )

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    assert any(
        "irregular horizontal grid spacing" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_irregular_vertical_spacing() -> None:
    """Apply vertical spacing variation independently from horizontal evidence."""

    screenshot = _custom_grid_screenshot(
        rows=6,
        columns=4,
        vertical_positions=(0.11, 0.48, 0.89),
    )

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    assert any(
        "irregular vertical grid spacing" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_rejects_insufficient_internal_line_coverage() -> None:
    """Reject short separator fragments even when their positions are regular."""

    screenshot = _custom_grid_screenshot(
        rows=5,
        columns=5,
        horizontal_coverage=0.42,
        vertical_coverage=0.42,
    )

    with pytest.raises(BoardDetectionError) as raised:
        OpenCvBoardDetector().detect(screenshot)

    assert any(
        "insufficient horizontal grid coverage" in candidate.rejection_reasons
        or "insufficient vertical grid coverage" in candidate.rejection_reasons
        for candidate in raised.value.diagnostics.candidates
    )


def test_grid_diagnostics_and_scores_are_normalized() -> None:
    """Expose exact dimensions and bounded scores through primitive diagnostics."""

    analysis = OpenCvBoardDetector().analyze(_custom_grid_screenshot(rows=5, columns=7))
    selected = analysis.diagnostics.selected_candidate

    assert selected is not None
    assert selected.estimated_rows == 5
    assert selected.estimated_columns == 7
    assert selected.horizontal_grid_line_count == 6
    assert selected.vertical_grid_line_count == 8
    assert all(
        0.0 <= position <= 1.0 for position in selected.horizontal_grid_line_positions
    )
    assert all(
        0.0 <= position <= 1.0 for position in selected.vertical_grid_line_positions
    )
    assert 0.0 <= selected.grid_evidence_score <= 1.0
    assert 0.0 <= selected.confidence <= 1.0


def test_debug_overlay_contains_de_duplicated_grid_lines(tmp_path: Path) -> None:
    """Draw primitive separator diagnostics into an explicitly requested PNG."""

    screenshot = _custom_grid_screenshot(rows=5, columns=7)
    analysis = OpenCvBoardDetector().analyze(screenshot)
    output_path = tmp_path / "vision" / "board_detection.png"

    OpenCvBoardDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        analysis,
        output_path,
        debug=True,
        draw_grid_lines=True,
    )
    loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert loaded is not None
    horizontal_color_pixels = (
        (loaded[:, :, 0] >= 200) & (loaded[:, :, 1] >= 160) & (loaded[:, :, 2] <= 80)
    )
    vertical_color_pixels = (
        (loaded[:, :, 0] >= 180) & (loaded[:, :, 1] <= 150) & (loaded[:, :, 2] >= 180)
    )
    assert np.any(horizontal_color_pixels)
    assert np.any(vertical_color_pixels)


@pytest.mark.parametrize(
    "settings_factory",
    (
        lambda: BoardDetectionSettings(minimum_horizontal_grid_line_count=3),
        lambda: BoardDetectionSettings(minimum_vertical_grid_line_count=3),
        lambda: BoardDetectionSettings(minimum_estimated_rows=2),
        lambda: BoardDetectionSettings(minimum_estimated_columns=2),
        lambda: BoardDetectionSettings(
            maximum_horizontal_spacing_coefficient_of_variation=0.0
        ),
        lambda: BoardDetectionSettings(
            maximum_vertical_spacing_coefficient_of_variation=0.0
        ),
        lambda: BoardDetectionSettings(minimum_horizontal_line_coverage=0.0),
        lambda: BoardDetectionSettings(minimum_vertical_line_coverage=0.0),
        lambda: BoardDetectionSettings(minimum_grid_evidence_score=0.0),
        lambda: BoardDetectionSettings(grid_line_cluster_distance_relative=0.0),
        lambda: BoardDetectionSettings(grid_border_line_exclusion_tolerance=0.5),
        lambda: BoardDetectionSettings(horizontal_line_kernel_relative_length=0.0),
        lambda: BoardDetectionSettings(vertical_line_kernel_relative_length=0.0),
        lambda: BoardDetectionSettings(minimum_grid_line_response=0.0),
        lambda: BoardDetectionSettings(grid_adaptive_block_relative_size=0.0),
        lambda: BoardDetectionSettings(
            geometry_confidence_weight=0.60,
            grid_confidence_weight=0.40,
        ),
    ),
)
def test_grid_settings_reject_unsafe_values(
    settings_factory: Callable[[], BoardDetectionSettings],
) -> None:
    """Validate every grid-setting family before detector composition."""

    with pytest.raises(ValueError):
        settings_factory()


@pytest.mark.skip(reason="No anonymized local BlueStacks puzzle fixture is available.")
def test_detects_a_board_in_an_anonymized_bluestacks_fixture() -> None:
    """Reserve integration coverage until a safe puzzle screenshot is contributed."""
