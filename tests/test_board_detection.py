"""Deterministic tests for classical puzzle-board localization and debug output."""

from collections.abc import Callable
from datetime import UTC, datetime
from inspect import getsource
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.opencv_board_detection_renderer import (
    OpenCvBoardDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_grid_envelope_refinement import (
    OpenCvGridEnvelopeRefiner,
    _GridEnvelopeRefinementCandidate,
)
from logicforge.infrastructure.opencv_internal_grid_evidence import (
    InternalGridEvidence,
    OpenCvInternalGridEvidenceAnalyzer,
)
from logicforge.vision.board_detector import (
    BoardDetectionError,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.screenshot import Screenshot
from synthetic_vision import (
    advertisement_like_screenshot as _advertisement_like_screenshot,
)
from synthetic_vision import custom_grid_screenshot as _custom_grid_screenshot
from synthetic_vision import (
    live_like_9x9_weak_grid_case,
    truncated_outer_grid_envelope_screenshot,
)


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


def test_board_detector_accepts_live_like_9x9_with_one_weak_separator() -> None:
    """Use the shared internal analyzer to retain a correct board candidate."""

    screenshot, _ = live_like_9x9_weak_grid_case(4)

    analysis = OpenCvBoardDetector().analyze(screenshot)
    selected = analysis.diagnostics.selected_candidate

    assert selected is not None
    assert selected.estimated_rows == 9
    assert selected.estimated_columns == 9
    assert selected.horizontal_grid_line_count == 10
    assert selected.vertical_grid_line_count == 10
    assert selected.grid_evidence_score >= 0.65


def test_refines_low_contrast_rightmost_column_to_full_9x9_envelope() -> None:
    """Recover the live-like ninth column that lies outside a regular 9x8 seed."""

    screenshot, expected = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
    )
    original = screenshot.image.copy()

    analyses = tuple(OpenCvBoardDetector().analyze(screenshot) for _ in range(3))
    analysis = analyses[0]
    seed = analysis.diagnostics.selected_candidate
    refinement = analysis.diagnostics.selected_refinement

    assert analyses[0] == analyses[1] == analyses[2]
    assert seed is not None
    assert refinement is not None
    assert (seed.estimated_rows, seed.estimated_columns) == (9, 8)
    assert refinement.direction == "right"
    assert (refinement.refined_rows, refinement.refined_columns) == (9, 9)
    assert refinement.old_border_match_score > 0.0
    assert refinement.supported_separator_fraction >= 0.65
    assert refinement.separator_continuation_score >= 0.12
    assert refinement.accepted
    assert analysis.detection.x <= expected.x
    assert analysis.detection.x + analysis.detection.width >= (
        expected.x + expected.width
    )
    assert seed.x + seed.width < analysis.detection.x + analysis.detection.width
    assert np.array_equal(screenshot.image, original)


def test_contrastive_outer_column_is_detected_without_envelope_refinement() -> None:
    """Keep a complete contour-derived 9x9 candidate unchanged."""

    screenshot, expected = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
        low_contrast_outer_band=False,
    )

    analysis = OpenCvBoardDetector().analyze(screenshot)
    selected = analysis.diagnostics.selected_candidate

    assert selected is not None
    assert (selected.estimated_rows, selected.estimated_columns) == (9, 9)
    assert analysis.diagnostics.selected_refinement is None
    assert analysis.detection.x == pytest.approx(expected.x, abs=8)
    assert analysis.detection.width == pytest.approx(expected.width, abs=12)


@pytest.mark.parametrize(
    ("side", "expected_direction", "seed_dimensions"),
    (
        ("left", "left", (9, 8)),
        ("top", "top", (8, 9)),
        ("bottom", "bottom", (8, 9)),
    ),
)
def test_refines_one_low_contrast_outer_band_on_other_sides(
    side: str,
    expected_direction: str,
    seed_dimensions: tuple[int, int],
) -> None:
    """Apply one generic extension without a hard-coded right-side preference."""

    screenshot, expected = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side=side,
    )

    analysis = OpenCvBoardDetector().analyze(screenshot)
    seed = analysis.diagnostics.selected_candidate
    refinement = analysis.diagnostics.selected_refinement

    assert seed is not None
    assert refinement is not None
    assert (seed.estimated_rows, seed.estimated_columns) == seed_dimensions
    assert refinement.direction == expected_direction
    assert (refinement.refined_rows, refinement.refined_columns) == (9, 9)
    assert analysis.detection.x <= expected.x
    assert analysis.detection.y <= expected.y
    assert (
        analysis.detection.x + analysis.detection.width >= expected.x + expected.width
    )
    assert analysis.detection.y + analysis.detection.height >= (
        expected.y + expected.height
    )


def test_real_rectangular_9x8_grid_remains_9x8() -> None:
    """Do not use non-square geometry itself as extension evidence."""

    analysis = OpenCvBoardDetector().analyze(
        _custom_grid_screenshot(rows=9, columns=8, board_width=480, board_height=420)
    )
    selected = analysis.diagnostics.selected_candidate

    assert selected is not None
    assert (selected.estimated_rows, selected.estimated_columns) == (9, 8)
    assert analysis.diagnostics.selected_refinement is None


@pytest.mark.parametrize(
    ("continuation", "panel"),
    ((False, False), (False, True)),
)
def test_does_not_extend_into_empty_background_or_unrelated_panel(
    continuation: bool,
    panel: bool,
) -> None:
    """Require aligned separator continuation rather than color or extra area."""

    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
        continue_orthogonal_separators=continuation,
        unrelated_external_panel=panel,
    )

    analysis = OpenCvBoardDetector().analyze(screenshot)
    selected = analysis.diagnostics.selected_candidate
    right_attempts = tuple(
        attempt
        for attempt in analysis.diagnostics.envelope_refinements
        if attempt.direction == "right"
        and selected is not None
        and attempt.seed_x == selected.x
        and attempt.seed_y == selected.y
    )

    assert selected is not None
    assert (selected.estimated_rows, selected.estimated_columns) == (9, 8)
    assert analysis.diagnostics.selected_refinement is None
    assert right_attempts
    assert all(not attempt.accepted for attempt in right_attempts)
    assert any(
        "orthogonal separators" in reason
        for attempt in right_attempts
        for reason in attempt.rejection_reasons
    )


def test_envelope_refinement_can_be_disabled_without_changing_seed() -> None:
    """Expose an explicit switch that preserves the contour-derived 9x8 result."""

    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
    )
    analysis = OpenCvBoardDetector(
        BoardDetectionSettings(grid_envelope_refinement_enabled=False)
    ).analyze(screenshot)
    selected = analysis.diagnostics.selected_candidate

    assert selected is not None
    assert (selected.estimated_rows, selected.estimated_columns) == (9, 8)
    assert tuple(
        attempt.direction
        for attempt in analysis.diagnostics.envelope_refinements
        if attempt.seed_x == selected.x and attempt.seed_y == selected.y
    ) == ("left", "right", "top", "bottom")
    assert all(
        not attempt.accepted and "refinement disabled" in attempt.rejection_reasons
        for attempt in analysis.diagnostics.envelope_refinements
        if attempt.seed_x == selected.x and attempt.seed_y == selected.y
    )
    assert analysis.diagnostics.selected_refinement is None


def test_internal_line_recovery_switch_does_not_disable_envelope_refinement() -> None:
    """Keep internal-line and envelope recovery independently configurable."""

    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
    )
    analysis = OpenCvBoardDetector(
        BoardDetectionSettings(grid_missing_line_recovery_enabled=False)
    ).analyze(screenshot)

    assert analysis.diagnostics.selected_refinement is not None
    assert (
        analysis.diagnostics.selected_refinement.refined_rows,
        analysis.diagnostics.selected_refinement.refined_columns,
    ) == (9, 9)


def test_generic_envelope_refiner_has_no_cats_color_or_square_assumptions() -> None:
    """Keep envelope vision puzzle-neutral and independent of public grid API."""

    source = getsource(OpenCvGridEnvelopeRefiner).casefold()

    assert "cats" not in source
    assert "color_count" not in source
    assert "rows == columns" not in source
    assert "opencvgriddetector" not in source


@pytest.mark.parametrize(
    ("board_x", "board_y", "expected_reason"),
    (
        (525, 110, "outside content area"),
        (230, 40, "outside content area"),
        (455, 110, "outside screenshot"),
    ),
)
def test_refinement_respects_content_and_screenshot_boundaries(
    board_x: int,
    board_y: int,
    expected_reason: str,
) -> None:
    """Reject candidates entering the titlebar, toolbar, or capture boundary."""

    screenshot_width = 1100 if board_x == 525 else 1000
    if expected_reason == "outside screenshot":
        screenshot_width = 995
    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right" if board_y == 110 else "top",
        screenshot_width=screenshot_width,
        board_x=board_x,
        board_y=board_y,
    )

    analysis = OpenCvBoardDetector().analyze(screenshot)

    assert analysis.diagnostics.selected_refinement is None
    assert any(
        expected_reason in reason
        for attempt in analysis.diagnostics.envelope_refinements
        for reason in attempt.rejection_reasons
    )


def test_added_size_score_rejects_half_and_two_cell_bands() -> None:
    """Keep added-size evidence independent from color, area, and aspect ratio."""

    settings = BoardDetectionSettings()
    refiner = OpenCvGridEnvelopeRefiner(
        settings,
        OpenCvInternalGridEvidenceAnalyzer(settings),
    )

    assert refiner._added_size_score(0.5) == 0.0
    assert refiner._added_size_score(1.0) == 1.0
    assert refiner._added_size_score(2.0) == 0.0


def test_equal_opposite_refinements_fail_closed_as_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the seed when different directions have scores inside ambiguity delta."""

    screenshot = _custom_grid_screenshot(rows=5, columns=5)
    analysis = OpenCvBoardDetector(
        BoardDetectionSettings(grid_envelope_refinement_enabled=False)
    ).analyze(screenshot)
    seed = analysis.diagnostics.selected_candidate
    assert seed is not None
    evidence = InternalGridEvidence(
        horizontal_line_positions=seed.horizontal_grid_line_positions,
        vertical_line_positions=seed.vertical_grid_line_positions,
        horizontal_line_count=seed.horizontal_grid_line_count,
        vertical_line_count=seed.vertical_grid_line_count,
        estimated_rows=seed.estimated_rows,
        estimated_columns=seed.estimated_columns,
        horizontal_spacing_coefficient_of_variation=(
            seed.horizontal_spacing_coefficient_of_variation
        ),
        vertical_spacing_coefficient_of_variation=(
            seed.vertical_spacing_coefficient_of_variation
        ),
        horizontal_spacing_regularity=seed.horizontal_spacing_regularity,
        vertical_spacing_regularity=seed.vertical_spacing_regularity,
        horizontal_line_coverage=seed.horizontal_line_coverage,
        vertical_line_coverage=seed.vertical_line_coverage,
        score=seed.grid_evidence_score,
    )
    settings = BoardDetectionSettings()
    refiner = OpenCvGridEnvelopeRefiner(
        settings,
        OpenCvInternalGridEvidenceAnalyzer(settings),
    )

    def fake_evaluate(
        *args: object, **kwargs: object
    ) -> _GridEnvelopeRefinementCandidate:
        del args
        direction = str(kwargs["direction"])
        accepted = direction in {"left", "right"}
        diagnostic = BoardEnvelopeRefinementDiagnostic(
            seed.x,
            seed.y,
            seed.width,
            seed.height,
            seed.x - (10 if direction == "left" else 0),
            seed.y,
            seed.width + 10,
            seed.height,
            direction,
            10,
            5,
            5,
            5,
            6,
            1.0 if accepted else 0.0,
            0.8 if accepted else 0.0,
            1.0 if accepted else 0.0,
            1.0 if accepted else 0.0,
            0.9 if accepted else 0.0,
            0.90 if accepted else 0.0,
            accepted,
        )
        return _GridEnvelopeRefinementCandidate(diagnostic, evidence, 0.0, 0.0)

    monkeypatch.setattr(refiner, "_evaluate_direction", fake_evaluate)

    grayscale = cast(
        NDArray[np.uint8],
        cv2.cvtColor(screenshot.image, cv2.COLOR_BGR2GRAY),
    )
    result = refiner.refine(
        grayscale,
        seed,
        evidence,
    )

    assert result.selected_diagnostic is None
    assert result.selected_grid_evidence is None
    assert any(
        "ambiguous grid-envelope refinements" in reason
        for diagnostic in result.diagnostics
        for reason in diagnostic.rejection_reasons
    )


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
        lambda: BoardDetectionSettings(
            grid_weak_horizontal_line_kernel_relative_length=0.0
        ),
        lambda: BoardDetectionSettings(
            grid_weak_vertical_line_kernel_relative_length=float("nan")
        ),
        lambda: BoardDetectionSettings(
            grid_missing_line_minimum_gap_factor=float("inf")
        ),
        lambda: BoardDetectionSettings(
            grid_missing_line_minimum_gap_factor=2.5,
            grid_missing_line_maximum_gap_factor=2.0,
        ),
        lambda: BoardDetectionSettings(
            grid_missing_line_search_half_width_fraction=0.0
        ),
        lambda: BoardDetectionSettings(
            grid_missing_line_search_half_width_fraction=0.51
        ),
        lambda: BoardDetectionSettings(grid_missing_line_minimum_weak_response=-0.01),
        lambda: BoardDetectionSettings(
            grid_missing_line_maximum_other_gap_deviation=float("nan")
        ),
        lambda: BoardDetectionSettings(grid_missing_line_minimum_cv_improvement=1.01),
        lambda: BoardDetectionSettings(grid_missing_line_maximum_recovered_per_axis=-1),
        lambda: BoardDetectionSettings(grid_missing_line_maximum_recovered_per_axis=2),
        lambda: BoardDetectionSettings(
            grid_missing_line_maximum_recovered_per_axis=0.5  # type: ignore[arg-type]
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_minimum_added_size_ratio=float("nan")
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_minimum_added_size_ratio=1.10,
            grid_envelope_maximum_added_size_ratio=1.20,
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_minimum_added_size_ratio=0.90,
            grid_envelope_maximum_added_size_ratio=0.80,
        ),
        lambda: BoardDetectionSettings(grid_envelope_maximum_added_cells_per_side=-1),
        lambda: BoardDetectionSettings(grid_envelope_maximum_added_cells_per_side=2),
        lambda: BoardDetectionSettings(
            grid_envelope_maximum_added_cells_per_side=0.5  # type: ignore[arg-type]
        ),
        lambda: BoardDetectionSettings(grid_envelope_minimum_seed_rows=0),
        lambda: BoardDetectionSettings(grid_envelope_minimum_seed_columns=0),
        lambda: BoardDetectionSettings(
            grid_envelope_separator_position_tolerance_ratio=0.0
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_continuation_probe_thickness_ratio=float("inf")
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_minimum_line_continuation_response=-0.01
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_minimum_supported_separator_fraction=1.01
        ),
        lambda: BoardDetectionSettings(
            grid_envelope_maximum_spacing_cv_increase=float("nan")
        ),
        lambda: BoardDetectionSettings(grid_envelope_maximum_grid_score_drop=1.01),
        lambda: BoardDetectionSettings(grid_envelope_minimum_refinement_score=-0.01),
        lambda: BoardDetectionSettings(grid_envelope_ambiguity_delta=1.01),
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
