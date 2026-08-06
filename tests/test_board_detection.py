"""Deterministic tests for classical puzzle-board localization and debug output."""

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


@pytest.mark.skip(reason="No anonymized local BlueStacks puzzle fixture is available.")
def test_detects_a_board_in_an_anonymized_bluestacks_fixture() -> None:
    """Reserve integration coverage until a safe puzzle screenshot is contributed."""
