"""Cats screen-state board tests."""

from inspect import getsource

import numpy as np
import pytest

from cats_screen_state_test_support import (
    _assert_no_backend_objects,
    _FakeBoardDetector,
    _FakeGridDetector,
    _OpenCvFailingBoardDetector,
)
from logicforge.infrastructure import (
    opencv_cats_screen_state_detector as detector_module,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.plugins.cats import (
    CatsScreenState,
)
from synthetic_cats_screen_states import (
    synthetic_board_screen,
    synthetic_level_complete_screen,
    synthetic_ranking_screen,
    synthetic_unknown_screen,
)
from synthetic_cats_tile_grids import synthetic_cats_tile_grid


def test_regular_grid_screen_is_board_without_action() -> None:
    """Delegate ordinary board recognition to the existing detector pair."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_board_screen())

    assert result.state is CatsScreenState.BOARD
    assert result.action_point is None
    assert result.diagnostics.board_candidate is not None
    assert result.diagnostics.detected_rows == 6
    assert result.diagnostics.detected_columns == 6


def test_cats_tile_lattice_is_board_without_reliable_outer_contour() -> None:
    """Use tile-grid-first geometry for a stable Cats-like board by default."""

    screenshot = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        pastel_outer_column=True,
    ).screenshot

    result = OpenCvCatsScreenStateDetector().detect(screenshot)

    assert result.state is CatsScreenState.BOARD
    assert result.diagnostics.detected_rows == 9
    assert result.diagnostics.detected_columns == 9


def test_successful_tile_grid_does_not_invoke_generic_board_fallback() -> None:
    """Never let contour fallback replace a complete primary Cats lattice."""

    board_detector = _FakeBoardDetector(fail=True)
    grid_detector = _FakeGridDetector(fail=True)
    screenshot = synthetic_cats_tile_grid(rows=9, columns=9).screenshot
    detector = OpenCvCatsScreenStateDetector(
        tile_grid_detector=OpenCvCatsTileGridDetector(),
        board_detector=board_detector,
        grid_detector=grid_detector,
    )

    result = detector.detect(screenshot)

    assert result.state is CatsScreenState.BOARD
    assert board_detector.calls == 0
    assert grid_detector.calls == 0


def test_empty_unknown_screen_returns_unknown_without_action() -> None:
    """Return a normal zero-confidence result when no known evidence exists."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert result.confidence == 0.0
    assert result.action_point is None


def test_overlay_states_do_not_invoke_injected_board_detectors() -> None:
    """Prove both transition checks run before delegated BOARD analysis."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    detector = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    )

    assert detector.detect(synthetic_ranking_screen()).state is CatsScreenState.RANKING
    assert (
        detector.detect(synthetic_level_complete_screen()).state
        is CatsScreenState.LEVEL_COMPLETE
    )
    assert board_detector.calls == 0
    assert grid_detector.calls == 0


def test_all_confidences_and_scores_are_in_unit_interval() -> None:
    """Keep transition and board evidence safe for public diagnostics."""

    detector = OpenCvCatsScreenStateDetector()
    results = (
        detector.detect(synthetic_board_screen()),
        detector.detect(synthetic_ranking_screen()),
        detector.detect(synthetic_level_complete_screen()),
        detector.detect(synthetic_unknown_screen()),
    )

    for result in results:
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.diagnostics.level_button_score <= 1.0
        assert 0.0 <= result.diagnostics.ranking_score <= 1.0


def test_repeated_analysis_is_exactly_deterministic() -> None:
    """Return equal immutable diagnostics independently of contour ordering."""

    screenshot = synthetic_ranking_screen(include_bubble=True)
    detector = OpenCvCatsScreenStateDetector()

    results = tuple(detector.detect(screenshot) for _ in range(3))

    assert results[0] == results[1] == results[2]


def test_diagnostics_contain_no_opencv_or_numpy_objects() -> None:
    """Expose only frozen primitive plugin models across the adapter boundary."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_ranking_screen())

    _assert_no_backend_objects(result.diagnostics)


def test_detection_does_not_mutate_screenshot() -> None:
    """Treat immutable source pixels as read-only throughout all analyses."""

    screenshot = synthetic_level_complete_screen(include_ranking_cards=True)
    expected = screenshot.image.copy()

    OpenCvCatsScreenStateDetector().detect(screenshot)

    assert np.array_equal(screenshot.image, expected)


def test_injected_board_and_grid_detectors_produce_board() -> None:
    """Compose the public ports without requiring their concrete OpenCV adapters."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    result = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.BOARD
    assert result.confidence == 0.82
    assert board_detector.calls == 1
    assert grid_detector.calls == 1


def test_board_detector_rejection_becomes_unknown_not_exception() -> None:
    """Treat ordinary absence of a board as a classification outcome."""

    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(fail=True),
        grid_detector=_FakeGridDetector(),
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert "board detector rejected the screenshot" in (
        result.diagnostics.rejection_reasons
    )


def test_grid_detector_rejection_becomes_unknown_not_exception() -> None:
    """Retain the board candidate but fail closed when public grid evidence fails."""

    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(),
        grid_detector=_FakeGridDetector(fail=True),
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert result.diagnostics.board_candidate is not None
    assert "grid detector rejected the board" in result.diagnostics.rejection_reasons


def test_opencv_board_processing_failure_raises_typed_detection_error() -> None:
    """Distinguish an unusable backend from an ordinary UNKNOWN screen."""

    detector = OpenCvCatsScreenStateDetector(
        board_detector=_OpenCvFailingBoardDetector(),
        grid_detector=_FakeGridDetector(),
    )

    with pytest.raises(CatsScreenStateDetectionError):
        detector.detect(synthetic_unknown_screen())


def test_detector_source_uses_no_ocr_template_or_fixed_resolution() -> None:
    """Keep classification geometric, color-based, and scale-relative."""

    source = getsource(detector_module).casefold()

    for forbidden in (
        "pytesseract",
        "easyocr",
        "matchtemplate",
        "template matching",
        "800x",
        "1000x",
    ):
        assert forbidden not in source


def test_detector_contains_no_live_resolution_or_coordinate_constants() -> None:
    """Keep production geometry proportional and independent from regression sizes."""

    source = getsource(detector_module)

    assert "916" not in source
    assert "1920" not in source
    assert "y=32" not in source
