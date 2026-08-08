"""Cats screen-state viewport tests."""

import numpy as np
import pytest

from cats_screen_state_test_support import (
    _bluestacks_window,
    _FakeBoardDetector,
    _FakeGridDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    OpenCvCatsScreenStateDetector,
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


@pytest.mark.parametrize(
    ("width", "height", "factory", "expected_state"),
    (
        (400, 720, synthetic_level_complete_screen, CatsScreenState.LEVEL_COMPLETE),
        (810, 1440, synthetic_ranking_screen, CatsScreenState.RANKING),
        (700, 700, synthetic_board_screen, CatsScreenState.BOARD),
        (500, 900, synthetic_board_screen, CatsScreenState.BOARD),
    ),
)
def test_detector_supports_multiple_resolutions_and_portrait_ratios(
    width: int,
    height: int,
    factory: object,
    expected_state: CatsScreenState,
) -> None:
    """Use relative thresholds across square and BlueStacks-like portrait frames."""

    screenshot_factory = factory
    assert callable(screenshot_factory)
    screenshot = screenshot_factory(width=width, height=height)

    assert OpenCvCatsScreenStateDetector().detect(screenshot).state is expected_state


@pytest.mark.parametrize(
    ("screenshot_width", "viewport_x", "left_ad"),
    ((916, 321, True), (1920, 679, False)),
)
def test_detects_game_viewport_in_wide_bluestacks_windows(
    screenshot_width: int,
    viewport_x: int,
    left_ad: bool,
) -> None:
    """Locate the game strip for both observed live window proportions."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(
            CatsScreenState.LEVEL_COMPLETE,
            screenshot_width=screenshot_width,
            viewport_x=viewport_x,
            left_ad=left_ad,
        )
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x == pytest.approx(viewport_x, abs=25)
    assert viewport.y == pytest.approx(33, abs=3)
    assert viewport.width == pytest.approx(562, abs=30)
    assert viewport.height == pytest.approx(999, abs=3)
    assert viewport.width / viewport.height == pytest.approx(9 / 16, abs=0.035)
    assert 0.0 <= result.diagnostics.game_viewport_score <= 1.0


def test_viewport_uses_full_screenshot_coordinates_and_excludes_side_ui() -> None:
    """Keep the ad left of and toolbar right of the selected viewport."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.LEVEL_COMPLETE)
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x >= 321 - 25
    assert viewport.x + viewport.width <= 321 + 562 + 25
    assert viewport.y > 0


def test_viewport_detection_is_deterministic_and_does_not_mutate_pixels() -> None:
    """Return identical global geometry while preserving the immutable capture."""

    screenshot = _bluestacks_window(CatsScreenState.RANKING)
    expected = screenshot.image.copy()
    detector = OpenCvCatsScreenStateDetector()

    results = tuple(detector.detect(screenshot) for _ in range(3))

    assert results[0] == results[1] == results[2]
    assert np.array_equal(screenshot.image, expected)


def test_narrow_full_content_viewport_uses_fallback() -> None:
    """Accept whole portrait content without two better side boundaries."""

    screenshot = synthetic_level_complete_screen(width=560, height=1000)
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert viewport is not None
    assert viewport.x == 0
    assert viewport.width == screenshot.width


def test_viewport_at_left_edge_is_supported() -> None:
    """Allow one screenshot edge to be a legitimate viewport boundary."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        viewport_x=0,
        left_ad=False,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert viewport is not None
    assert viewport.x == 0


def test_advertisement_and_toolbar_do_not_win_viewport_selection() -> None:
    """Reject complex ad content and a narrow toolbar as game viewports."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        screenshot_width=1920,
        viewport_x=679,
        left_ad=True,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x == pytest.approx(679, abs=25)
    assert viewport.width > 500
    assert viewport.x + viewport.width < 1900


def test_no_sensible_viewport_returns_none_and_specific_reason() -> None:
    """Do not run overlay geometry against a landscape full screenshot."""

    screenshot = synthetic_unknown_screen(width=1200, height=500)
    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(fail=True),
        grid_detector=_FakeGridDetector(),
    ).detect(screenshot)

    assert result.diagnostics.game_viewport_candidate is None
    assert "no reliable Cats game viewport was found" in (
        result.diagnostics.rejection_reasons
    )


def test_missing_viewport_does_not_block_full_screenshot_board_attempt() -> None:
    """Keep BOARD independent from transition-screen viewport availability."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    result = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    ).detect(synthetic_unknown_screen(width=1200, height=500))

    assert result.state is CatsScreenState.BOARD
    assert result.diagnostics.game_viewport_candidate is None
    assert board_detector.calls == 1
    assert grid_detector.calls == 1
