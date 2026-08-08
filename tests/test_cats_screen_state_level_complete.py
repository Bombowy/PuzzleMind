"""Cats screen-state level tests."""

import cv2
import pytest

from cats_screen_state_test_support import (
    ORANGE,
    _bluestacks_window,
    _orange_shape_screenshot,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    OpenCvCatsScreenStateDetector,
)
from logicforge.plugins.cats import (
    CatsScreenPoint,
    CatsScreenState,
)
from synthetic_cats_screen_states import (
    synthetic_level_complete_screen,
    synthetic_unknown_screen,
)
from synthetic_vision import screenshot_from_image


def test_large_lower_orange_button_is_level_complete() -> None:
    """Recognize the mandatory transition through color and geometry alone."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())

    assert result.state is CatsScreenState.LEVEL_COMPLETE


def test_level_complete_action_is_exact_button_center() -> None:
    """Expose the center of the detected orange rectangle as screenshot action."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())
    button = result.diagnostics.level_button_candidate

    assert button is not None
    assert result.action_point is not None
    assert (result.action_point.x, result.action_point.y) == (
        button.center_x,
        button.center_y,
    )


@pytest.mark.parametrize(
    "rectangle",
    (
        (380, 850, 40, 40),
        (130, 100, 540, 85),
        (100, 860, 600, 16),
    ),
)
def test_small_top_or_very_thin_orange_shapes_are_not_level_complete(
    rectangle: tuple[int, int, int, int],
) -> None:
    """Reject icons, upper banners, and thin strips despite saturated orange."""

    result = OpenCvCatsScreenStateDetector().detect(_orange_shape_screenshot(rectangle))

    assert result.state is not CatsScreenState.LEVEL_COMPLETE


def test_orange_confetti_is_not_level_complete() -> None:
    """Require one large coherent lower component instead of many small accents."""

    image = synthetic_unknown_screen().image.copy()
    for index in range(18):
        x = 35 + (index * 97) % 720
        y = 620 + (index * 53) % 300
        cv2.circle(image, (x, y), 5, ORANGE, cv2.FILLED)

    result = OpenCvCatsScreenStateDetector().detect(screenshot_from_image(image))

    assert result.state is not CatsScreenState.LEVEL_COMPLETE


def test_valid_level_button_is_not_shadowed_by_higher_scoring_invalid_shape() -> None:
    """Select the best accepted component before retaining rejected diagnostics."""

    image = synthetic_unknown_screen().image.copy()
    cv2.rectangle(image, (180, 760), (620, 830), ORANGE, cv2.FILLED)
    cv2.rectangle(image, (50, 850), (750, 940), ORANGE, cv2.FILLED)

    result = OpenCvCatsScreenStateDetector().detect(screenshot_from_image(image))

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.diagnostics.level_button_candidate is not None
    assert result.diagnostics.level_button_candidate.width < 600


def test_level_complete_visible_over_board_has_priority() -> None:
    """Classify the orange transition before any background grid evidence."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())

    assert result.state is CatsScreenState.LEVEL_COMPLETE


def test_level_complete_has_priority_over_ranking() -> None:
    """Prefer the characteristic lower button when both overlays are present."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_level_complete_screen(include_ranking_cards=True)
    )

    assert result.state is CatsScreenState.LEVEL_COMPLETE


@pytest.mark.parametrize(
    ("screenshot_width", "viewport_x", "left_ad"),
    ((916, 321, True), (1920, 679, False)),
)
def test_live_proportioned_level_button_is_viewport_relative(
    screenshot_width: int,
    viewport_x: int,
    left_ad: bool,
) -> None:
    """Recognize a 70%-viewport button below 45% of the full window width."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        screenshot_width=screenshot_width,
        viewport_x=viewport_x,
        left_ad=left_ad,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate
    button = result.diagnostics.level_button_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.confidence >= 0.60
    assert viewport is not None
    assert button is not None
    assert button.width / screenshot.width < 0.45
    assert button.width / viewport.width > 0.65
    assert result.action_point == CatsScreenPoint(button.center_x, button.center_y)


def test_orange_advertisement_element_cannot_trigger_level_complete() -> None:
    """Ignore a large orange ad component lying outside the selected game crop."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING)
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.RANKING
    assert viewport is not None
    assert result.diagnostics.level_button_candidate is None
