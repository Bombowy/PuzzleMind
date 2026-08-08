"""Cats screen-state ranking tests."""

import pytest

from cats_screen_state_test_support import (
    _bluestacks_window,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    OpenCvCatsScreenStateDetector,
)
from logicforge.plugins.cats import (
    CatsScreenState,
)
from synthetic_cats_screen_states import (
    synthetic_ranking_screen,
)


@pytest.mark.parametrize("card_count", [2, 3])
def test_two_or_three_aligned_bright_cards_are_ranking(card_count: int) -> None:
    """Accept a vertical stack without requiring exactly three cards or text."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=card_count)
    )

    assert result.state is CatsScreenState.RANKING
    assert len(result.diagnostics.ranking_card_candidates) == card_count


def test_one_bright_card_is_not_ranking() -> None:
    """Reject a single modal card as insufficient stack evidence."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=1)
    )

    assert result.state is not CatsScreenState.RANKING
    assert "only one viewport-relative ranking card was accepted" in (
        result.diagnostics.rejection_reasons
    )


def test_unaligned_bright_rectangles_are_not_ranking() -> None:
    """Reject multiple cards that do not share a sufficiently aligned stack."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=3, aligned=False)
    )

    assert result.state is not CatsScreenState.RANKING


def test_ranking_action_is_below_stack_and_inside_screenshot() -> None:
    """Derive a safe lower action from stack union rather than fixed pixels."""

    screenshot = synthetic_ranking_screen()
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    action = result.action_point
    cards = result.diagnostics.ranking_card_candidates

    assert action is not None
    assert action.y > max(card.y + card.height for card in cards)
    assert 0 <= action.x < screenshot.width
    assert 0 <= action.y < screenshot.height


def test_ranking_action_scales_with_resolution() -> None:
    """Keep action ratios stable when no absolute screenshot size is assumed."""

    small = synthetic_ranking_screen(width=480, height=720)
    large = synthetic_ranking_screen(width=960, height=1440)
    small_action = OpenCvCatsScreenStateDetector().detect(small).action_point
    large_action = OpenCvCatsScreenStateDetector().detect(large).action_point

    assert small_action is not None
    assert large_action is not None
    assert small_action.x / small.width == pytest.approx(
        large_action.x / large.width,
        abs=0.01,
    )
    assert small_action.y / small.height == pytest.approx(
        large_action.y / large.height,
        abs=0.01,
    )


def test_ranking_visible_over_board_has_priority_over_board() -> None:
    """Classify the modal stack before the partially visible background board."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_ranking_screen())

    assert result.state is CatsScreenState.RANKING


@pytest.mark.parametrize("card_count", [2, 3])
def test_full_window_ranking_cards_are_viewport_relative(card_count: int) -> None:
    """Recognize interrupted neutral, cream, and peach cards inside the game."""

    screenshot = _bluestacks_window(
        CatsScreenState.RANKING,
        card_count=card_count,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.RANKING
    assert viewport is not None
    assert len(result.diagnostics.ranking_card_candidates) == card_count
    assert result.diagnostics.ranking_score >= 0.64
    assert all(
        card.width / viewport.width > 0.74
        for card in result.diagnostics.ranking_card_candidates
    )
    assert result.action_point is not None
    assert result.action_point.y > max(
        card.y + card.height for card in result.diagnostics.ranking_card_candidates
    )
    assert viewport.x <= result.action_point.x < viewport.x + viewport.width


def test_one_or_unaligned_full_window_card_stack_is_not_ranking() -> None:
    """Require at least two aligned viewport-relative cards."""

    one = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING, card_count=1)
    )
    unaligned = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING, aligned=False)
    )

    assert one.state is not CatsScreenState.RANKING
    assert unaligned.state is not CatsScreenState.RANKING


def test_bright_ad_rectangles_do_not_trigger_ranking() -> None:
    """Do not count the synthetic ad's light rectangles as ranking cards."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.LEVEL_COMPLETE)
    )

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.diagnostics.ranking_card_candidates == ()
