"""Cats autoplay safety tests."""

import pytest

from cats_autoplay_test_support import (
    FakeWindowLocator,
    _detection,
    _runner,
    _settings,
    _window,
)
from logicforge.application import cats as cats_app
from logicforge.automation.mouse import MouseButton, ScreenPoint
from logicforge.plugins.cats import (
    CatsScreenState,
)


@pytest.mark.parametrize(
    ("state", "expected_phase", "counter_name"),
    (
        (
            CatsScreenState.RANKING,
            cats_app.CatsAutomationPhase.WAITING_FOR_LEVEL_COMPLETE,
            "ranking_clicks",
        ),
        (
            CatsScreenState.LEVEL_COMPLETE,
            cats_app.CatsAutomationPhase.WAITING_FOR_NEXT_BOARD,
            "level_button_clicks",
        ),
    ),
)
def test_overlay_emits_one_left_click_and_converts_desktop_coordinates(
    state: CatsScreenState,
    expected_phase: cats_app.CatsAutomationPhase,
    counter_name: str,
) -> None:
    """Single-click accepted screenshot actions, including negative desktop x."""

    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(state),),
        settings=_settings(max_levels=0, timeout=0.2),
    )

    with pytest.raises(cats_app.CatsAutomationTimeoutError):
        runner.run()

    expected_y = 70 if state is CatsScreenState.RANKING else 85
    assert mouse.clicks[0] == (
        ScreenPoint(-1470, 1 + expected_y),
        MouseButton.LEFT,
    )
    assert len(mouse.clicks) == 1
    assert getattr(runner.summary(), counter_name) == 1
    assert runner._phase is expected_phase


def test_bounds_change_skips_stale_click_and_recaptures() -> None:
    """Never translate old geometry by a window delta after BlueStacks moves."""

    old = _window()
    moved = _window(x=-1300)
    locator = FakeWindowLocator((old, moved, old, old, old, old))
    states = (
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, _, capturer, _, mouse, _, _, _, _ = _runner(
        states,
        locator=locator,
    )

    runner.run()

    assert len(capturer.calls) == 4
    assert mouse.clicks[0][0] == ScreenPoint(-1470, 71)
    assert all(point.x != -1270 for point, _ in mouse.clicks)


def test_every_low_level_cat_click_rechecks_window_bounds() -> None:
    """Guard every delegated click while retaining the existing plan executor."""

    runner, locator, capturer, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),)
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(mouse.clicks) == 8
    assert len(capturer.calls) == 1
    assert locator.calls == 1 + 1 + len(mouse.clicks)


def test_bounds_change_before_first_cat_click_forces_fresh_poll() -> None:
    """Discard the old plan when movement occurs between validation and click."""

    old = _window()
    moved = _window(x=-1300)
    locator = FakeWindowLocator((old, old, moved))
    runner, _, capturer, _, mouse, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        locator=locator,
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 2
    assert analyzer.calls == 2
    assert len(solver.calls) == 2
    assert len(mouse.clicks) == 8
    assert mouse.clicks[0][0].x >= moved.bounds.x


def test_stationary_overlay_retries_only_after_delay_and_stops_at_limit() -> None:
    """Throttle identical overlays and fail after bounded delayed retries."""

    runner, _, _, _, mouse, clock, renderer, _, _ = _runner(
        (_detection(CatsScreenState.RANKING),),
        settings=_settings(
            max_levels=0,
            timeout=10.0,
            poll_interval=0.1,
            overlay_retry=0.3,
            max_overlay_retries=2,
        ),
    )

    with pytest.raises(cats_app.CatsAutomationTimeoutError, match="2 overlay retries"):
        runner.run()
    runner.save_failure_overlay()

    assert len(mouse.clicks) == 3
    assert clock.now >= 0.9
    assert len(renderer.calls) == 1


def test_unknown_never_clicks_or_solves_and_eventually_times_out() -> None:
    """Keep stationary UNKNOWN passive while enforcing progress timeout."""

    runner, _, _, _, mouse, clock, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.UNKNOWN),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
    )

    with pytest.raises(cats_app.CatsAutomationTimeoutError):
        runner.run()

    assert mouse.clicks == []
    assert analyzer.calls == 0
    assert solver.calls == []
    assert clock.sleeps == [0.1, 0.1, 0.1]


def test_unknown_can_be_interleaved_and_state_change_resets_timeout() -> None:
    """Treat recognized state changes as progress without clicking UNKNOWN."""

    states = (
        _detection(CatsScreenState.UNKNOWN),
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.UNKNOWN),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        states,
        settings=_settings(timeout=0.15, poll_interval=0.1),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert summary.ranking_clicks == 1
    assert summary.level_button_clicks == 1
    assert len(mouse.clicks) == 10


def test_stationary_board_waiting_for_transition_times_out() -> None:
    """Do not let an unchanged post-solve BOARD reset progress forever."""

    runner, _, _, _, _, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
    )

    with pytest.raises(cats_app.CatsAutomationTimeoutError):
        runner.run()

    assert analyzer.calls == 1
    assert len(solver.calls) == 1


def test_max_levels_one_stops_without_poll_sleep_or_transition_click() -> None:
    """Return immediately after the requested board count is clicked."""

    runner, _, capturer, _, mouse, clock, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=1),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 1
    assert len(mouse.clicks) == 8
    assert 0.1 not in clock.sleeps


def test_max_levels_zero_continues_until_timeout() -> None:
    """Interpret zero as unlimited rather than immediate completion."""

    runner, _, capturer, _, _, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.15),
    )

    with pytest.raises(cats_app.CatsAutomationTimeoutError):
        runner.run()

    assert len(capturer.calls) > 1


def test_successful_session_does_not_save_failure_overlay() -> None:
    """Keep normal polling free from debug filesystem writes."""

    runner, _, _, _, _, _, renderer, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),)
    )

    runner.run()

    assert renderer.calls == []
