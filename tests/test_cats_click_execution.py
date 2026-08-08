"""Cats solve click_execution tests."""

import pytest

from cats_solve_test_support import (
    _click_targets,
    _FakeMouseController,
    _FakeSleep,
    _offset_window,
    _rectangular_grid_detection,
    _rectangular_logical_board,
)
from logicforge.application import cats as cats_app
from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.infrastructure.windows import MouseAutomationError


def test_execute_one_target_emits_exactly_two_left_clicks() -> None:
    """Represent one Cats action as exactly one left-button double-click."""

    controller = _FakeMouseController()

    executed = cats_app.execute_cat_click_plan(
        (_click_targets()[0],),
        controller,
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert executed == 1
    assert len(controller.clicks) == 2
    assert all(button is MouseButton.LEFT for _, button in controller.clicks)


def test_execute_reuses_identical_desktop_point_for_both_clicks() -> None:
    """Do not recalculate or move the second click away from its target."""

    controller = _FakeMouseController()
    target = _click_targets()[0]

    cats_app.execute_cat_click_plan(
        (target,),
        controller,
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert controller.clicks == [
        (ScreenPoint(target.desktop_x, target.desktop_y), MouseButton.LEFT),
        (ScreenPoint(target.desktop_x, target.desktop_y), MouseButton.LEFT),
    ]


def test_execute_uses_desktop_not_screenshot_coordinates() -> None:
    """Create ScreenPoint solely from the previously mapped desktop values."""

    controller = _FakeMouseController()
    target = cats_app.CatClickTarget(0, 0, 1, 2, 901, 902)

    cats_app.execute_cat_click_plan(
        (target,),
        controller,
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert {point for point, _ in controller.clicks} == {ScreenPoint(901, 902)}
    assert ScreenPoint(1, 2) not in {point for point, _ in controller.clicks}


def test_execute_two_targets_emits_four_clicks_in_row_major_plan_order() -> None:
    """Finish both clicks for one target before advancing to the next target."""

    controller = _FakeMouseController()

    executed = cats_app.execute_cat_click_plan(
        _click_targets(),
        controller,
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert executed == 2
    assert [point for point, _ in controller.clicks] == [
        ScreenPoint(420, 330),
        ScreenPoint(420, 330),
        ScreenPoint(440, 350),
        ScreenPoint(440, 350),
    ]


def test_execute_interleaves_clicks_and_delays_without_trailing_sleep() -> None:
    """Prove the exact click/sleep sequence across consecutive targets."""

    events: list[str] = []

    class RecordingController(MouseController):
        """Record click order into a timeline shared with the fake sleeper."""

        def click(
            self,
            point: ScreenPoint,
            button: MouseButton = MouseButton.LEFT,
        ) -> None:
            del point, button
            events.append("click")

    def record_sleep(seconds: float) -> None:
        del seconds
        events.append("sleep")

    cats_app.execute_cat_click_plan(
        _click_targets(),
        RecordingController(),
        click_delay_seconds=0.01,
        sleep_function=record_sleep,
    )

    assert events == [
        "click",
        "sleep",
        "click",
        "sleep",
        "click",
        "sleep",
        "click",
    ]


def test_execute_default_delay_is_ten_milliseconds() -> None:
    """Use 0.01 seconds when no execution delay override is supplied."""

    sleeper = _FakeSleep()

    cats_app.execute_cat_click_plan(
        (_click_targets()[0],),
        _FakeMouseController(),
        sleep_function=sleeper,
    )

    assert sleeper.calls == [0.01]


def test_execute_two_targets_sleeps_between_all_consecutive_clicks_only() -> None:
    """Emit 2N-1 equal pauses and never sleep after the final click."""

    sleeper = _FakeSleep()

    cats_app.execute_cat_click_plan(
        _click_targets(),
        _FakeMouseController(),
        click_delay_seconds=0.025,
        sleep_function=sleeper,
    )

    assert sleeper.calls == [0.025, 0.025, 0.025]


def test_execute_accepts_zero_delay() -> None:
    """Allow deterministic immediate consecutive calls when explicitly requested."""

    sleeper = _FakeSleep()

    assert (
        cats_app.execute_cat_click_plan(
            (_click_targets()[0],),
            _FakeMouseController(),
            click_delay_seconds=0,
            sleep_function=sleeper,
        )
        == 1
    )
    assert sleeper.calls == [0]


def test_execute_rejects_negative_delay_before_any_click() -> None:
    """Fail validation before emitting input or sleeping."""

    controller = _FakeMouseController()
    sleeper = _FakeSleep()

    with pytest.raises(cats_app.CatClickExecutionError, match="greater than"):
        cats_app.execute_cat_click_plan(
            _click_targets(),
            controller,
            click_delay_seconds=-0.001,
            sleep_function=sleeper,
        )

    assert controller.clicks == []
    assert sleeper.calls == []


def test_execute_empty_plan_returns_zero_without_click_or_sleep() -> None:
    """Treat an empty solved plan as a safe successful no-op."""

    controller = _FakeMouseController()
    sleeper = _FakeSleep()

    executed = cats_app.execute_cat_click_plan(
        (),
        controller,
        sleep_function=sleeper,
    )

    assert executed == 0
    assert controller.clicks == []
    assert sleeper.calls == []


@pytest.mark.parametrize("fail_on_call", [0, 1])
def test_execute_stops_on_first_or_second_click_failure(fail_on_call: int) -> None:
    """Propagate native failure without attempting any later target."""

    controller = _FakeMouseController(fail_on_call=fail_on_call)

    with pytest.raises(MouseAutomationError, match="synthetic native"):
        cats_app.execute_cat_click_plan(
            _click_targets(),
            controller,
            click_delay_seconds=0,
            sleep_function=_FakeSleep(),
        )

    assert len(controller.clicks) == fail_on_call + 1
    assert all(point == ScreenPoint(420, 330) for point, _ in controller.clicks)


def test_execute_does_not_mutate_immutable_targets() -> None:
    """Consume the caller's tuple and values as read-only execution input."""

    targets = _click_targets()
    expected = targets

    cats_app.execute_cat_click_plan(
        targets,
        _FakeMouseController(),
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert targets == expected


def test_execute_does_not_mutate_board_or_grid() -> None:
    """Keep logical and detected geometry untouched by pointer execution."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    grid = _rectangular_grid_detection()
    board_before = tuple(tuple(row) for row in board.cells)
    grid_before = grid
    targets = cats_app.build_cat_click_plan(board, grid, _offset_window())

    cats_app.execute_cat_click_plan(
        targets,
        _FakeMouseController(),
        click_delay_seconds=0,
        sleep_function=_FakeSleep(),
    )

    assert tuple(tuple(row) for row in board.cells) == board_before
    assert grid == grid_before
