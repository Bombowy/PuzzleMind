"""Cats autoplay retry tests."""

from collections.abc import Callable

import pytest

from cats_autoplay_test_support import (
    BoardAnalysisOutcome,
    _board_analysis_failure_state,
    _board_detection_error,
    _board_input,
    _color_detection_error,
    _detection,
    _existing_cat_detection_error,
    _geometry_board_input,
    _grid_detection_error,
    _runner,
    _settings,
)
from logicforge.application.cats import autoplay
from logicforge.application.cats import validation as cats_validation
from logicforge.plugins.cats import (
    CatsScreenState,
)
from logicforge.vision.color_detector import (
    ColorDetectionError,
)


@pytest.mark.parametrize(
    ("error_factory", "failure_count"),
    (
        (_color_detection_error, 1),
        (_board_detection_error, 2),
        (_grid_detection_error, 2),
        (_existing_cat_detection_error, 2),
    ),
    ids=("color", "board", "grid", "existing-cat"),
)
def test_transient_board_vision_errors_retry_new_frames_then_solve_once(
    error_factory: Callable[[], RuntimeError],
    failure_count: int,
) -> None:
    """Retry rejected animation frames and click only the stabilized analysis."""

    error = error_factory()
    valid_input = _board_input()
    outcomes: tuple[BoardAnalysisOutcome, ...] = (
        *((error,) * failure_count),
        valid_input,
    )
    runner, _, capturer, detector, mouse, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(timeout=20.0),
        board_inputs=outcomes,
    )

    summary = runner.run()

    expected_polls = failure_count + 1
    assert summary.solved_levels == 1
    assert len(capturer.calls) == expected_polls
    assert detector.calls == expected_polls
    assert analyzer.calls == expected_polls
    assert len({screenshot.timestamp for screenshot in analyzer.screenshots}) == (
        expected_polls
    )
    assert len(solver.calls) == 1
    assert solver.calls[0][1] is valid_input
    assert len(mouse.clicks) == 8
    assert runner._board_analysis_failure_started_at is None
    assert runner._last_board_analysis_error is None


def test_transient_failure_window_raises_current_error_at_deadline_without_clicks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Measure from the first failure and stop at the deterministic retry bound."""

    error = _color_detection_error()
    runner, _, capturer, detector, mouse, clock, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(
            max_levels=0,
            timeout=20.0,
            poll_interval=0.5,
            board_retry=3.0,
        ),
        board_inputs=(error,),
    )

    with pytest.raises(ColorDetectionError) as error_info:
        runner.run()

    assert error_info.value is error
    assert clock.now == pytest.approx(3.0)
    assert len(capturer.calls) == 7
    assert detector.calls == 7
    assert analyzer.calls == 7
    assert solver.calls == []
    assert mouse.clicks == []
    assert runner._board_analysis_failure_started_at == 0.0
    output = capsys.readouterr().out
    assert output.count("transient ColorDetectionError; retrying") == 3
    assert output.count("analysis did not stabilize within 3.0s") == 1


def test_board_unknown_board_state_change_resets_transient_failure_window() -> None:
    """Start a fresh retry window after BOARD leaves and later returns."""

    error = _board_detection_error()
    runner, _, _, _, _, clock, _, _, _ = _runner(
        (
            _detection(CatsScreenState.BOARD),
            _detection(CatsScreenState.UNKNOWN),
            _detection(CatsScreenState.BOARD),
        ),
        settings=_settings(max_levels=0, timeout=20.0),
        board_inputs=(error,),
    )

    assert runner._run_execute_poll() is False
    assert runner._board_analysis_failure_started_at == 0.0
    assert runner._run_execute_poll() is False
    assert _board_analysis_failure_state(runner) == (None, None)
    assert runner._run_execute_poll() is False
    assert runner._board_analysis_failure_started_at == pytest.approx(0.2)
    assert clock.now == pytest.approx(0.3)


def test_transient_board_failure_does_not_update_global_progress_timestamp() -> None:
    """Leave the independent 20-second progress fuse untouched by bad frames."""

    runner, _, _, _, _, clock, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=20.0),
        board_inputs=(_grid_detection_error(),),
    )
    clock.advance(5.0)
    runner._last_progress_at = 1.25

    assert runner._run_execute_poll() is False

    assert runner._last_progress_at == 1.25


def test_dry_run_transient_vision_error_remains_fail_fast() -> None:
    """Keep the existing one-capture dry-run policy outside execute retries."""

    error = _color_detection_error()
    runner, _, capturer, detector, mouse, clock, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(execute=False),
        board_inputs=(error,),
    )

    with pytest.raises(ColorDetectionError) as error_info:
        runner.run()

    assert error_info.value is error
    assert len(capturer.calls) == 1
    assert detector.calls == 1
    assert analyzer.calls == 1
    assert solver.calls == []
    assert mouse.clicks == []
    assert clock.sleeps == []


def test_transient_9x8_geometry_never_solves_clicks_or_updates_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retry live-like 9x8/colors=9 evidence before Board construction."""

    invalid_input = _geometry_board_input(9, 8, 9)
    runner, _, _, _, mouse, _, renderer, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
        board_inputs=(invalid_input,),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()
    runner.save_failure_overlay()

    output = capsys.readouterr().out
    assert analyzer.calls > 1
    assert solver.calls == []
    assert mouse.clicks == []
    assert runner.summary().solved_levels == 0
    assert runner._last_solved_color_matrix is None
    assert runner._phase is autoplay.CatsAutomationPhase.READY_FOR_BOARD
    assert analyzer.calls == 4
    assert runner._board_analysis_failure_started_at == 0.0
    assert "transient CatsBoardGeometryMismatchError; retrying" in output
    assert "new level accepted" not in output
    assert len(renderer.calls) == 1


def test_transient_9x8_then_9x9_solves_only_the_valid_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Re-run the full poll and accept the corrected 9x9 geometry once."""

    invalid_input = _geometry_board_input(9, 8, 9)
    valid_input = _geometry_board_input(9, 9, 9)
    runner, _, capturer, _, mouse, _, _, analyzer, solver = _runner(
        (
            _detection(CatsScreenState.BOARD),
            _detection(CatsScreenState.BOARD),
        ),
        board_inputs=(invalid_input, valid_input),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 2
    assert analyzer.calls == 2
    assert len(solver.calls) == 1
    assert solver.calls[0][1] is valid_input
    assert len(mouse.clicks) == 18
    assert runner._board_analysis_failure_started_at is None
    assert runner._last_board_analysis_error is None
    assert capsys.readouterr().out.count("new level accepted") == 1


def test_cats_input_geometry_guard_rejects_wrong_color_count() -> None:
    """Reject square 9x9 geometry when immutable vision reports eight colors."""

    with pytest.raises(
        cats_validation.CatsBoardGeometryMismatchError,
        match=r"grid=9x9, colors=8",
    ):
        cats_validation.validate_cats_board_input_geometry(
            _geometry_board_input(9, 9, 8)
        )


def test_cats_input_geometry_guard_rejects_matrix_shape_before_solver() -> None:
    """Validate immutable matrix dimensions before any logical Board is created."""

    invalid_input = _geometry_board_input(9, 9, 9)
    object.__setattr__(
        invalid_input.color_result,
        "color_matrix",
        invalid_input.color_result.color_matrix[:-1],
    )
    runner, _, _, _, mouse, _, _, _, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.15),
        board_inputs=(invalid_input,),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    assert solver.calls == []
    assert mouse.clicks == []


def test_dry_run_9x8_returns_typed_validation_failure_without_click() -> None:
    """Surface controlled code-8-compatible geometry failure in one dry poll."""

    runner, _, capturer, _, mouse, _, _, _, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(execute=False),
        board_inputs=(_geometry_board_input(9, 8, 9),),
    )

    with pytest.raises(cats_validation.CatsBoardGeometryMismatchError):
        runner.run()

    assert len(capturer.calls) == 1
    assert solver.calls == []
    assert mouse.clicks == []
