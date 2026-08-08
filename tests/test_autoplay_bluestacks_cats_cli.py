"""Cats autoplay cli tests."""

from inspect import getsource

import pytest
from scripts import autoplay_bluestacks_cats as autoplay
from scripts import solve_bluestacks_cats as solve_script

from cats_autoplay_test_support import (
    FakeMouseController,
    _settings,
)
from logicforge.application.cats import autoplay as cats_autoplay
from logicforge.automation.mouse import MouseController
from logicforge.core import BoardStateError
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
)
from logicforge.infrastructure.windows import MouseAutomationError
from logicforge.plugins.cats import (
    CatsScreenState,
)
from logicforge.vision.board_detector import (
    BoardDetectionDiagnostics,
    BoardDetectionError,
)
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionError,
)
from logicforge.vision.grid_detector import (
    GridDetectionDiagnostics,
    GridDetectionError,
)
from logicforge.vision.window_capture import (
    WindowCaptureError,
)


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    (
        (WindowCaptureError("capture"), 1),
        (CatsScreenStateDetectionError("screen"), 2),
        (
            BoardDetectionError(
                "board",
                BoardDetectionDiagnostics(0, (), None, 0),
            ),
            3,
        ),
        (
            GridDetectionError(
                "grid",
                GridDetectionDiagnostics(
                    0, 0, 1, 1, (), (), (), (), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, ()
                ),
            ),
            4,
        ),
        (
            ColorDetectionError(
                "color",
                ColorDetectionDiagnostics(1, 1, 1.0, (), (), (), None, ("error",)),
            ),
            5,
        ),
        (BoardStateError("board state"), 6),
        (solve_script.CatClickPlanError("plan"), 7),
        (autoplay.CatsSolutionValidationError("validation"), 8),
        (autoplay.CatsBoardGeometryMismatchError("geometry"), 8),
        (MouseAutomationError("mouse"), 9),
        (solve_script.CatClickExecutionError("execution"), 9),
        (autoplay.CatsAutomationTimeoutError("timeout"), 10),
    ),
)
def test_main_maps_expected_errors_to_documented_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    exception: RuntimeError,
    expected_code: int,
) -> None:
    """Translate expected operational failures without broad exception handling."""

    class FailingRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self) -> autoplay.CatsAutoplaySummary:
            raise exception

        def summary(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(0, 0, 0, 0, CatsScreenState.UNKNOWN)

        def save_failure_overlay(self) -> None:
            return None

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", FailingRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", object)

    assert autoplay.main(("--execute",)) == expected_code


def test_keyboard_interrupt_returns_130_without_click(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch Ctrl+C only at main and print the partial summary."""

    mouse = FakeMouseController()

    class InterruptedRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self) -> autoplay.CatsAutoplaySummary:
            raise KeyboardInterrupt

        def summary(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(1, 0, 0, 8, CatsScreenState.BOARD)

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", InterruptedRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", lambda: mouse)

    assert autoplay.main(("--execute",)) == 130
    assert mouse.clicks == []
    assert "Cats autoplay stopped by user." in capsys.readouterr().out


def test_dry_main_does_not_construct_win32_mouse_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep native mouse composition strictly behind --execute."""

    class SuccessfulRunner:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["mouse_controller"] is None

        def run(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(0, 0, 0, 0, CatsScreenState.UNKNOWN)

    def reject_mouse() -> MouseController:
        raise AssertionError("dry-run must not construct Win32MouseController")

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", SuccessfulRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", reject_mouse)

    assert autoplay.main(()) == 0


def test_cli_defaults_and_validation() -> None:
    """Expose the documented defaults and reject every unsafe boundary."""

    arguments = autoplay.parse_arguments(())

    assert arguments.execute is False
    assert arguments.click_delay_ms == 10
    assert arguments.poll_interval_ms == 100
    assert arguments.transition_timeout_seconds == 20.0
    assert arguments.board_analysis_retry_seconds == 3.0
    assert arguments.overlay_retry_ms == 750
    assert arguments.max_overlay_retries == 3
    assert arguments.new_board_delay_ms == 300
    assert arguments.max_levels == 0
    settings = autoplay.settings_from_arguments(arguments)
    assert settings.board_analysis_retry_seconds == 3.0


@pytest.mark.parametrize("value", (0.0, -1.0, float("nan"), float("inf")))
def test_board_analysis_retry_setting_requires_finite_positive_value(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="board_analysis_retry_seconds"):
        _settings(board_retry=value)


def test_cli_copies_board_analysis_retry_seconds_into_settings() -> None:
    arguments = autoplay.parse_arguments(
        ("--board-analysis-retry-seconds", "1.75"),
    )

    settings = autoplay.settings_from_arguments(arguments)

    assert settings.board_analysis_retry_seconds == 1.75


def test_summary_prints_all_required_counts(capsys: pytest.CaptureFixture[str]) -> None:
    """Report levels, overlays, cat targets, low-level clicks, and final state."""

    autoplay.print_autoplay_summary(
        autoplay.CatsAutoplaySummary(2, 1, 1, 16, CatsScreenState.BOARD)
    )

    output = capsys.readouterr().out
    assert "Solved levels: 2" in output
    assert "Ranking clicks: 1" in output
    assert "Level-button clicks: 1" in output
    assert "Cat targets executed: 8" in output
    assert "Low-level cat clicks: 16" in output
    assert "Final screen state: BOARD" in output


def test_autoplay_uses_shared_tile_grid_analysis_without_copying_cv_pipeline() -> None:
    """Share application analysis without any production scripts-to-scripts import."""

    autoplay_source = getsource(autoplay).casefold()
    solve_source = getsource(solve_script).casefold()
    assert "from scripts." not in autoplay_source
    assert "from scripts." not in solve_source
    assert "analyze_cats_board_with_ports" in autoplay_source
    assert "analyze_cats_board_with_ports" in solve_source
    assert "solve_cats_exact" not in autoplay_source
    assert "logicforge.plugins.cats.exact_search" not in autoplay_source


@pytest.mark.parametrize(
    "arguments",
    (
        ("--click-delay-ms", "-1"),
        ("--poll-interval-ms", "9"),
        ("--transition-timeout-seconds", "0"),
        ("--board-analysis-retry-seconds", "0"),
        ("--overlay-retry-ms", "99"),
        ("--max-overlay-retries", "0"),
        ("--new-board-delay-ms", "-1"),
        ("--max-levels", "-1"),
    ),
)
def test_cli_rejects_invalid_values(arguments: tuple[str, str]) -> None:
    """Use argparse failures for invalid timing and count options."""

    with pytest.raises(SystemExit) as error:
        autoplay.parse_arguments(arguments)

    assert error.value.code == 2


def test_source_uses_only_accepted_screen_detection() -> None:
    """Forbid duplicated vision heuristics and unsafe external technologies."""

    source = (getsource(autoplay) + getsource(cats_autoplay)).casefold()
    forbidden = (
        "cv2.inrange",
        "cv2.findcontours",
        "pytesseract",
        "easyocr",
        "matchtemplate",
        "subprocess",
        "pyautogui",
        "pynput",
    )

    for text in forbidden:
        assert text not in source
    assert "opencvcatsscreenstatedetector" in source
    assert "detection.action_point" in source
