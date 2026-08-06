"""Tests for the one-shot BlueStacks Cats screen-state diagnostic script."""

from datetime import UTC, datetime
from inspect import getsource
from pathlib import Path

import numpy as np
import pytest
from scripts import detect_bluestacks_cats_screen_state as state_script

from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
)
from logicforge.infrastructure.opencv_cats_screen_state_renderer import (
    CatsScreenStateDebugRenderError,
)
from logicforge.plugins.cats import (
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDiagnostics,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowCaptureError,
    WindowInfo,
)


def _screenshot() -> Screenshot:
    """Return one immutable synthetic capture without desktop access."""

    return Screenshot(
        image=np.zeros((600, 800, 3), dtype=np.uint8),
        width=800,
        height=600,
        timestamp=datetime(2026, 8, 6, tzinfo=UTC),
    )


def _detection(state: CatsScreenState) -> CatsScreenStateDetection:
    """Build one valid public result for each script output branch."""

    level_button = (
        CatsScreenRect(180, 500, 440, 60)
        if state is CatsScreenState.LEVEL_COMPLETE
        else None
    )
    ranking_cards = (
        (
            CatsScreenRect(200, 180, 400, 60),
            CatsScreenRect(200, 270, 400, 60),
            CatsScreenRect(200, 360, 400, 60),
        )
        if state is CatsScreenState.RANKING
        else ()
    )
    board = (
        CatsScreenRect(220, 120, 360, 360) if state is CatsScreenState.BOARD else None
    )
    action = (
        CatsScreenPoint(400, 530)
        if state is CatsScreenState.LEVEL_COMPLETE
        else CatsScreenPoint(400, 500) if state is CatsScreenState.RANKING else None
    )
    confidence = 0.0 if state is CatsScreenState.UNKNOWN else 0.9
    return CatsScreenStateDetection(
        state=state,
        confidence=confidence,
        action_point=action,
        diagnostics=CatsScreenStateDiagnostics(
            game_viewport_candidate=CatsScreenRect(120, 20, 340, 580),
            game_viewport_score=0.82,
            level_button_candidate=level_button,
            level_button_score=(0.9 if level_button is not None else 0.0),
            ranking_card_candidates=ranking_cards,
            ranking_score=(0.9 if ranking_cards else 0.0),
            board_candidate=board,
            board_confidence=(0.92 if board is not None else None),
            grid_confidence=(0.90 if board is not None else None),
            detected_rows=(8 if board is not None else None),
            detected_columns=(8 if board is not None else None),
            rejection_reasons=(
                ("synthetic viewport-relative rejection",)
                if state is CatsScreenState.UNKNOWN
                else ()
            ),
        ),
    )


class _FakeCaptureService:
    """Record one window lookup and one in-memory capture."""

    capture_error: WindowCaptureError | None = None
    locate_calls = 0
    capture_calls = 0
    screenshot = _screenshot()
    window = WindowInfo(
        title="BlueStacks App Player",
        bounds=WindowBounds(x=300, y=200, width=800, height=600),
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept production composition arguments without native access."""

    def locate_window(self) -> WindowInfo:
        """Return deterministic shifted desktop geometry or a typed failure."""

        type(self).locate_calls += 1
        if self.capture_error is not None:
            raise self.capture_error
        return self.window

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        """Return one screenshot while asserting capture debug remains disabled."""

        del window
        assert debug is False
        type(self).capture_calls += 1
        return self.screenshot


class _FakeDetector:
    """Return one configured public result or typed processing failure."""

    result = _detection(CatsScreenState.UNKNOWN)
    error: CatsScreenStateDetectionError | None = None
    instances = 0
    calls = 0

    def __init__(self) -> None:
        """Count concrete detector composition without retaining input state."""

        type(self).instances += 1

    def detect(self, screenshot: Screenshot) -> CatsScreenStateDetection:
        """Count one classification and return the configured branch."""

        del screenshot
        type(self).calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class _FakeRenderer:
    """Record one explicit debug save without filesystem or OpenCV access."""

    error: CatsScreenStateDebugRenderError | None = None
    instances = 0
    calls = 0

    def __init__(self) -> None:
        """Count renderer composition."""

        type(self).instances += 1

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Count one requested render and return its resolved destination."""

        del screenshot, detection
        assert debug is True
        type(self).calls += 1
        if self.error is not None:
            raise self.error
        return destination.resolve()


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: CatsScreenState = CatsScreenState.UNKNOWN,
    capture_error: WindowCaptureError | None = None,
    detection_error: CatsScreenStateDetectionError | None = None,
    render_error: CatsScreenStateDebugRenderError | None = None,
) -> None:
    """Reset and inject every external collaborator used by the script."""

    _FakeCaptureService.capture_error = capture_error
    _FakeCaptureService.locate_calls = 0
    _FakeCaptureService.capture_calls = 0
    _FakeDetector.result = _detection(state)
    _FakeDetector.error = detection_error
    _FakeDetector.instances = 0
    _FakeDetector.calls = 0
    _FakeRenderer.error = render_error
    _FakeRenderer.instances = 0
    _FakeRenderer.calls = 0
    monkeypatch.setattr(state_script, "WindowCaptureService", _FakeCaptureService)
    monkeypatch.setattr(state_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(state_script, "MssWindowCapturer", object)
    monkeypatch.setattr(state_script, "OpenCvCatsScreenStateDetector", _FakeDetector)
    monkeypatch.setattr(
        state_script,
        "OpenCvCatsScreenStateDebugRenderer",
        _FakeRenderer,
    )


def test_main_performs_one_capture_classification_and_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the diagnostic pipeline strictly one-shot at every stage."""

    _configure(monkeypatch, state=CatsScreenState.BOARD)

    assert state_script.main() == 0
    assert _FakeCaptureService.locate_calls == 1
    assert _FakeCaptureService.capture_calls == 1
    assert _FakeDetector.instances == 1
    assert _FakeDetector.calls == 1
    assert _FakeRenderer.instances == 1
    assert _FakeRenderer.calls == 1


def test_board_output_contains_rows_and_columns(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print delegated board and grid geometry for BOARD classification."""

    _configure(monkeypatch, state=CatsScreenState.BOARD)

    assert state_script.main() == 0
    output = capsys.readouterr().out
    assert "State: BOARD" in output
    assert "Grid: 8 rows x 8 columns" in output


def test_ranking_output_contains_screenshot_and_desktop_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate the diagnostic screenshot point only for printed desktop output."""

    _configure(monkeypatch, state=CatsScreenState.RANKING)

    assert state_script.main() == 0
    output = capsys.readouterr().out
    assert "Ranking cards detected: 3" in output
    assert "Action screenshot: (400, 500)" in output
    assert "Action desktop: (700, 700)" in output


def test_level_complete_output_contains_button_and_desktop_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print detected button bounds and the translated future action point."""

    _configure(monkeypatch, state=CatsScreenState.LEVEL_COMPLETE)

    assert state_script.main() == 0
    output = capsys.readouterr().out
    assert "Level button: x=180, y=500, width=440, height=60" in output
    assert "Action screenshot: (400, 530)" in output
    assert "Action desktop: (700, 730)" in output


def test_unknown_prints_no_action_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Treat an unrecognized but processed frame as a normal successful outcome."""

    _configure(monkeypatch, state=CatsScreenState.UNKNOWN)

    assert state_script.main() == 0
    output = capsys.readouterr().out
    assert "State: UNKNOWN" in output
    assert "Action screenshot: none" in output
    assert "Rejection reasons:" in output
    assert "- synthetic viewport-relative rejection" in output


def test_output_always_contains_global_viewport_and_score(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print full-screenshot viewport geometry and its bounded score."""

    _configure(monkeypatch, state=CatsScreenState.RANKING)

    assert state_script.main() == 0
    output = capsys.readouterr().out
    assert "Game viewport: x=120, y=20, width=340, height=580" in output
    assert "Game viewport score: 0.820" in output


def test_capture_error_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated one-shot capture failure boundary."""

    _configure(
        monkeypatch,
        capture_error=WindowCaptureError("synthetic capture failure"),
    )

    assert state_script.main() == 1
    assert "BlueStacks capture failed" in capsys.readouterr().err


def test_detection_error_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose only genuine detector processing failures as exit code two."""

    _configure(
        monkeypatch,
        detection_error=CatsScreenStateDetectionError("synthetic detection failure"),
    )

    assert state_script.main() == 2
    assert "Cats screen-state detection failed" in capsys.readouterr().err


def test_renderer_error_returns_three(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose an explicit overlay persistence failure as exit code three."""

    _configure(
        monkeypatch,
        render_error=CatsScreenStateDebugRenderError("synthetic render failure"),
    )

    assert state_script.main() == 3
    assert "Cats screen-state debug rendering failed" in capsys.readouterr().err


def test_script_contains_no_mouse_or_click_technology() -> None:
    """Keep this diagnostic command completely independent from input emission."""

    source = getsource(state_script)

    for forbidden in (
        "MouseController",
        "Win32MouseController",
        "ScreenPoint",
        "pyautogui",
        "pynput",
        ".click(",
        "mouse_event",
        "SetCursorPos",
    ):
        assert forbidden not in source
