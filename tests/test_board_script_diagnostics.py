"""One-shot board-script diagnostics for maximal grid envelopes."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from scripts import detect_bluestacks_board as board_script

from logicforge.config.settings import BoardDetectionSettings
from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionAnalysis,
    BoardDetectionDiagnostics,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowBounds, WindowInfo


def _screenshot() -> Screenshot:
    """Return one immutable in-memory capture."""

    return Screenshot(
        image=np.zeros((90, 90, 3), dtype=np.uint8),
        width=90,
        height=90,
        timestamp=datetime.now(UTC),
    )


def _refinement(
    direction: str = "right",
    *,
    accepted: bool = True,
    rejection_reasons: tuple[str, ...] = (),
) -> BoardEnvelopeRefinementDiagnostic:
    """Build one stable refinement attempt for terminal-output assertions."""

    return BoardEnvelopeRefinementDiagnostic(
        0,
        0,
        80,
        90,
        0,
        0,
        90,
        90,
        direction,
        10,
        9,
        8,
        9,
        9,
        0.95,
        0.80,
        1.0,
        0.90,
        0.93,
        0.88,
        accepted,
        rejection_reasons,
    )


def test_board_script_prints_selected_refinement_and_final_board(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print seed, direction, support, and score beside final public bounds."""

    refinement = _refinement()
    diagnostics = BoardDetectionDiagnostics(
        1,
        (),
        None,
        1,
        envelope_refinements=(refinement,),
        selected_refinement=refinement,
    )

    board_script.print_detection_information(
        WindowInfo("BlueStacks App Player", WindowBounds(0, 0, 90, 90)),
        _screenshot(),
        BoardDetection(0, 0, 90, 90, 0.9),
        0.01,
        Path("artifacts/vision/board_detection.png"),
        diagnostics,
    )

    output = capsys.readouterr().out
    assert "Board size: width=90, height=90" in output
    assert "Board envelope refined: yes" in output
    assert "Seed board: x=0, y=0, width=80, height=90" in output
    assert "Refinement direction: right" in output
    assert "Added pixels: 10" in output
    assert "Seed grid: 9x8" in output
    assert "Refined grid: 9x9" in output
    assert "Separator continuation score: 0.800" in output
    assert "Supported separator fraction: 1.000" in output
    assert "Refinement score: 0.880" in output
    assert "Envelope refinement attempts: 1" in output
    assert "Accepted: yes" in output
    assert "Rejection reasons: none" in output


def test_board_script_prints_no_refinement_for_complete_contour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use an explicit negative diagnostic for an already-complete board."""

    board_script.print_detection_information(
        WindowInfo("BlueStacks App Player", WindowBounds(0, 0, 90, 90)),
        _screenshot(),
        BoardDetection(0, 0, 90, 90, 0.9),
        0.01,
        Path("artifacts/vision/board_detection.png"),
        BoardDetectionDiagnostics(1, (), None, 1),
    )

    output = capsys.readouterr().out
    assert "Board envelope refined: no" in output
    assert "Envelope refinement attempts: 0" in output


def test_board_script_prints_every_rejected_attempt_in_direction_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose complete rejected evidence even when no refinement was selected."""

    attempts = tuple(
        _refinement(
            direction,
            accepted=False,
            rejection_reasons=(f"{direction} synthetic rejection", "shared reason"),
        )
        for direction in ("bottom", "right", "left", "top")
    )
    diagnostics = BoardDetectionDiagnostics(
        1,
        (),
        None,
        1,
        envelope_refinements=attempts,
        selected_refinement=None,
    )

    board_script.print_envelope_refinement_information(diagnostics)

    output = capsys.readouterr().out
    assert "Board envelope refined: no" in output
    assert "Envelope refinement attempts: 4" in output
    offsets = tuple(
        output.index(f"Refinement attempt: {direction}")
        for direction in ("left", "right", "top", "bottom")
    )
    assert offsets == tuple(sorted(offsets))
    assert "Seed rectangle: x=0, y=0, width=80, height=90" in output
    assert "Candidate rectangle: x=0, y=0, width=90, height=90" in output
    assert "Added pixels: 10" in output
    assert "Seed grid: 9x8" in output
    assert "Candidate grid: 9x9" in output
    assert "Old-border match score: 0.950" in output
    assert "Separator continuation score: 0.800" in output
    assert "Supported separator fraction: 1.000" in output
    assert "Spacing score: 0.900" in output
    assert "Refined grid score: 0.930" in output
    assert "Refinement score: 0.880" in output
    assert output.count("Accepted: no") == 4
    for direction in ("left", "right", "top", "bottom"):
        assert f"- {direction} synthetic rejection" in output
    assert output.count("- shared reason") == 4


def test_refinement_threshold_defaults_remain_calibrated() -> None:
    """Lock diagnostics work against accidental heuristic calibration changes."""

    settings = BoardDetectionSettings()
    assert settings.minimum_grid_line_response == 0.30
    assert settings.grid_envelope_minimum_added_size_ratio == 0.75
    assert settings.grid_envelope_maximum_added_size_ratio == 1.25
    assert settings.grid_envelope_separator_position_tolerance_ratio == 0.18
    assert settings.grid_envelope_continuation_probe_thickness_ratio == 0.08
    assert settings.grid_envelope_minimum_line_continuation_response == 0.12
    assert settings.grid_envelope_minimum_supported_separator_fraction == 0.65
    assert settings.grid_envelope_maximum_spacing_cv_increase == 0.03
    assert settings.grid_envelope_maximum_grid_score_drop == 0.08
    assert settings.grid_envelope_minimum_refinement_score == 0.63
    assert settings.grid_envelope_ambiguity_delta == 0.03


def test_board_script_main_captures_once_and_never_clicks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain one-shot capture and input-free diagnostics after refinement support."""

    screenshot = _screenshot()

    class CaptureService:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def locate_window(self) -> WindowInfo:
            return WindowInfo("BlueStacks App Player", WindowBounds(0, 0, 90, 90))

        def capture_window(
            self,
            window: WindowInfo,
            *,
            debug: bool = False,
        ) -> Screenshot:
            del window, debug
            type(self).calls += 1
            return screenshot

    class Detector:
        def analyze(self, captured: Screenshot) -> BoardDetectionAnalysis:
            assert captured is screenshot
            detection = BoardDetection(0, 0, 90, 90, 0.9)
            return BoardDetectionAnalysis(
                detection,
                BoardDetectionDiagnostics(1, (), None, 1),
            )

    class Renderer:
        draw_rejected_candidates: bool | None = None

        def save_debug_overlay(self, *args: object, **kwargs: object) -> Path:
            del args
            type(self).draw_rejected_candidates = bool(
                kwargs["draw_rejected_candidates"]
            )
            return board_script.DEBUG_OUTPUT_PATH

    monkeypatch.setattr(board_script, "WindowCaptureService", CaptureService)
    monkeypatch.setattr(board_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(board_script, "MssWindowCapturer", object)
    monkeypatch.setattr(board_script, "OpenCvBoardDetector", Detector)
    monkeypatch.setattr(board_script, "OpenCvBoardDetectionDebugRenderer", Renderer)

    assert board_script.main() == 0
    assert CaptureService.calls == 1
    assert Renderer.draw_rejected_candidates is True
    source = Path(board_script.__file__).read_text(encoding="utf-8").casefold()
    assert "mousecontroller" not in source
    assert ".click(" not in source
