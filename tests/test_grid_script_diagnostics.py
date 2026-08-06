"""Deterministic tests for user-facing grid-script failure diagnostics."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from scripts import detect_bluestacks_grid as grid_script

from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionAnalysis,
    BoardDetectionDiagnostics,
    BoardDetectionError,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionDiagnostics,
    GridDetectionError,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowCaptureError,
    WindowInfo,
)

DetectorFactory = Callable[[object], object]


def _screenshot(width: int, height: int) -> Screenshot:
    """Create an immutable blank capture without accessing a desktop."""

    return Screenshot(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        width=width,
        height=height,
        timestamp=datetime.now(UTC),
    )


def _board_error() -> BoardDetectionError:
    """Create a representative typed board failure for script tests."""

    return BoardDetectionError(
        "synthetic board failure",
        BoardDetectionDiagnostics(
            contour_count=0,
            candidates=(),
            selected_candidate=None,
            competitive_candidate_count=0,
        ),
    )


def _grid_error() -> GridDetectionError:
    """Create a representative typed grid failure for script tests."""

    return GridDetectionError(
        "synthetic grid failure",
        GridDetectionDiagnostics(
            board_x=0,
            board_y=0,
            board_width=30,
            board_height=30,
            normalized_horizontal_positions=(),
            normalized_vertical_positions=(),
            horizontal_lines=(),
            vertical_lines=(),
            estimated_rows=0,
            estimated_columns=0,
            horizontal_spacing_coefficient_of_variation=0.0,
            vertical_spacing_coefficient_of_variation=0.0,
            horizontal_coverage=0.0,
            vertical_coverage=0.0,
            grid_evidence_score=0.0,
            rejection_reasons=("insufficient horizontal grid lines",),
        ),
    )


def _valid_grid() -> GridDetection:
    """Return minimal complete geometry for exercising the render-failure path."""

    return GridDetection(
        horizontal_lines=(0, 30),
        vertical_lines=(0, 30),
        rows=1,
        columns=1,
        cells=(
            CellBounds(
                row=0,
                column=0,
                x=0,
                y=0,
                width=30,
                height=30,
                center_x=15,
                center_y=15,
            ),
        ),
        confidence=1.0,
    )


def _refinement_attempt(
    direction: str,
    *,
    accepted: bool,
    rejection_reasons: tuple[str, ...] = (),
) -> BoardEnvelopeRefinementDiagnostic:
    """Build one complete primitive attempt for script-output tests."""

    return BoardEnvelopeRefinementDiagnostic(
        seed_x=337,
        seed_y=266,
        seed_width=478,
        seed_height=531,
        refined_x=337,
        refined_y=266,
        refined_width=531,
        refined_height=531,
        direction=direction,
        added_pixels=53,
        seed_rows=10,
        seed_columns=9,
        refined_rows=10,
        refined_columns=10,
        old_border_match_score=0.91,
        separator_continuation_score=0.62,
        supported_separator_fraction=0.70,
        spacing_score=0.83,
        refined_grid_score=0.90,
        refinement_score=0.72,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
    )


class _CaptureService:
    """Return injected in-memory captures through the script's service surface."""

    screenshot: Screenshot
    capture_error: WindowCaptureError | None = None
    capture_calls = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept production composition arguments without using desktop adapters."""

    def locate_window(self) -> WindowInfo:
        """Return stable synthetic window metadata or the configured failure."""

        if self.capture_error is not None:
            raise self.capture_error
        return WindowInfo(
            title="BlueStacks App Player",
            bounds=WindowBounds(0, 0, self.screenshot.width, self.screenshot.height),
        )

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        """Return the configured screenshot without monitor access."""

        type(self).capture_calls += 1
        return self.screenshot


class _BoardDetector:
    """Return one board or raise an injected typed failure."""

    error: BoardDetectionError | None = None

    def analyze(self, screenshot: Screenshot) -> BoardDetectionAnalysis:
        """Exercise the script's board success and failure branches."""

        if self.error is not None:
            raise self.error
        detection = BoardDetection(x=0, y=0, width=30, height=30, confidence=1.0)
        return BoardDetectionAnalysis(
            detection=detection,
            diagnostics=BoardDetectionDiagnostics(0, (), None, 0),
        )


class _GridDetector:
    """Return complete geometry or raise an injected typed failure."""

    error: GridDetectionError | None = None

    def detect(self, screenshot: Screenshot, board: BoardDetection) -> GridDetection:
        """Exercise the script's grid success and failure branches."""

        if self.error is not None:
            raise self.error
        return _valid_grid()


class _Renderer:
    """Avoid filesystem writes while exposing success and failure render paths."""

    return_success_path = True

    def save_failure_debug_overlay(self, *args: object, **kwargs: object) -> Path:
        """Represent a successfully generated rejected-candidate overlay."""

        return grid_script.DEBUG_OUTPUT_PATH

    def save_debug_overlay(self, *args: object, **kwargs: object) -> Path | None:
        """Allow the test to trigger the script's established render exit code."""

        if self.return_success_path:
            return grid_script.DEBUG_OUTPUT_PATH
        return None


def _factory(instance: object) -> DetectorFactory:
    """Adapt one fake detector instance to a settings-accepting constructor."""

    def create(settings: object) -> object:
        return instance

    return create


def _configure_script(
    monkeypatch: pytest.MonkeyPatch,
    screenshot: Screenshot,
    *,
    board_error: BoardDetectionError | None = None,
    grid_error: GridDetectionError | None = None,
    capture_error: WindowCaptureError | None = None,
    render_success: bool = True,
) -> None:
    """Replace every OS/OpenCV collaborator with a deterministic in-memory fake."""

    _CaptureService.screenshot = screenshot
    _CaptureService.capture_error = capture_error
    _CaptureService.capture_calls = 0
    board_detector = _BoardDetector()
    board_detector.error = board_error
    grid_detector = _GridDetector()
    grid_detector.error = grid_error
    renderer = _Renderer()
    renderer.return_success_path = render_success

    monkeypatch.setattr(grid_script, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(grid_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(grid_script, "MssWindowCapturer", object)
    monkeypatch.setattr(grid_script, "OpenCvBoardDetector", _factory(board_detector))
    monkeypatch.setattr(grid_script, "OpenCvGridDetector", _factory(grid_detector))
    monkeypatch.setattr(
        grid_script,
        "OpenCvGridDetectionDebugRenderer",
        lambda: renderer,
    )


@pytest.mark.parametrize(
    ("width", "height"),
    ((439, 470), (440, 469)),
)
def test_screenshot_below_either_recommended_dimension_is_advisory(
    width: int, height: int
) -> None:
    """Treat either undersized dimension as useful post-failure context."""

    assert grid_script.is_screenshot_below_recommended_size(_screenshot(width, height))


@pytest.mark.parametrize(
    ("width", "height"),
    ((440, 470), (600, 700)),
)
def test_screenshot_at_or_above_recommendation_is_not_flagged(
    width: int, height: int
) -> None:
    """Keep the exact recommendation inclusive and avoid warnings above it."""

    assert not grid_script.is_screenshot_below_recommended_size(
        _screenshot(width, height)
    )


def test_small_board_failure_keeps_error_and_prints_resize_advice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Add actionable context after a board failure without changing exit code 2."""

    _configure_script(monkeypatch, _screenshot(439, 470), board_error=_board_error())

    assert grid_script.main() == 2
    error_output = capsys.readouterr().err
    assert "Board detection failed: synthetic board failure" in error_output
    assert "BlueStacks window may be too small for reliable detection." in error_output
    assert "Captured resolution: 439x470." in error_output
    assert "Recommended minimum: 440x470." in error_output


def test_small_grid_failure_keeps_error_and_prints_resize_advice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Add the same advice after grid failure while preserving exit code 3."""

    _configure_script(monkeypatch, _screenshot(440, 469), grid_error=_grid_error())

    assert grid_script.main() == 3
    error_output = capsys.readouterr().err
    assert "Grid detection failed after" in error_output
    assert "synthetic grid failure" in error_output
    assert "BlueStacks window may be too small for reliable detection." in error_output
    assert "Captured resolution: 440x469." in error_output


def test_large_board_failure_does_not_print_size_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Avoid attributing failures to size at the recommended resolution."""

    _configure_script(monkeypatch, _screenshot(440, 470), board_error=_board_error())

    assert grid_script.main() == 2
    error_output = capsys.readouterr().err
    assert "synthetic board failure" in error_output
    assert "may be too small" not in error_output


def test_capture_failure_exit_code_remains_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep capture failures distinct from detection and render failures."""

    _configure_script(
        monkeypatch,
        _screenshot(440, 470),
        capture_error=WindowCaptureError("synthetic capture failure"),
    )

    assert grid_script.main() == 1
    assert "synthetic capture failure" in capsys.readouterr().err


def test_render_failure_exit_code_remains_four(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve the explicit missing-debug-output failure status."""

    _configure_script(monkeypatch, _screenshot(440, 470), render_success=False)

    assert grid_script.main() == 4
    assert "Grid debug rendering produced no output path." in capsys.readouterr().err


def test_success_output_lists_lines_and_cell_extents(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose final recovered geometry without capturing or writing another image."""

    screenshot = _screenshot(30, 30)
    window = WindowInfo("BlueStacks App Player", WindowBounds(0, 0, 30, 30))
    board = BoardDetection(0, 0, 30, 30, 1.0)

    grid_script.print_detection_information(
        window,
        screenshot,
        board,
        _valid_grid(),
        0.01,
        0.02,
    )

    output = capsys.readouterr().out
    assert "Horizontal lines: (0, 30)" in output
    assert "Vertical lines: (0, 30)" in output
    assert "Row heights: (30,)" in output
    assert "Column widths: (30,)" in output
    assert "Board envelope refined: no" in output
    assert "Envelope refinement attempts: 0" in output


def test_refined_success_output_lists_seed_direction_and_scores(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose selected maximal-envelope evidence beside final public 9x9 geometry."""

    screenshot = _screenshot(90, 90)
    window = WindowInfo("BlueStacks App Player", WindowBounds(0, 0, 90, 90))
    board = BoardDetection(0, 0, 90, 90, 0.9)
    refinement = BoardEnvelopeRefinementDiagnostic(
        seed_x=0,
        seed_y=0,
        seed_width=80,
        seed_height=90,
        refined_x=0,
        refined_y=0,
        refined_width=90,
        refined_height=90,
        direction="right",
        added_pixels=10,
        seed_rows=9,
        seed_columns=8,
        refined_rows=9,
        refined_columns=9,
        old_border_match_score=0.95,
        separator_continuation_score=0.80,
        supported_separator_fraction=1.0,
        spacing_score=0.90,
        refined_grid_score=0.93,
        refinement_score=0.88,
        accepted=True,
    )
    diagnostics = BoardDetectionDiagnostics(
        1,
        (),
        None,
        1,
        envelope_refinements=(refinement,),
        selected_refinement=refinement,
    )
    lines = tuple(range(0, 91, 10))
    cells = tuple(
        CellBounds(
            row, column, column * 10, row * 10, 10, 10, column * 10 + 5, row * 10 + 5
        )
        for row in range(9)
        for column in range(9)
    )
    grid = GridDetection(lines, lines, 9, 9, cells, 0.93)

    grid_script.print_detection_information(
        window,
        screenshot,
        board,
        grid,
        0.01,
        0.02,
        diagnostics,
    )

    output = capsys.readouterr().out
    assert "Board envelope refined: yes" in output
    assert "Seed board: x=0, y=0, width=80, height=90" in output
    assert "Refinement direction: right" in output
    assert "Added pixels: 10" in output
    assert "Seed grid: 9x8" in output
    assert "Refined grid: 9x9" in output
    assert "Separator continuation score: 0.800" in output
    assert "Supported separator fraction: 1.000" in output
    assert "Refinement score: 0.880" in output
    assert "Detected rows: 9" in output
    assert "Detected columns: 9" in output
    assert "Envelope refinement attempts: 1" in output
    assert "Refinement attempt: right" in output
    assert "Old-border match score: 0.950" in output
    assert "Spacing score: 0.900" in output
    assert "Refined grid score: 0.930" in output
    assert "Accepted: yes" in output
    assert "Rejection reasons: none" in output


def test_unselected_grid_refinement_prints_all_rejected_attempts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep every rejected direction visible for live envelope calibration."""

    attempts = tuple(
        _refinement_attempt(
            direction,
            accepted=False,
            rejection_reasons=(
                f"{direction} failed mandatory evidence",
                "refined grid score dropped beyond configured tolerance",
            ),
        )
        for direction in ("top", "bottom", "right", "left")
    )
    diagnostics = BoardDetectionDiagnostics(
        contour_count=1,
        candidates=(),
        selected_candidate=None,
        competitive_candidate_count=1,
        envelope_refinements=attempts,
        selected_refinement=None,
    )

    grid_script.print_envelope_refinement_information(diagnostics)

    output = capsys.readouterr().out
    assert "Board envelope refined: no" in output
    assert "Envelope refinement attempts: 4" in output
    offsets = tuple(
        output.index(f"Refinement attempt: {direction}")
        for direction in ("left", "right", "top", "bottom")
    )
    assert offsets == tuple(sorted(offsets))
    assert "Seed rectangle: x=337, y=266, width=478, height=531" in output
    assert "Candidate rectangle: x=337, y=266, width=531, height=531" in output
    assert "Added pixels: 53" in output
    assert "Seed grid: 10x9" in output
    assert "Candidate grid: 10x10" in output
    assert "Old-border match score: 0.910" in output
    assert "Separator continuation score: 0.620" in output
    assert "Supported separator fraction: 0.700" in output
    assert "Spacing score: 0.830" in output
    assert "Refined grid score: 0.900" in output
    assert "Refinement score: 0.720" in output
    assert output.count("Accepted: no") == 4
    for direction in ("left", "right", "top", "bottom"):
        assert f"- {direction} failed mandatory evidence" in output
    assert output.count("- refined grid score dropped beyond configured tolerance") == 4


def test_success_main_still_captures_once_and_contains_no_clicks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the refined diagnostic workflow one-shot and input-free."""

    _configure_script(monkeypatch, _screenshot(440, 470))

    assert grid_script.main() == 0
    assert _CaptureService.capture_calls == 1
    source = Path(grid_script.__file__).read_text(encoding="utf-8").casefold()
    assert "mousecontroller" not in source
    assert ".click(" not in source
