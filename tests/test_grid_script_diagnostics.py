"""Deterministic tests for user-facing grid-script failure diagnostics."""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from scripts import detect_bluestacks_grid as grid_script

from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionDiagnostics,
    BoardDetectionError,
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


class _CaptureService:
    """Return injected in-memory captures through the script's service surface."""

    screenshot: Screenshot
    capture_error: WindowCaptureError | None = None

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

        return self.screenshot


class _BoardDetector:
    """Return one board or raise an injected typed failure."""

    error: BoardDetectionError | None = None

    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Exercise the script's board success and failure branches."""

        if self.error is not None:
            raise self.error
        return BoardDetection(x=0, y=0, width=30, height=30, confidence=1.0)


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
