"""Tests for the one-frame Cats tile-grid diagnostic script."""

from pathlib import Path

import pytest
from scripts import detect_bluestacks_cats_tile_grid as tile_script

from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowBounds, WindowInfo
from synthetic_cats_tile_grids import synthetic_cats_tile_grid


class _CaptureService:
    """Expose one immutable screenshot without touching a desktop."""

    screenshot: Screenshot
    capture_calls = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def locate_window(self) -> WindowInfo:
        return WindowInfo(
            "BlueStacks App Player",
            WindowBounds(0, 0, self.screenshot.width, self.screenshot.height),
        )

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        type(self).capture_calls += 1
        return self.screenshot


class _Detector:
    calls = 0
    detection = OpenCvCatsTileGridDetector().detect(
        synthetic_cats_tile_grid(rows=5, columns=5).screenshot
    )

    def detect(self, screenshot: Screenshot):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return self.detection


class _Renderer:
    calls = 0

    def save_debug_overlay(self, *args: object, **kwargs: object) -> Path:
        type(self).calls += 1
        return tile_script.DEBUG_OUTPUT_PATH


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = synthetic_cats_tile_grid(rows=5, columns=5)
    _CaptureService.screenshot = fixture.screenshot
    _CaptureService.capture_calls = 0
    _Detector.calls = 0
    _Renderer.calls = 0
    monkeypatch.setattr(tile_script, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(tile_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(tile_script, "MssWindowCapturer", object)
    monkeypatch.setattr(tile_script, "OpenCvCatsTileGridDetector", _Detector)
    monkeypatch.setattr(tile_script, "OpenCvCatsTileGridDebugRenderer", _Renderer)


def test_script_prints_complete_lattice_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose raw counts, fitted geometry, occupancy, confidence, and artifact path."""

    _configure(monkeypatch)

    assert tile_script.main() == 0
    output = capsys.readouterr().out
    for label in (
        "Window title:",
        "Screenshot resolution:",
        "Raw components:",
        "Tile candidates:",
        "Accepted tiles: 25",
        "Rows: 5",
        "Columns: 5",
        "Median tile width:",
        "Median tile height:",
        "Horizontal pitch:",
        "Vertical pitch:",
        "Horizontal pitch CV:",
        "Vertical pitch CV:",
        "Occupancy: 1.000",
        "Board: x=",
        "Cell count: 25",
        "Grid confidence:",
        "Debug output path:",
    ):
        assert label in output


def test_script_captures_and_detects_exactly_once_without_mouse_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the diagnostic command single-frame and incapable of clicking."""

    _configure(monkeypatch)

    assert tile_script.main() == 0
    assert _CaptureService.capture_calls == 1
    assert _Detector.calls == 1
    assert _Renderer.calls == 1
    source = Path(tile_script.__file__).read_text(encoding="utf-8").lower()
    assert "mousecontroller" not in source
    assert "pyautogui" not in source
    assert "pynput" not in source
