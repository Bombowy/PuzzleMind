"""Tests for Cats tile-grid-to-existing-color diagnostic composition."""

from pathlib import Path

import pytest
from scripts import detect_bluestacks_cats_colors as color_script

from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowBounds, WindowInfo
from synthetic_cats_tile_grids import CATS_TILE_PALETTE, synthetic_cats_tile_grid

_FIXTURE = synthetic_cats_tile_grid(
    rows=9,
    columns=9,
    palette=CATS_TILE_PALETTE[:9],
)
_TILE_GRID = OpenCvCatsTileGridDetector().detect(_FIXTURE.screenshot)
_COLORS = OpenCvColorDetector().detect(_FIXTURE.screenshot, _TILE_GRID.grid)


class _CaptureService:
    capture_calls = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def locate_window(self) -> WindowInfo:
        return WindowInfo(
            "BlueStacks App Player",
            WindowBounds(0, 0, _FIXTURE.screenshot.width, _FIXTURE.screenshot.height),
        )

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        type(self).capture_calls += 1
        return _FIXTURE.screenshot


class _TileDetector:
    calls = 0

    def detect(self, screenshot: Screenshot):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return _TILE_GRID


class _ColorDetector:
    calls = 0

    def __init__(self, settings: object) -> None:
        del settings

    def detect(self, screenshot: Screenshot, grid):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        return _COLORS


class _Renderer:
    calls = 0

    def __init__(self, settings: object) -> None:
        del settings

    def save_debug_overlay(self, *args: object, **kwargs: object) -> Path:
        type(self).calls += 1
        return color_script.DEBUG_OUTPUT_PATH


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    _CaptureService.capture_calls = 0
    _TileDetector.calls = 0
    _ColorDetector.calls = 0
    _Renderer.calls = 0
    monkeypatch.setattr(color_script, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(color_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(color_script, "MssWindowCapturer", object)
    monkeypatch.setattr(color_script, "OpenCvCatsTileGridDetector", _TileDetector)
    monkeypatch.setattr(color_script, "OpenCvColorDetector", _ColorDetector)
    monkeypatch.setattr(color_script, "OpenCvColorDetectionDebugRenderer", _Renderer)


def test_script_prints_nine_by_nine_color_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose the geometry, 81 cells, nine classes, matrix, and confidence."""

    _configure(monkeypatch)

    assert color_script.main() == 0
    output = capsys.readouterr().out
    assert "Grid: 9x9" in output
    assert "Cells: 81" in output
    assert "Detected color classes: 9" in output
    assert "Color matrix:" in output
    assert "Color confidence:" in output


def test_script_uses_one_capture_one_tile_fit_one_color_pass_and_never_clicks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the manual command in-memory, single-frame, and read-only."""

    _configure(monkeypatch)

    assert color_script.main() == 0
    assert _CaptureService.capture_calls == 1
    assert _TileDetector.calls == 1
    assert _ColorDetector.calls == 1
    assert _Renderer.calls == 1
    source = Path(color_script.__file__).read_text(encoding="utf-8").lower()
    assert "mousecontroller" not in source
    assert "pyautogui" not in source
    assert "pynput" not in source
