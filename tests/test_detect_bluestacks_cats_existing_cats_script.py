"""Tests for the single-frame, zero-click existing-cat diagnostic command."""

from pathlib import Path

import pytest
from scripts import detect_bluestacks_cats_existing_cats as script

from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowBounds, WindowInfo
from synthetic_cats_tile_grids import synthetic_cats_tile_grid

_FIXTURE = synthetic_cats_tile_grid(
    rows=6,
    columns=6,
    omitted_slots=frozenset({(1, 0)}),
    cat_sprite_slots=frozenset({(1, 0)}),
)
_TILE_GRID = OpenCvCatsTileGridDetector().detect(_FIXTURE.screenshot)
_COLORS = OpenCvColorDetector().detect(_FIXTURE.screenshot, _TILE_GRID.grid)
_EXISTING = OpenCvCatsExistingCatDetector().detect(
    _FIXTURE.screenshot,
    _TILE_GRID.grid,
    _COLORS,
)


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
        del window, debug
        type(self).capture_calls += 1
        return _FIXTURE.screenshot


class _TileDetector:
    calls = 0

    def detect(self, screenshot: Screenshot):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        assert screenshot is _FIXTURE.screenshot
        return _TILE_GRID


class _ColorDetector:
    calls = 0

    def detect(self, screenshot: Screenshot, grid):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        assert screenshot is _FIXTURE.screenshot
        assert grid is _TILE_GRID.grid
        return _COLORS


class _ExistingDetector:
    calls = 0

    def detect(self, screenshot: Screenshot, grid, colors):  # type: ignore[no-untyped-def]
        type(self).calls += 1
        assert screenshot is _FIXTURE.screenshot
        assert grid is _TILE_GRID.grid
        assert colors is _COLORS
        return _EXISTING


class _Renderer:
    calls = 0

    def save_debug_overlay(self, *args: object, **kwargs: object) -> Path:
        type(self).calls += 1
        return script.DEBUG_OUTPUT_PATH


def _configure(monkeypatch: pytest.MonkeyPatch) -> None:
    _CaptureService.capture_calls = 0
    _TileDetector.calls = 0
    _ColorDetector.calls = 0
    _ExistingDetector.calls = 0
    _Renderer.calls = 0
    monkeypatch.setattr(script, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(script, "MssWindowCapturer", object)
    monkeypatch.setattr(script, "OpenCvCatsTileGridDetector", _TileDetector)
    monkeypatch.setattr(script, "OpenCvColorDetector", _ColorDetector)
    monkeypatch.setattr(script, "OpenCvCatsExistingCatDetector", _ExistingDetector)
    monkeypatch.setattr(script, "OpenCvCatsExistingCatDebugRenderer", _Renderer)


def test_script_runs_each_read_only_stage_once_and_prints_useful_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure(monkeypatch)
    assert script.main() == 0
    output = capsys.readouterr().out
    assert "rows: 6" in output
    assert "columns: 6" in output
    assert "cell count: 36" in output
    assert "color_count: 6" in output
    assert "count: 1" in output
    assert "(1,0)" in output
    assert "foreground_ratio=" in output
    assert _CaptureService.capture_calls == 1
    assert _TileDetector.calls == 1
    assert _ColorDetector.calls == 1
    assert _ExistingDetector.calls == 1
    assert _Renderer.calls == 1
    source = Path(script.__file__).read_text(encoding="utf-8").lower()
    assert "mouse" not in source
    assert "pyautogui" not in source
    assert "pynput" not in source
