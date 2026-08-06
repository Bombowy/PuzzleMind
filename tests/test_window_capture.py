"""Unit tests for the infrastructure-neutral window capture application flow."""

from pathlib import Path

import pytest

from logicforge.vision.window_capture import (
    CaptureResult,
    WindowBounds,
    WindowCapturer,
    WindowCaptureService,
    WindowInfo,
    WindowLocator,
)


class StubWindowLocator(WindowLocator):
    """Return a predetermined window so orchestration tests avoid native APIs."""

    def __init__(self, window: WindowInfo) -> None:
        """Store the immutable window value supplied by the test fixture."""

        self._window = window

    def locate(self) -> WindowInfo:
        """Return the configured window exactly once per service invocation."""

        return self._window


class RecordingWindowCapturer(WindowCapturer):
    """Record capture arguments and return a deterministic infrastructure result."""

    def __init__(self) -> None:
        """Initialize observable call state for later assertions."""

        self.captured_window: WindowInfo | None = None
        self.destination: Path | None = None

    def capture(self, window: WindowInfo, destination: Path) -> CaptureResult:
        """Record delegation without reading pixels or touching the filesystem."""

        self.captured_window = window
        self.destination = destination
        return CaptureResult(
            window=window,
            output_path=destination,
            resolution_width=window.bounds.width,
            resolution_height=window.bounds.height,
            elapsed_seconds=0.0,
        )


def test_window_bounds_convert_to_an_explicit_capture_box() -> None:
    """Ensure MSS receives only the requested window rectangle coordinates."""

    bounds = WindowBounds(x=120, y=80, width=800, height=600)

    assert bounds.as_bbox() == (120, 80, 920, 680)


@pytest.mark.parametrize(("width", "height"), [(0, 600), (800, 0), (-1, 600)])
def test_window_bounds_reject_non_positive_dimensions(width: int, height: int) -> None:
    """Prevent invalid native geometry from becoming a desktop capture request."""

    with pytest.raises(ValueError, match="must both be positive"):
        WindowBounds(x=0, y=0, width=width, height=height)


def test_capture_service_delegates_the_resolved_window_rectangle(
    tmp_path: Path,
) -> None:
    """Verify the application service performs lookup followed by one capture."""

    window = WindowInfo(
        title="BlueStacks App Player",
        bounds=WindowBounds(x=10, y=20, width=1280, height=720),
    )
    locator = StubWindowLocator(window)
    capturer = RecordingWindowCapturer()
    service = WindowCaptureService(locator=locator, capturer=capturer)
    destination = tmp_path / "bluestacks_capture.png"

    result = service.capture(destination)

    assert capturer.captured_window == window
    assert capturer.destination == destination
    assert result.output_path == destination
    assert result.resolution_width == 1280
    assert result.resolution_height == 720
