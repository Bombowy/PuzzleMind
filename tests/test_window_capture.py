"""Unit tests for the in-memory window capture application flow."""

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.infrastructure.opencv_debug_image import save_debug_image
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowCaptureError,
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
    """Record capture arguments and return a deterministic in-memory screenshot."""

    def __init__(self, screenshot: Screenshot) -> None:
        """Store the screenshot returned by every test capture invocation."""

        self._screenshot = screenshot
        self.captured_window: WindowInfo | None = None

    def capture(self, window: WindowInfo) -> Screenshot:
        """Record delegation without invoking MSS or touching the filesystem."""

        self.captured_window = window
        return self._screenshot


@pytest.fixture
def window() -> WindowInfo:
    """Provide stable BlueStacks geometry shared by service tests."""

    return WindowInfo(
        title="BlueStacks App Player",
        bounds=WindowBounds(x=10, y=20, width=4, height=3),
    )


@pytest.fixture
def screenshot() -> Screenshot:
    """Provide a small immutable BGR screenshot without native capture."""

    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    return Screenshot(
        image=image,
        width=4,
        height=3,
        timestamp=datetime.now(UTC),
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


def test_capture_service_returns_the_in_memory_screenshot(
    window: WindowInfo,
    screenshot: Screenshot,
) -> None:
    """Verify lookup and capture return pixels without introducing persistence."""

    capturer = RecordingWindowCapturer(screenshot)
    service = WindowCaptureService(locator=StubWindowLocator(window), capturer=capturer)

    result = service.capture(debug=False)

    assert result is screenshot
    assert capturer.captured_window == window


def test_debug_false_does_not_create_an_image(
    tmp_path: Path,
    window: WindowInfo,
    screenshot: Screenshot,
) -> None:
    """Guarantee the default in-memory pipeline has no filesystem side effects."""

    debug_path = tmp_path / "nested" / "capture.png"
    service = WindowCaptureService(
        locator=StubWindowLocator(window),
        capturer=RecordingWindowCapturer(screenshot),
        debug_image_saver=save_debug_image,
        debug_image_path=debug_path,
    )

    result = service.capture(debug=False)

    assert result is screenshot
    assert not debug_path.exists()
    assert not debug_path.parent.exists()


def test_debug_true_saves_the_bgr_screenshot_with_opencv(
    tmp_path: Path,
    window: WindowInfo,
    screenshot: Screenshot,
) -> None:
    """Verify explicit debug mode writes a PNG without changing returned pixels."""

    debug_path = tmp_path / "nested" / "capture.png"
    service = WindowCaptureService(
        locator=StubWindowLocator(window),
        capturer=RecordingWindowCapturer(screenshot),
        debug_image_saver=save_debug_image,
        debug_image_path=debug_path,
    )

    result = service.capture(debug=True)
    decoded_image = cv2.imread(str(debug_path), cv2.IMREAD_COLOR)

    assert result is screenshot
    assert debug_path.is_file()
    assert decoded_image is not None
    assert np.array_equal(decoded_image, screenshot.image)


def test_debug_true_requires_an_explicit_saver(
    window: WindowInfo,
    screenshot: Screenshot,
) -> None:
    """Reject debug persistence when composition omitted its infrastructure port."""

    service = WindowCaptureService(
        locator=StubWindowLocator(window),
        capturer=RecordingWindowCapturer(screenshot),
    )

    with pytest.raises(WindowCaptureError, match="without a configured"):
        service.capture(debug=True)
