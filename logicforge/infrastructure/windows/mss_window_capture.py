"""MSS adapter that persists pixels from one prevalidated window rectangle."""

from pathlib import Path
from time import perf_counter

from mss import MSS
from mss.exception import ScreenShotError
from mss.tools import to_png

from logicforge.vision.window_capture import (
    CaptureResult,
    WindowCaptureError,
    WindowCapturer,
    WindowInfo,
)


class MssWindowCapturer(WindowCapturer):
    """Capture only explicit window bounds through MSS and persist them as PNG.

    The adapter never reads ``MSS.monitors`` and never calls a full-screen capture
    helper. Its sole capture input is the bounding box supplied by ``WindowInfo``.
    """

    def capture(self, window: WindowInfo, destination: Path) -> CaptureResult:
        """Capture the selected rectangle, create its directory, and save a PNG."""

        output_path = destination.resolve()
        started_at = perf_counter()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with MSS() as screenshot_session:
                screenshot = screenshot_session.grab(window.bounds.as_bbox())
                to_png(screenshot.rgb, screenshot.size, output=str(output_path))
        except (OSError, ScreenShotError) as error:
            raise WindowCaptureError(
                f'Could not capture "{window.title}" to "{output_path}".'
            ) from error

        elapsed_seconds = perf_counter() - started_at
        return CaptureResult(
            window=window,
            output_path=output_path,
            resolution_width=screenshot.width,
            resolution_height=screenshot.height,
            elapsed_seconds=elapsed_seconds,
        )
