"""MSS adapter that captures one prevalidated window rectangle into memory."""

from datetime import UTC, datetime

import numpy as np
from mss import MSS
from mss.exception import ScreenShotError

from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCapturer,
    WindowInfo,
)


class MssWindowCapturer(WindowCapturer):
    """Capture explicit window bounds as an in-memory BGR NumPy array.

    MSS exposes BGRA pixels. The alpha channel is discarded without color-channel
    reordering, leaving the BGR layout expected by OpenCV detector adapters.
    This adapter performs no encoding, file creation, or disk I/O.
    """

    def capture(self, window: WindowInfo) -> Screenshot:
        """Capture the selected rectangle and return an immutable BGR screenshot."""

        try:
            with MSS() as screenshot_session:
                captured_frame = screenshot_session.grab(window.bounds.as_bbox())
        except ScreenShotError as error:
            raise WindowCaptureError(f'Could not capture "{window.title}".') from error

        captured_at = datetime.now(UTC)
        bgra_image = np.asarray(captured_frame, dtype=np.uint8)
        bgr_image = bgra_image[:, :, :3]
        return Screenshot(
            image=bgr_image,
            width=captured_frame.width,
            height=captured_frame.height,
            timestamp=captured_at,
        )
