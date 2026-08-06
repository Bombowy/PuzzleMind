"""Optional OpenCV persistence helper for in-memory debugging screenshots."""

from pathlib import Path

import cv2

from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowCaptureError


def save_debug_image(screenshot: Screenshot, destination: Path) -> Path:
    """Persist a BGR screenshot as PNG solely when explicitly requested by a caller.

    The helper is intentionally separate from MSS capture so normal pipeline usage
    never creates directories, encodes images, or writes to disk.
    """

    output_path = destination.resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = cv2.imwrite(str(output_path), screenshot.image)
    except (OSError, cv2.error) as error:
        raise WindowCaptureError(
            f'Could not save the debug screenshot to "{output_path}".'
        ) from error

    if not written:
        raise WindowCaptureError(
            f'OpenCV could not encode the debug screenshot at "{output_path}".'
        )
    return output_path
