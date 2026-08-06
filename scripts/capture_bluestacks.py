"""Capture BlueStacks into memory and optionally save a debugging PNG."""

from pathlib import Path
from time import perf_counter
from typing import Final

from logicforge.infrastructure.opencv_debug_image import save_debug_image
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowCaptureService, WindowInfo

DEBUG: Final = True
DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/bluestacks_capture.png")


def print_capture_information(
    window: WindowInfo,
    screenshot: Screenshot,
    elapsed_seconds: float,
    *,
    debug: bool,
) -> None:
    """Print geometry, in-memory resolution, timing, and optional debug output."""

    bounds = window.bounds
    print(f"Window title: {window.title}")
    print(f"Position: x={bounds.x}, y={bounds.y}")
    print(f"Size: width={bounds.width}, height={bounds.height}")
    print(f"Capture resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"Elapsed capture time: {elapsed_seconds:.4f} seconds")
    if debug:
        print(f"Saved to: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Compose adapters, capture one in-memory screenshot, and print diagnostics."""

    service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
        debug_image_saver=save_debug_image,
        debug_image_path=DEBUG_OUTPUT_PATH,
    )
    window = service.locate_window()
    started_at = perf_counter()
    screenshot = service.capture_window(window, debug=DEBUG)
    elapsed_seconds = perf_counter() - started_at
    print_capture_information(
        window,
        screenshot,
        elapsed_seconds,
        debug=DEBUG,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
