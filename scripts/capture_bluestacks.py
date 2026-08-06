"""Capture the BlueStacks App Player window into the vision artifact directory."""

from pathlib import Path

from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.vision.window_capture import CaptureResult, WindowCaptureService

OUTPUT_PATH = Path("artifacts/vision/bluestacks_capture.png")


def print_capture_information(result: CaptureResult) -> None:
    """Print window geometry and timing useful during screenshot debugging."""

    bounds = result.window.bounds
    print(f"Window title: {result.window.title}")
    print(f"Position: x={bounds.x}, y={bounds.y}")
    print(f"Size: width={bounds.width}, height={bounds.height}")
    print(
        "Capture resolution: "
        f"{result.resolution_width}x{result.resolution_height} pixels"
    )
    print(f"Elapsed capture time: {result.elapsed_seconds:.4f} seconds")
    print(f"Saved to: {result.output_path}")


def main() -> int:
    """Compose Windows adapters and create one BlueStacks debugging screenshot."""

    service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
    )
    result = service.capture(OUTPUT_PATH)
    print_capture_information(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
