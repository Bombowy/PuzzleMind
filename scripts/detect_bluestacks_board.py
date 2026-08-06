"""Capture BlueStacks, locate its puzzle board, and save one debug overlay."""

from pathlib import Path
from time import perf_counter
from typing import Final

from logicforge.infrastructure.opencv_board_detection_renderer import (
    OpenCvBoardDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowCaptureService, WindowInfo

DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/board_detection.png")


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detection: BoardDetection,
    elapsed_seconds: float,
    output_path: Path,
) -> None:
    """Print the operational evidence needed to inspect one manual detection run."""

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"Board position: x={detection.x}, y={detection.y}")
    print(f"Board size: width={detection.width}, height={detection.height}")
    print(f"Board confidence: {detection.confidence:.3f}")
    print(f"Elapsed detection time: {elapsed_seconds:.4f} seconds")
    print(f"Debug output path: {output_path.as_posix()}")


def main() -> int:
    """Compose existing capture adapters with detection and explicit debug output."""

    capture_service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
    )
    window = capture_service.locate_window()
    screenshot = capture_service.capture_window(window, debug=False)

    detector = OpenCvBoardDetector()
    started_at = perf_counter()
    analysis = detector.analyze(screenshot)
    elapsed_seconds = perf_counter() - started_at

    renderer = OpenCvBoardDetectionDebugRenderer()
    saved_path = renderer.save_debug_overlay(
        screenshot,
        analysis,
        DEBUG_OUTPUT_PATH,
        debug=True,
    )
    if saved_path is None:
        raise RuntimeError("Debug rendering was enabled but produced no output path.")

    print_detection_information(
        window,
        screenshot,
        analysis.detection,
        elapsed_seconds,
        DEBUG_OUTPUT_PATH,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
