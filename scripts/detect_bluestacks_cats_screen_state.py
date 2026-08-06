"""Capture BlueStacks once and classify the visible Cats screen without input."""

import sys
from pathlib import Path

from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_renderer import (
    CatsScreenStateDebugRenderError,
    OpenCvCatsScreenStateDebugRenderer,
)
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.plugins.cats import CatsScreenState, CatsScreenStateDetection
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG_OUTPUT_PATH = Path("artifacts/vision/cats_screen_state.png")


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detection: CatsScreenStateDetection,
) -> None:
    """Print state-specific primitive evidence and screenshot/desktop actions."""

    diagnostics = detection.diagnostics
    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"State: {detection.state.name}")
    print(f"Confidence: {detection.confidence:.3f}")
    viewport = diagnostics.game_viewport_candidate
    if viewport is None:
        print("Game viewport: none")
    else:
        print(
            "Game viewport: "
            f"x={viewport.x}, y={viewport.y}, width={viewport.width}, "
            f"height={viewport.height}"
        )
    print(f"Game viewport score: {diagnostics.game_viewport_score:.3f}")
    print(f"Level-button score: {diagnostics.level_button_score:.3f}")
    print(f"Ranking score: {diagnostics.ranking_score:.3f}")

    if detection.state is CatsScreenState.BOARD:
        board = diagnostics.board_candidate
        if board is not None:
            print(
                "Board: "
                f"x={board.x}, y={board.y}, width={board.width}, "
                f"height={board.height}"
            )
        if diagnostics.detected_rows is not None and (
            diagnostics.detected_columns is not None
        ):
            print(
                f"Grid: {diagnostics.detected_rows} rows x "
                f"{diagnostics.detected_columns} columns"
            )
    elif detection.state is CatsScreenState.RANKING:
        print(f"Ranking cards detected: {len(diagnostics.ranking_card_candidates)}")
    elif detection.state is CatsScreenState.LEVEL_COMPLETE:
        button = diagnostics.level_button_candidate
        if button is not None:
            print(
                "Level button: "
                f"x={button.x}, y={button.y}, width={button.width}, "
                f"height={button.height}"
            )

    if detection.action_point is None:
        print("Action screenshot: none")
    else:
        action = detection.action_point
        print(f"Action screenshot: ({action.x}, {action.y})")
        print(
            "Action desktop: "
            f"({window.bounds.x + action.x}, {window.bounds.y + action.y})"
        )
    if detection.state is CatsScreenState.UNKNOWN:
        print("Rejection reasons:")
        for reason in diagnostics.rejection_reasons:
            print(f"- {reason}")
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Run one capture, one classification, and one explicit debug render."""

    capture_service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
    )
    try:
        window = capture_service.locate_window()
        screenshot = capture_service.capture_window(window, debug=False)
    except WindowCaptureError as error:
        print(f"BlueStacks capture failed: {error}", file=sys.stderr)
        return 1

    detector = OpenCvCatsScreenStateDetector()
    try:
        detection = detector.detect(screenshot)
    except CatsScreenStateDetectionError as error:
        print(f"Cats screen-state detection failed: {error}", file=sys.stderr)
        return 2

    renderer = OpenCvCatsScreenStateDebugRenderer()
    try:
        renderer.save_debug_overlay(
            screenshot,
            detection,
            DEBUG_OUTPUT_PATH,
            debug=True,
        )
    except CatsScreenStateDebugRenderError as error:
        print(f"Cats screen-state debug rendering failed: {error}", file=sys.stderr)
        return 3

    print_detection_information(window, screenshot, detection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
