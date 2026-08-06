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
from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionDiagnostics,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowCaptureService, WindowInfo

DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/board_detection.png")
_REFINEMENT_DIRECTION_ORDER: Final = {
    "left": 0,
    "right": 1,
    "top": 2,
    "bottom": 3,
}


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    detection: BoardDetection,
    elapsed_seconds: float,
    output_path: Path,
    diagnostics: BoardDetectionDiagnostics | None = None,
) -> None:
    """Print the operational evidence needed to inspect one manual detection run."""

    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print(f"Board position: x={detection.x}, y={detection.y}")
    print(f"Board size: width={detection.width}, height={detection.height}")
    print(f"Board confidence: {detection.confidence:.3f}")
    print_envelope_refinement_information(diagnostics)
    print(f"Elapsed detection time: {elapsed_seconds:.4f} seconds")
    print(f"Debug output path: {output_path.as_posix()}")


def print_envelope_refinement_information(
    diagnostics: BoardDetectionDiagnostics | None,
) -> None:
    """Print selected maximal-envelope evidence without backend-specific objects."""

    refinement = diagnostics.selected_refinement if diagnostics is not None else None
    if refinement is None:
        print("Board envelope refined: no")
    else:
        print("Board envelope refined: yes")
        print(
            "Seed board: "
            f"x={refinement.seed_x}, y={refinement.seed_y}, "
            f"width={refinement.seed_width}, height={refinement.seed_height}"
        )
        print(f"Refinement direction: {refinement.direction}")
        print(f"Added pixels: {refinement.added_pixels}")
        print(f"Seed grid: {refinement.seed_rows}x{refinement.seed_columns}")
        print(f"Refined grid: {refinement.refined_rows}x{refinement.refined_columns}")
        print(
            "Separator continuation score: "
            f"{refinement.separator_continuation_score:.3f}"
        )
        print(
            "Supported separator fraction: "
            f"{refinement.supported_separator_fraction:.3f}"
        )
        print(f"Refinement score: {refinement.refinement_score:.3f}")

    attempts = diagnostics.envelope_refinements if diagnostics is not None else ()
    ordered_attempts = sorted(
        attempts,
        key=lambda attempt: (
            attempt.seed_y,
            attempt.seed_x,
            attempt.seed_width,
            attempt.seed_height,
            _REFINEMENT_DIRECTION_ORDER[attempt.direction],
        ),
    )
    print(f"Envelope refinement attempts: {len(ordered_attempts)}")
    for attempt in ordered_attempts:
        _print_refinement_attempt(attempt)


def _print_refinement_attempt(
    attempt: BoardEnvelopeRefinementDiagnostic,
) -> None:
    """Print every primitive metric and rejection reason for one direction."""

    print(f"Refinement attempt: {attempt.direction}")
    print(
        "Seed rectangle: "
        f"x={attempt.seed_x}, y={attempt.seed_y}, "
        f"width={attempt.seed_width}, height={attempt.seed_height}"
    )
    print(
        "Candidate rectangle: "
        f"x={attempt.refined_x}, y={attempt.refined_y}, "
        f"width={attempt.refined_width}, height={attempt.refined_height}"
    )
    print(f"Added pixels: {attempt.added_pixels}")
    print(f"Seed grid: {attempt.seed_rows}x{attempt.seed_columns}")
    print(f"Candidate grid: {attempt.refined_rows}x{attempt.refined_columns}")
    print(f"Old-border match score: {attempt.old_border_match_score:.3f}")
    print(
        "Separator continuation score: " f"{attempt.separator_continuation_score:.3f}"
    )
    print(
        "Supported separator fraction: " f"{attempt.supported_separator_fraction:.3f}"
    )
    print(f"Spacing score: {attempt.spacing_score:.3f}")
    print(f"Refined grid score: {attempt.refined_grid_score:.3f}")
    print(f"Refinement score: {attempt.refinement_score:.3f}")
    print(f"Accepted: {'yes' if attempt.accepted else 'no'}")
    if attempt.rejection_reasons:
        print("Rejection reasons:")
        for reason in attempt.rejection_reasons:
            print(f"- {reason}")
    else:
        print("Rejection reasons: none")


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
        draw_rejected_candidates=True,
    )
    if saved_path is None:
        raise RuntimeError("Debug rendering was enabled but produced no output path.")

    print_detection_information(
        window,
        screenshot,
        analysis.detection,
        elapsed_seconds,
        DEBUG_OUTPUT_PATH,
        analysis.diagnostics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
