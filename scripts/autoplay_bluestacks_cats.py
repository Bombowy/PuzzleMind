"""Thin CLI composition root for the complete Cats autoplay state machine."""

import argparse
import sys
from collections.abc import Callable, Sequence
from math import isfinite

from logicforge.application.cats import (
    CatClickExecutionError,
    CatClickPlanError,
    CatsAutomationError,
    CatsAutomationPhase,
    CatsAutomationTimeoutError,
    CatsAutoplayRunner,
    CatsAutoplaySettings,
    CatsAutoplaySummary,
    CatsBoardGeometryMismatchError,
    CatsBoardInput,
    CatsSolutionValidationError,
    CatsSolvedBoard,
    CatsSolveStatus,
    validate_cats_board_input_geometry,
    validate_complete_cats_solution,
)
from logicforge.application.cats.analysis import (
    analyze_captured_cats_board as analyze_cats_board_with_ports,
)
from logicforge.application.cats.solving import solve_analyzed_cats_board
from logicforge.config.settings import BoardDetectionSettings, ColorDetectionSettings
from logicforge.core import BoardStateError
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_renderer import (
    CatsScreenStateDebugRenderError,
    OpenCvCatsScreenStateDebugRenderer,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.windows import (
    MouseAutomationError,
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
    Win32MouseController,
)
from logicforge.plugins.cats import CatsExistingCatDetectionError
from logicforge.vision.board_detector import BoardDetectionError
from logicforge.vision.color_detector import ColorDetectionError
from logicforge.vision.grid_detector import GridDetectionError
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import WindowCaptureError

__all__ = [
    "CatsAutomationError",
    "CatsAutomationPhase",
    "CatsAutomationTimeoutError",
    "CatsAutoplayRunner",
    "CatsAutoplaySettings",
    "CatsAutoplaySummary",
    "CatsBoardGeometryMismatchError",
    "CatsSolutionValidationError",
    "CatsSolveStatus",
    "CatsSolvedBoard",
    "analyze_captured_cats_board",
    "main",
    "parse_arguments",
    "print_autoplay_summary",
    "settings_from_arguments",
    "validate_cats_board_input_geometry",
    "validate_complete_cats_solution",
]


def _non_negative_int(value: str) -> int:
    """Parse one non-negative integer CLI setting."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _minimum_int(minimum: int) -> Callable[[str], int]:
    """Create one integer parser with an inclusive minimum."""

    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"value must be at least {minimum}")
        return parsed

    return parse


def _positive_float(value: str) -> float:
    """Parse a finite strictly positive float CLI setting."""

    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return parsed


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit execution, timing, retry, and level-limit settings."""

    parser = argparse.ArgumentParser(
        description="Automatically solve and advance visible Cats levels."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable mouse input and continuous polling.",
    )
    parser.add_argument(
        "--click-delay-ms",
        type=_non_negative_int,
        default=10,
        help="Delay between low-level cat clicks in milliseconds (default: 10).",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=_minimum_int(10),
        default=100,
        help="Polling interval in milliseconds (default: 100).",
    )
    parser.add_argument(
        "--transition-timeout-seconds",
        type=_positive_float,
        default=20.0,
        help="Maximum seconds without meaningful progress (default: 20).",
    )
    parser.add_argument(
        "--board-analysis-retry-seconds",
        type=_positive_float,
        default=3.0,
        help=(
            "Maximum seconds to retry transient BOARD analysis failures "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--overlay-retry-ms",
        type=_minimum_int(100),
        default=750,
        help="Minimum stationary-overlay retry delay in milliseconds (default: 750).",
    )
    parser.add_argument(
        "--max-overlay-retries",
        type=_minimum_int(1),
        default=3,
        help="Maximum retries after an initial overlay click (default: 3).",
    )
    parser.add_argument(
        "--new-board-delay-ms",
        type=_non_negative_int,
        default=300,
        help="Delay before accepting a board after level click (default: 300).",
    )
    parser.add_argument(
        "--max-levels",
        type=_non_negative_int,
        default=0,
        help="Stop after this many solved boards; zero is unlimited (default: 0).",
    )
    return parser.parse_args(arguments)


def settings_from_arguments(arguments: argparse.Namespace) -> CatsAutoplaySettings:
    """Convert CLI milliseconds once into immutable second-based settings."""

    return CatsAutoplaySettings(
        execute=arguments.execute,
        click_delay_seconds=arguments.click_delay_ms / 1000.0,
        poll_interval_seconds=arguments.poll_interval_ms / 1000.0,
        transition_timeout_seconds=arguments.transition_timeout_seconds,
        board_analysis_retry_seconds=arguments.board_analysis_retry_seconds,
        overlay_retry_seconds=arguments.overlay_retry_ms / 1000.0,
        max_overlay_retries=arguments.max_overlay_retries,
        new_board_delay_seconds=arguments.new_board_delay_ms / 1000.0,
        max_levels=arguments.max_levels,
    )


def analyze_captured_cats_board(screenshot: Screenshot) -> CatsBoardInput:
    """Compose the unchanged Cats analysis pipeline from concrete OpenCV adapters."""

    def fallback_geometry_detectors() -> tuple[
        OpenCvBoardDetector,
        OpenCvGridDetector,
    ]:
        board_settings = BoardDetectionSettings()
        return (
            OpenCvBoardDetector(board_settings),
            OpenCvGridDetector(board_settings),
        )

    return analyze_cats_board_with_ports(
        screenshot,
        tile_grid_detector=OpenCvCatsTileGridDetector(),
        fallback_geometry_detectors=fallback_geometry_detectors,
        color_detector=OpenCvColorDetector(ColorDetectionSettings()),
        existing_cat_detector=OpenCvCatsExistingCatDetector(),
    )


def print_autoplay_summary(summary: CatsAutoplaySummary) -> None:
    """Print deterministic primitive session totals."""

    print("Cats autoplay summary")
    print(f"Solved levels: {summary.solved_levels}")
    print(f"Ranking clicks: {summary.ranking_clicks}")
    print(f"Level-button clicks: {summary.level_button_clicks}")
    print(f"Cat targets executed: {summary.low_level_cat_clicks // 2}")
    print(f"Low-level cat clicks: {summary.low_level_cat_clicks}")
    print(f"Final screen state: {summary.final_screen_state.name}")


def _save_failure_without_masking(runner: CatsAutoplayRunner) -> None:
    """Attempt the single requested debug write without replacing the main error."""

    try:
        saved = runner.save_failure_overlay()
    except CatsScreenStateDebugRenderError as error:
        print(
            f"Cats autoplay failure overlay could not be saved: {error}",
            file=sys.stderr,
        )
    else:
        if saved is not None:
            print(f"Cats autoplay failure overlay: {saved.as_posix()}", file=sys.stderr)


def main(arguments: Sequence[str] | None = None) -> int:
    """Compose real adapters and translate only expected operational failures."""

    parsed = parse_arguments(() if arguments is None else arguments)
    settings = settings_from_arguments(parsed)
    runner = CatsAutoplayRunner(
        settings=settings,
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
        detector=OpenCvCatsScreenStateDetector(),
        renderer=OpenCvCatsScreenStateDebugRenderer(),
        mouse_controller=Win32MouseController() if settings.execute else None,
        analyze_board=analyze_captured_cats_board,
        solve_board=solve_analyzed_cats_board,
    )
    try:
        summary = runner.run()
    except KeyboardInterrupt:
        print("Cats autoplay stopped by user.")
        print_autoplay_summary(runner.summary())
        return 130
    except WindowCaptureError as error:
        print(f"BlueStacks capture failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 1
    except CatsScreenStateDetectionError as error:
        print(f"Cats screen-state detection failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 2
    except BoardDetectionError as error:
        _save_failure_without_masking(runner)
        print(f"Board detection failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 3
    except GridDetectionError as error:
        _save_failure_without_masking(runner)
        print(f"Grid detection failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 4
    except ColorDetectionError as error:
        _save_failure_without_masking(runner)
        print(f"Color detection failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 5
    except CatsExistingCatDetectionError as error:
        _save_failure_without_masking(runner)
        print(f"Existing cat detection failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 5
    except BoardStateError as error:
        _save_failure_without_masking(runner)
        print(f"Cats deduction failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 6
    except CatClickPlanError as error:
        _save_failure_without_masking(runner)
        print(f"Cats click-plan mapping failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 7
    except CatsSolutionValidationError as error:
        _save_failure_without_masking(runner)
        print(f"Cats solution validation failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 8
    except (MouseAutomationError, CatClickExecutionError) as error:
        _save_failure_without_masking(runner)
        print(f"Cats click execution failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 9
    except CatsAutomationError as error:
        _save_failure_without_masking(runner)
        print(f"Cats autoplay failed: {error}", file=sys.stderr)
        print_autoplay_summary(runner.summary())
        return 10

    print_autoplay_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
