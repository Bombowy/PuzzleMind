"""Run the complete Cats state machine with explicit opt-in mouse execution."""

import argparse
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Protocol

from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.core import BoardStateError
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_renderer import (
    CatsScreenStateDebugRenderError,
    OpenCvCatsScreenStateDebugRenderer,
)
from logicforge.infrastructure.windows import (
    MouseAutomationError,
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
    Win32MouseController,
)
from logicforge.plugins.cats import (
    CatsExistingCatDetectionError,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDetector,
)
from logicforge.vision.board_detector import BoardDetectionError
from logicforge.vision.color_detector import ColorDetectionError
from logicforge.vision.grid_detector import GridDetectionError
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCapturer,
    WindowInfo,
    WindowLocator,
)
from scripts.solve_bluestacks_cats import (
    CatClickExecutionError,
    CatClickPlanError,
    CatClickTarget,
    CatsBoardInput,
    CatsSolvedBoard,
    analyze_captured_cats_board,
    collect_cat_coordinates,
    execute_cat_click_plan,
    print_cat_click_plan,
    print_solve_information,
    solve_analyzed_cats_board,
)

FAILURE_OVERLAY_PATH = Path("artifacts/vision/cats_autoplay_failure.png")

type SleepFunction = Callable[[float], None]
type MonotonicFunction = Callable[[], float]
type AnalyzeBoardFunction = Callable[[Screenshot], CatsBoardInput]
type SolveBoardFunction = Callable[[WindowInfo, CatsBoardInput], CatsSolvedBoard]


class CatPlanExecutor(Protocol):
    """Describe the existing click-plan executor used by autoplay."""

    def __call__(
        self,
        targets: tuple[CatClickTarget, ...],
        mouse_controller: MouseController,
        *,
        click_delay_seconds: float,
        sleep_function: SleepFunction,
    ) -> int:
        """Execute every planned target and return the target count."""


class FailureOverlayRenderer(Protocol):
    """Describe only the existing debug persistence operation autoplay needs."""

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Persist one final failure overlay."""


class CatsSolutionValidationError(RuntimeError):
    """Report an unsafe, partial, or internally inconsistent Cats solution."""


class CatsBoardGeometryMismatchError(CatsSolutionValidationError):
    """Reject Cats vision geometry before constructing or solving a Board."""


class CatsAutomationError(RuntimeError):
    """Report an inconsistent autoplay phase or required action."""


class CatsAutomationTimeoutError(CatsAutomationError):
    """Report lack of meaningful progress or exhausted overlay retries."""


class CatsAutomationPhase(StrEnum):
    """Identify the solver arming and transition-waiting phases."""

    READY_FOR_BOARD = "ready_for_board"
    WAITING_FOR_TRANSITION = "waiting_for_transition"
    WAITING_FOR_LEVEL_COMPLETE = "waiting_for_level_complete"
    WAITING_FOR_NEXT_BOARD = "waiting_for_next_board"


@dataclass(frozen=True, slots=True)
class CatsAutoplaySettings:
    """Configure deterministic click, polling, retry, timeout, and level limits."""

    execute: bool
    click_delay_seconds: float = 0.01
    poll_interval_seconds: float = 0.10
    transition_timeout_seconds: float = 20.0
    overlay_retry_seconds: float = 0.75
    max_overlay_retries: int = 3
    new_board_delay_seconds: float = 0.30
    max_levels: int = 0

    def __post_init__(self) -> None:
        """Reject non-finite or unsafe timings and invalid count limits."""

        finite_values = {
            "click_delay_seconds": self.click_delay_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "transition_timeout_seconds": self.transition_timeout_seconds,
            "overlay_retry_seconds": self.overlay_retry_seconds,
            "new_board_delay_seconds": self.new_board_delay_seconds,
        }
        for name, value in finite_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.click_delay_seconds < 0.0:
            raise ValueError("click_delay_seconds must be non-negative.")
        if self.poll_interval_seconds < 0.01:
            raise ValueError("poll_interval_seconds must be at least 0.01.")
        if self.transition_timeout_seconds <= 0.0:
            raise ValueError("transition_timeout_seconds must be positive.")
        if self.overlay_retry_seconds < 0.10:
            raise ValueError("overlay_retry_seconds must be at least 0.10.")
        if self.new_board_delay_seconds < 0.0:
            raise ValueError("new_board_delay_seconds must be non-negative.")
        if self.max_overlay_retries < 1:
            raise ValueError("max_overlay_retries must be at least one.")
        if self.max_levels < 0:
            raise ValueError("max_levels must be non-negative.")


@dataclass(frozen=True, slots=True)
class CatsAutoplaySummary:
    """Expose primitive final session counts without retaining backend objects."""

    solved_levels: int
    ranking_clicks: int
    level_button_clicks: int
    low_level_cat_clicks: int
    final_screen_state: CatsScreenState


@dataclass(frozen=True, slots=True)
class _OverlayIdentity:
    """Identify one stationary accepted transition overlay."""

    state: CatsScreenState
    action_x: int
    action_y: int
    rectangles: tuple[CatsScreenRect, ...]


class _WindowBoundsChangedError(RuntimeError):
    """Abort one stale click before forwarding it to the real controller."""


class _BoundsCheckingMouseController(MouseController):
    """Re-read BlueStacks bounds immediately before every low-level cat click."""

    def __init__(
        self,
        *,
        delegate: MouseController,
        locator: WindowLocator,
        expected_window: WindowInfo,
    ) -> None:
        self._delegate = delegate
        self._locator = locator
        self._expected_window = expected_window

    def click(
        self,
        point: ScreenPoint,
        button: MouseButton = MouseButton.LEFT,
    ) -> None:
        current_window = self._locator.locate()
        if current_window.bounds != self._expected_window.bounds:
            raise _WindowBoundsChangedError
        self._delegate.click(point, button)


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
        overlay_retry_seconds=arguments.overlay_retry_ms / 1000.0,
        max_overlay_retries=arguments.max_overlay_retries,
        new_board_delay_seconds=arguments.new_board_delay_ms / 1000.0,
        max_levels=arguments.max_levels,
    )


def validate_complete_cats_solution(solved: CatsSolvedBoard) -> None:
    """Require a complete one-cat-per-row, column, and original-color solution."""

    if solved.status != "COMPLETE":
        raise CatsSolutionValidationError(
            f"Cats solution status is {solved.status}, not COMPLETE."
        )

    board = solved.logical_board
    grid = solved.board_input.grid
    color_result = solved.board_input.color_result
    matrix = color_result.color_matrix
    if not (
        len(board.cells) == grid.rows == len(matrix)
        and all(len(row) == grid.columns for row in board.cells)
        and all(len(row) == grid.columns for row in matrix)
    ):
        raise CatsSolutionValidationError(
            "Logical board, color matrix, and grid dimensions are inconsistent."
        )
    if not grid.rows == grid.columns == color_result.color_count:
        raise CatsSolutionValidationError(
            "Rows, columns, and color_count must be equal for a complete Cats solution."
        )

    unsupported = tuple(
        (row, column, value)
        for row, values in enumerate(board.cells)
        for column, value in enumerate(values)
        if value not in {"K", "X"}
    )
    if unsupported:
        row, column, value = unsupported[0]
        if value.startswith("C") and value[1:].isdigit():
            raise CatsSolutionValidationError(
                f"Unresolved Cats cell remains at ({row}, {column}): {value}."
            )
        raise CatsSolutionValidationError(
            f"Unsupported Cats board value at ({row}, {column}): {value!r}."
        )

    cats = collect_cat_coordinates(board)
    for row in range(grid.rows):
        row_count = sum(cat_row == row for cat_row, _ in cats)
        if row_count != 1:
            raise CatsSolutionValidationError(
                f"Row {row} contains {row_count} cats instead of exactly one."
            )
    for column in range(grid.columns):
        column_count = sum(cat_column == column for _, cat_column in cats)
        if column_count != 1:
            raise CatsSolutionValidationError(
                f"Column {column} contains {column_count} cats instead of exactly one."
            )

    original_ids = tuple(color_id for row in matrix for color_id in row)
    expected_color_ids = set(original_ids)
    cat_color_ids = tuple(matrix[row][column] for row, column in cats)
    color_counts = Counter(cat_color_ids)
    for color_id in sorted(expected_color_ids):
        if color_counts[color_id] != 1:
            raise CatsSolutionValidationError(
                f"Original color {color_id} has {color_counts[color_id]} cats "
                "instead of exactly one."
            )
    if len(set(cat_color_ids)) != color_result.color_count:
        raise CatsSolutionValidationError(
            "The number of cat colors does not equal color_count."
        )

    for first, second in combinations(cats, 2):
        if max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1:
            raise CatsSolutionValidationError(
                f"Cats at {first} and {second} touch orthogonally or diagonally."
            )

    planned_coordinates = tuple(
        (target.row, target.column) for target in solved.click_plan
    )
    if len(planned_coordinates) != len(set(planned_coordinates)):
        raise CatsSolutionValidationError("Cat click plan contains duplicate targets.")
    existing_coordinates = tuple(
        (cat.row, cat.column) for cat in solved.board_input.existing_cat_detection.cats
    )
    if len(existing_coordinates) != len(set(existing_coordinates)):
        raise CatsSolutionValidationError(
            "Existing cat evidence contains duplicate coordinates."
        )
    for row, column in existing_coordinates:
        if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
            raise CatsSolutionValidationError(
                f"Existing cat coordinate ({row}, {column}) is outside the grid."
            )
        if (row, column) not in cats:
            raise CatsSolutionValidationError(
                f"Existing cat coordinate ({row}, {column}) is not K on final Board."
            )
    expected_new_cats = tuple(
        coordinate for coordinate in cats if coordinate not in set(existing_coordinates)
    )
    if planned_coordinates != expected_new_cats:
        raise CatsSolutionValidationError(
            "Cat click plan does not exactly match row-major new K coordinates."
        )
    if not len(cats) == grid.rows == grid.columns == color_result.color_count:
        raise CatsSolutionValidationError(
            "Cat count must equal rows, columns, and color_count."
        )


def validate_cats_board_input_geometry(board_input: CatsBoardInput) -> None:
    """Require a square Cats grid and one color per row before logical solving."""

    grid = board_input.grid
    color_result = board_input.color_result
    matrix = color_result.color_matrix
    if grid.rows != grid.columns or grid.rows != color_result.color_count:
        raise CatsBoardGeometryMismatchError(
            "Cats board geometry mismatch: "
            f"grid={grid.rows}x{grid.columns}, "
            f"colors={color_result.color_count}."
        )
    if len(matrix) != grid.rows or any(len(row) != grid.columns for row in matrix):
        matrix_widths = tuple(len(row) for row in matrix)
        raise CatsBoardGeometryMismatchError(
            "Cats color_matrix geometry mismatch: "
            f"grid={grid.rows}x{grid.columns}, "
            f"matrix_rows={len(matrix)}, matrix_widths={matrix_widths}."
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


class CatsAutoplayRunner:
    """Coordinate capture, accepted state actions, solving, and safe retries."""

    def __init__(
        self,
        *,
        settings: CatsAutoplaySettings,
        locator: WindowLocator,
        capturer: WindowCapturer,
        detector: CatsScreenStateDetector,
        renderer: FailureOverlayRenderer,
        mouse_controller: MouseController | None,
        sleep_function: SleepFunction = time.sleep,
        monotonic_function: MonotonicFunction = time.monotonic,
        analyze_board: AnalyzeBoardFunction = analyze_captured_cats_board,
        solve_board: SolveBoardFunction = solve_analyzed_cats_board,
        execute_plan: CatPlanExecutor = execute_cat_click_plan,
    ) -> None:
        """Receive every stateful or external dependency through explicit ports."""

        self._settings = settings
        self._locator = locator
        self._capturer = capturer
        self._detector = detector
        self._renderer = renderer
        self._mouse_controller = mouse_controller
        self._sleep = sleep_function
        self._monotonic = monotonic_function
        self._analyze_board = analyze_board
        self._solve_board = solve_board
        self._execute_plan = execute_plan

        self._phase = CatsAutomationPhase.READY_FOR_BOARD
        self._solved_levels = 0
        self._ranking_clicks = 0
        self._level_button_clicks = 0
        self._low_level_cat_clicks = 0
        self._final_state = CatsScreenState.UNKNOWN
        self._last_screen_state: CatsScreenState | None = None
        self._last_progress_at = 0.0
        self._last_unknown_print_at: float | None = None
        self._last_solved_color_matrix: tuple[tuple[str, ...], ...] | None = None
        self._last_level_click_at: float | None = None
        self._overlay_identity: _OverlayIdentity | None = None
        self._overlay_last_click_at: float | None = None
        self._overlay_retries = 0
        self._latest_screenshot: Screenshot | None = None
        self._latest_detection: CatsScreenStateDetection | None = None
        self._failure_overlay_saved = False

    def run(self) -> CatsAutoplaySummary:
        """Run one dry poll or the explicit continuous autoplay loop."""

        self._last_progress_at = self._monotonic()
        if not self._settings.execute:
            self._run_dry_poll()
            return self.summary()

        while True:
            should_stop = self._run_execute_poll()
            if should_stop:
                return self.summary()

    def summary(self) -> CatsAutoplaySummary:
        """Return current primitive counts, including interrupted sessions."""

        return CatsAutoplaySummary(
            solved_levels=self._solved_levels,
            ranking_clicks=self._ranking_clicks,
            level_button_clicks=self._level_button_clicks,
            low_level_cat_clicks=self._low_level_cat_clicks,
            final_screen_state=self._final_state,
        )

    def save_failure_overlay(self) -> Path | None:
        """Persist at most one overlay for the latest successfully classified poll."""

        if self._failure_overlay_saved:
            return None
        if self._latest_screenshot is None or self._latest_detection is None:
            return None
        self._failure_overlay_saved = True
        return self._renderer.save_debug_overlay(
            self._latest_screenshot,
            self._latest_detection,
            FAILURE_OVERLAY_PATH,
            debug=True,
        )

    def _capture_and_classify(
        self,
    ) -> tuple[WindowInfo, Screenshot, CatsScreenStateDetection]:
        """Perform exactly one locate, one capture, and one classification per poll."""

        window = self._locator.locate()
        screenshot = self._capturer.capture(window)
        detection = self._detector.detect(screenshot)
        self._latest_screenshot = screenshot
        self._latest_detection = detection
        self._final_state = detection.state
        return window, screenshot, detection

    def _run_dry_poll(self) -> None:
        """Classify and plan exactly one capture without constructing mouse input."""

        window, screenshot, detection = self._capture_and_classify()
        self._print_screen(detection)
        if detection.state is CatsScreenState.BOARD:
            board_input = self._analyze_board(screenshot)
            validate_cats_board_input_geometry(board_input)
            solved = self._solve_board(window, board_input)
            self._print_board_solution(window, screenshot, solved)
            validate_complete_cats_solution(solved)
            print("[dry-run] complete Cats click plan validated; no clicks executed")
        elif detection.state in (
            CatsScreenState.RANKING,
            CatsScreenState.LEVEL_COMPLETE,
        ):
            action = detection.action_point
            if action is None:
                raise CatsAutomationError(
                    f"{detection.state.name} has no accepted action_point."
                )
            print(
                f"[dry-run] {detection.state.name} single click "
                f"screenshot=({action.x}, {action.y}) "
                f"desktop=({window.bounds.x + action.x}, "
                f"{window.bounds.y + action.y})"
            )
        else:
            print("Rejection reasons:")
            for reason in detection.diagnostics.rejection_reasons:
                print(f"- {reason}")

    def _run_execute_poll(self) -> bool:
        """Handle one classified frame and optionally stop at max-levels."""

        window, screenshot, detection = self._capture_and_classify()
        now = self._monotonic()
        state_changed = detection.state is not self._last_screen_state
        if state_changed:
            self._last_progress_at = now
            self._overlay_identity = None
            self._overlay_last_click_at = None
            self._overlay_retries = 0
            self._print_screen(detection)
        elif detection.state is CatsScreenState.UNKNOWN and (
            self._last_unknown_print_at is None
            or now - self._last_unknown_print_at >= 1.0
        ):
            self._print_screen(detection)
        self._last_screen_state = detection.state

        action_performed = False
        should_stop = False
        if detection.state is CatsScreenState.BOARD:
            action_performed, should_stop = self._handle_board(
                window,
                screenshot,
                detection,
                now,
            )
        elif detection.state is CatsScreenState.RANKING:
            action_performed = self._handle_overlay(window, detection, now)
        elif detection.state is CatsScreenState.LEVEL_COMPLETE:
            action_performed = self._handle_overlay(window, detection, now)
        elif detection.state is CatsScreenState.UNKNOWN:
            pass
        else:
            raise CatsAutomationError(
                f"Unsupported Cats screen state: {detection.state!r}."
            )

        if should_stop:
            return True
        if not action_performed:
            current_time = self._monotonic()
            if current_time - self._last_progress_at > (
                self._settings.transition_timeout_seconds
            ):
                raise CatsAutomationTimeoutError(
                    f"No meaningful Cats autoplay progress for "
                    f"{self._settings.transition_timeout_seconds:.3f} seconds."
                )
            self._sleep(self._settings.poll_interval_seconds)
        return False

    def _handle_board(
        self,
        window: WindowInfo,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        now: float,
    ) -> tuple[bool, bool]:
        """Solve an armed new board once, or wait through transition frames."""

        del detection
        if self._phase is CatsAutomationPhase.WAITING_FOR_TRANSITION:
            return False, False
        if self._phase is CatsAutomationPhase.WAITING_FOR_LEVEL_COMPLETE:
            return False, False
        if self._phase is CatsAutomationPhase.WAITING_FOR_NEXT_BOARD:
            if self._last_level_click_at is None:
                raise CatsAutomationError(
                    "WAITING_FOR_NEXT_BOARD has no level-button click timestamp."
                )
            if now - self._last_level_click_at < (
                self._settings.new_board_delay_seconds
            ):
                return False, False

        board_input = self._analyze_board(screenshot)
        print(
            f"[board] detected {board_input.grid.rows}x{board_input.grid.columns}, "
            f"colors={board_input.color_result.color_count}, "
            f"existing_cats={len(board_input.existing_cat_detection.cats)}"
        )
        for cat in board_input.existing_cat_detection.cats:
            color_id = board_input.color_result.color_matrix[cat.row][cat.column]
            print(
                f"[board] existing cat row={cat.row},column={cat.column},"
                f"color={color_id},confidence={cat.confidence:.3f}"
            )
        try:
            validate_cats_board_input_geometry(board_input)
        except CatsBoardGeometryMismatchError as error:
            print(
                "[board] rejected transient geometry: "
                f"grid={board_input.grid.rows}x{board_input.grid.columns}, "
                f"colors={board_input.color_result.color_count}; "
                f"{error}; waiting for retry"
            )
            return False, False
        matrix = board_input.color_result.color_matrix
        if matrix == self._last_solved_color_matrix:
            print("[board] old completed color_matrix still visible; waiting")
            return False, False

        solved = self._solve_board(window, board_input)
        self._print_board_solution(window, screenshot, solved)
        validate_complete_cats_solution(solved)
        print("[board] new level accepted")
        self._last_progress_at = now

        current_window = self._locator.locate()
        if current_window.bounds != window.bounds:
            print("[window] BlueStacks was moved; discarding stale board coordinates")
            return False, False
        checked_mouse_controller = _BoundsCheckingMouseController(
            delegate=self._require_mouse_controller(),
            locator=self._locator,
            expected_window=window,
        )
        try:
            executed_targets = self._execute_plan(
                solved.click_plan,
                checked_mouse_controller,
                click_delay_seconds=self._settings.click_delay_seconds,
                sleep_function=self._sleep,
            )
        except _WindowBoundsChangedError:
            print("[window] BlueStacks was moved; stopping stale cat click plan")
            return False, False
        self._solved_levels += 1
        self._low_level_cat_clicks += executed_targets * 2
        self._last_solved_color_matrix = matrix
        self._phase = CatsAutomationPhase.WAITING_FOR_TRANSITION
        self._last_progress_at = self._monotonic()
        print(
            f"[click] cat targets={executed_targets}, "
            f"low-level clicks={executed_targets * 2}, "
            f"delay={self._settings.click_delay_seconds * 1000:g} ms"
        )
        reached_limit = (
            self._settings.max_levels > 0
            and self._solved_levels >= self._settings.max_levels
        )
        return True, reached_limit

    def _handle_overlay(
        self,
        window: WindowInfo,
        detection: CatsScreenStateDetection,
        now: float,
    ) -> bool:
        """Single-click an accepted overlay with stationary-screen retry limits."""

        action = detection.action_point
        if action is None:
            raise CatsAutomationError(
                f"{detection.state.name} has no accepted action_point."
            )
        identity = self._overlay_identity_for(detection)
        is_new_overlay = identity != self._overlay_identity
        if is_new_overlay:
            self._overlay_identity = identity
            self._overlay_last_click_at = None
            self._overlay_retries = 0
        elif self._overlay_last_click_at is not None:
            elapsed = now - self._overlay_last_click_at
            if elapsed < self._settings.overlay_retry_seconds:
                return False
            if self._overlay_retries >= self._settings.max_overlay_retries:
                raise CatsAutomationTimeoutError(
                    f"{detection.state.name} remained unchanged after "
                    f"{self._settings.max_overlay_retries} overlay retries."
                )

        current_window = self._locator.locate()
        if current_window.bounds != window.bounds:
            print("[window] BlueStacks was moved; recapturing before overlay click")
            return False

        point = ScreenPoint(
            x=window.bounds.x + action.x,
            y=window.bounds.y + action.y,
        )
        self._require_mouse_controller().click(point, MouseButton.LEFT)
        if not is_new_overlay and self._overlay_last_click_at is not None:
            self._overlay_retries += 1
        self._overlay_last_click_at = now
        self._last_progress_at = now
        if detection.state is CatsScreenState.RANKING:
            self._ranking_clicks += 1
            self._phase = CatsAutomationPhase.WAITING_FOR_LEVEL_COMPLETE
            print(
                f"[click] ranking screenshot=({action.x}, {action.y}) "
                f"desktop=({point.x}, {point.y})"
            )
        else:
            self._level_button_clicks += 1
            self._phase = CatsAutomationPhase.WAITING_FOR_NEXT_BOARD
            self._last_level_click_at = now
            print(
                f"[click] level screenshot=({action.x}, {action.y}) "
                f"desktop=({point.x}, {point.y})"
            )
            print("[screen] waiting for new BOARD")
        return True

    def _print_screen(self, detection: CatsScreenStateDetection) -> None:
        """Print one concise state transition or throttled UNKNOWN diagnostic."""

        diagnostics = detection.diagnostics
        viewport = diagnostics.game_viewport_candidate
        message = (
            f"[screen] {detection.state.name} "
            f"confidence={detection.confidence:.3f} "
            f"viewport={diagnostics.game_viewport_score:.3f}"
        )
        if viewport is None:
            message += " rect=none"
        else:
            message += (
                f" rect=({viewport.x}, {viewport.y}, "
                f"{viewport.width}, {viewport.height})"
            )
        if detection.state is CatsScreenState.RANKING:
            message += f" cards={len(diagnostics.ranking_card_candidates)}"
        print(message)
        if detection.state is CatsScreenState.UNKNOWN:
            self._last_unknown_print_at = self._monotonic()

    @staticmethod
    def _overlay_identity_for(
        detection: CatsScreenStateDetection,
    ) -> _OverlayIdentity:
        """Combine state, action, and accepted geometry for retry identity."""

        action = detection.action_point
        if action is None:
            raise CatsAutomationError(
                f"{detection.state.name} has no accepted action_point."
            )
        if detection.state is CatsScreenState.RANKING:
            rectangles = detection.diagnostics.ranking_card_candidates
        else:
            candidate = detection.diagnostics.level_button_candidate
            rectangles = (candidate,) if candidate is not None else ()
        return _OverlayIdentity(
            state=detection.state,
            action_x=action.x,
            action_y=action.y,
            rectangles=rectangles,
        )

    @staticmethod
    def _print_board_solution(
        window: WindowInfo,
        screenshot: Screenshot,
        solved: CatsSolvedBoard,
    ) -> None:
        """Print the full initial/final board once for each solve attempt."""

        print_solve_information(
            window,
            screenshot,
            solved.board_input.detected_board,
            solved.board_input.grid,
            solved.board_input.color_result,
            solved.logical_board,
            solved.successful_applications,
        )
        print_cat_click_plan(solved.click_plan)
        print(
            f"[board] {solved.status}, "
            f"cats={len(collect_cat_coordinates(solved.logical_board))}, "
            f"new_targets={len(solved.click_plan)}, "
            f"rules={solved.successful_applications}"
        )

    def _require_mouse_controller(self) -> MouseController:
        """Fail closed if execution reaches a click without an injected controller."""

        if self._mouse_controller is None:
            raise CatsAutomationError(
                "Mouse controller is unavailable during execute mode."
            )
        return self._mouse_controller


def _save_failure_without_masking(
    runner: CatsAutoplayRunner,
) -> None:
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
