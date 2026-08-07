"""Dependency-injected Cats autoplay application state machine."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Protocol

from logicforge.application.cats.click_plan import (
    collect_cat_coordinates,
    execute_cat_click_plan,
)
from logicforge.application.cats.models import (
    CatClickTarget,
    CatsBoardInput,
    CatsSolvedBoard,
)
from logicforge.application.cats.presentation import (
    print_cat_click_plan,
    print_solve_information,
)
from logicforge.application.cats.validation import (
    CatsBoardGeometryMismatchError,
    validate_cats_board_input_geometry,
    validate_complete_cats_solution,
)
from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
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
    WindowCapturer,
    WindowInfo,
    WindowLocator,
)

FAILURE_OVERLAY_PATH = Path("artifacts/vision/cats_autoplay_failure.png")

type SleepFunction = Callable[[float], None]
type MonotonicFunction = Callable[[], float]
type AnalyzeBoardFunction = Callable[[Screenshot], CatsBoardInput]
type SolveBoardFunction = Callable[[WindowInfo, CatsBoardInput], CatsSolvedBoard]

_TRANSIENT_BOARD_ANALYSIS_ERRORS = (
    BoardDetectionError,
    GridDetectionError,
    ColorDetectionError,
    CatsExistingCatDetectionError,
)


class CatPlanExecutor(Protocol):
    """Describe the click-plan executor injected into autoplay."""

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
    """Describe only the debug persistence operation autoplay needs."""

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Persist one final failure overlay."""


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
    board_analysis_retry_seconds: float = 3.0
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
            "board_analysis_retry_seconds": self.board_analysis_retry_seconds,
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
        if self.board_analysis_retry_seconds <= 0.0:
            raise ValueError("board_analysis_retry_seconds must be positive.")
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
        analyze_board: AnalyzeBoardFunction,
        solve_board: SolveBoardFunction,
        sleep_function: SleepFunction = time.sleep,
        monotonic_function: MonotonicFunction = time.monotonic,
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
        self._board_analysis_failure_started_at: float | None = None
        self._last_board_analysis_error: RuntimeError | None = None
        self._last_board_analysis_failure_log_at: float | None = None
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
        previous_state = self._last_screen_state
        state_changed = detection.state is not self._last_screen_state
        if state_changed:
            if (
                previous_state is CatsScreenState.BOARD
                and detection.state is not CatsScreenState.BOARD
            ):
                self._reset_board_analysis_failure()
            if detection.state is not CatsScreenState.BOARD:
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

        try:
            board_input = self._analyze_board(screenshot)
        except _TRANSIENT_BOARD_ANALYSIS_ERRORS as error:
            return self._handle_transient_board_failure(error, now)
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
            return self._handle_transient_board_failure(error, now)
        self._reset_board_analysis_failure()
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

    def _handle_transient_board_failure(
        self,
        error: RuntimeError,
        now: float,
    ) -> tuple[bool, bool]:
        """Retry one newly captured BOARD per poll within one bounded window."""

        if self._board_analysis_failure_started_at is None:
            self._board_analysis_failure_started_at = now
        self._last_board_analysis_error = error
        elapsed = max(0.0, now - self._board_analysis_failure_started_at)
        retry_seconds = self._settings.board_analysis_retry_seconds
        error_name = type(error).__name__
        if elapsed >= retry_seconds:
            print(
                "[board] analysis did not stabilize within "
                f"{retry_seconds:.1f}s; raising {error_name}: {error}"
            )
            raise error
        if (
            self._last_board_analysis_failure_log_at is None
            or now - self._last_board_analysis_failure_log_at >= 1.0
        ):
            print(
                f"[board] transient {error_name}; retrying "
                f"({elapsed:.1f}/{retry_seconds:.1f}s): {error}"
            )
            self._last_board_analysis_failure_log_at = now
        return False, False

    def _reset_board_analysis_failure(self) -> None:
        """Forget one consecutive transient-failure window after stabilization."""

        self._board_analysis_failure_started_at = None
        self._last_board_analysis_error = None
        self._last_board_analysis_failure_log_at = None

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
            exact_search_result=solved.exact_search_result,
            status=solved.status,
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
