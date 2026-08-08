"""Shared fixtures for Cats autoplay application and CLI tests."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from logicforge.application import cats as cats_app
from logicforge.application.cats import autoplay
from logicforge.application.cats.models import CatsSolveStatus
from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.core import Board
from logicforge.plugins.cats import (
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
    CatsExistingCatDiagnostics,
    CatsExistingCatObservation,
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDetector,
    CatsScreenStateDiagnostics,
)
from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionDiagnostics,
    BoardDetectionError,
)
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionError,
    ColorDetectionResult,
    ColorObservation,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionDiagnostics,
    GridDetectionError,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowCapturer,
    WindowInfo,
    WindowLocator,
)

CAT_COLUMNS = (1, 3, 0, 2)


SIX_CAT_COLUMNS = (2, 0, 4, 1, 5, 3)


NINE_CAT_COLUMNS = (0, 2, 4, 1, 3, 6, 8, 5, 7)


COLUMN_COLOR_MATRIX = tuple(
    tuple(f"C{column}" for column in range(4)) for _ in range(4)
)


ROW_COLOR_MATRIX = tuple(tuple(f"C{row}" for _ in range(4)) for row in range(4))


type BoardAnalysisOutcome = cats_app.CatsBoardInput | RuntimeError


def _screenshot(index: int = 0) -> Screenshot:
    """Create one distinguishable immutable frame without desktop access."""

    return Screenshot(
        image=np.full((100, 80, 3), index, dtype=np.uint8),
        width=80,
        height=100,
        timestamp=datetime(2026, 8, 6, 0, 0, index, tzinfo=UTC),
    )


def _window(*, x: int = -1500, y: int = 1) -> WindowInfo:
    """Return stable virtual-desktop geometry, including negative x."""

    return WindowInfo(
        title="BlueStacks App Player",
        bounds=WindowBounds(x=x, y=y, width=916, height=1032),
    )


def _grid() -> GridDetection:
    """Build a complete 4x4 grid with deterministic screenshot centers."""

    horizontal = (10, 30, 50, 70, 90)
    vertical = (10, 25, 40, 55, 70)
    cells = tuple(
        CellBounds(
            row=row,
            column=column,
            x=vertical[column],
            y=horizontal[row],
            width=vertical[column + 1] - vertical[column],
            height=horizontal[row + 1] - horizontal[row],
            center_x=(vertical[column] + vertical[column + 1]) // 2,
            center_y=(horizontal[row] + horizontal[row + 1]) // 2,
        )
        for row in range(4)
        for column in range(4)
    )
    return GridDetection(
        horizontal_lines=horizontal,
        vertical_lines=vertical,
        rows=4,
        columns=4,
        cells=cells,
        confidence=0.9,
    )


def _color_result(
    matrix: tuple[tuple[str, ...], ...] = COLUMN_COLOR_MATRIX,
) -> ColorDetectionResult:
    """Build one valid immutable four-color vision result."""

    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=matrix[row][column],
            confidence=0.9,
            representative_lab=(120.0, 130.0, 140.0),
        )
        for row in range(4)
        for column in range(4)
    )
    return ColorDetectionResult(
        observations=observations,
        color_count=4,
        color_matrix=matrix,
        mean_confidence=0.9,
        diagnostics=ColorDetectionDiagnostics(
            rows=4,
            columns=4,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * 16,
            within_cell_spreads=(1.0,) * 16,
            cluster_centers_lab=tuple(
                (120.0 + index, 130.0, 140.0) for index in range(4)
            ),
            minimum_intercluster_distance=30.0,
        ),
    )


def _board_input(
    matrix: tuple[tuple[str, ...], ...] = COLUMN_COLOR_MATRIX,
    *,
    existing_coordinates: tuple[tuple[int, int], ...] = (),
) -> cats_app.CatsBoardInput:
    """Return complete immutable vision input for one board."""

    return cats_app.CatsBoardInput(
        detected_board=BoardDetection(10, 10, 60, 80, 0.9),
        grid=_grid(),
        color_result=_color_result(matrix),
        existing_cat_detection=CatsExistingCatDetection(
            cats=tuple(
                CatsExistingCatObservation(row, column, 0.9)
                for row, column in existing_coordinates
            ),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )


def _geometry_board_input(
    rows: int,
    columns: int,
    color_count: int,
) -> cats_app.CatsBoardInput:
    """Build arbitrary Cats input geometry without constructing a logical Board."""

    horizontal = tuple(5 + round(90 * index / rows) for index in range(rows + 1))
    vertical = tuple(5 + round(70 * index / columns) for index in range(columns + 1))
    cells = tuple(
        CellBounds(
            row=row,
            column=column,
            x=vertical[column],
            y=horizontal[row],
            width=vertical[column + 1] - vertical[column],
            height=horizontal[row + 1] - horizontal[row],
            center_x=(vertical[column] + vertical[column + 1]) // 2,
            center_y=(horizontal[row] + horizontal[row + 1]) // 2,
        )
        for row in range(rows)
        for column in range(columns)
    )
    grid = GridDetection(
        horizontal_lines=horizontal,
        vertical_lines=vertical,
        rows=rows,
        columns=columns,
        cells=cells,
        confidence=0.9,
    )
    matrix = tuple(
        tuple(f"C{(row * columns + column) % color_count}" for column in range(columns))
        for row in range(rows)
    )
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id=matrix[row][column],
            confidence=0.9,
            representative_lab=(120.0, 130.0, 140.0),
        )
        for row in range(rows)
        for column in range(columns)
    )
    color_result = ColorDetectionResult(
        observations=observations,
        color_count=color_count,
        color_matrix=matrix,
        mean_confidence=0.9,
        diagnostics=ColorDetectionDiagnostics(
            rows=rows,
            columns=columns,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * (rows * columns),
            within_cell_spreads=(1.0,) * (rows * columns),
            cluster_centers_lab=tuple(
                (120.0 + index, 130.0, 140.0) for index in range(color_count)
            ),
            minimum_intercluster_distance=30.0,
        ),
    )
    return cats_app.CatsBoardInput(
        detected_board=BoardDetection(5, 5, 70, 90, 0.9),
        grid=grid,
        color_result=color_result,
    )


def _board_detection_error() -> BoardDetectionError:
    return BoardDetectionError(
        "synthetic animated board",
        BoardDetectionDiagnostics(0, (), None, 0),
    )


def _grid_detection_error() -> GridDetectionError:
    return GridDetectionError(
        "synthetic animated grid",
        GridDetectionDiagnostics(
            0,
            0,
            1,
            1,
            (),
            (),
            (),
            (),
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            (),
        ),
    )


def _color_detection_error() -> ColorDetectionError:
    return ColorDetectionError(
        "synthetic animated colors",
        ColorDetectionDiagnostics(
            1,
            1,
            18.0,
            (),
            (),
            (),
            None,
            ("synthetic animation",),
        ),
    )


def _existing_cat_detection_error() -> CatsExistingCatDetectionError:
    return CatsExistingCatDetectionError(
        "synthetic animated cat",
        CatsExistingCatDiagnostics(
            cells=(),
            rejection_reasons=("synthetic animation",),
        ),
    )


def _solved_board(
    board_input: cats_app.CatsBoardInput | None = None,
    *,
    window: WindowInfo | None = None,
    status: CatsSolveStatus = CatsSolveStatus.COMPLETE,
    cat_columns: tuple[int, ...] | None = None,
) -> cats_app.CatsSolvedBoard:
    """Create one actual mutable Board and immutable solved wrapper."""

    actual_input = board_input or _board_input()
    actual_window = window or _window()
    logical_board = Board(actual_input.color_result)
    if status is CatsSolveStatus.COMPLETE:
        selected_cat_columns = cat_columns
        if selected_cat_columns is None:
            if actual_input.grid.rows == 9:
                selected_cat_columns = NINE_CAT_COLUMNS
            elif actual_input.grid.rows == 6:
                selected_cat_columns = SIX_CAT_COLUMNS
            else:
                selected_cat_columns = CAT_COLUMNS
        for row in range(actual_input.grid.rows):
            for column in range(actual_input.grid.columns):
                if column == selected_cat_columns[row]:
                    logical_board.set_cat(row, column)
                else:
                    logical_board.set_blocked(row, column)
    existing_coordinates = tuple(
        (cat.row, cat.column) for cat in actual_input.existing_cat_detection.cats
    )
    click_plan = cats_app.build_cat_click_plan(
        logical_board,
        actual_input.grid,
        actual_window,
        existing_cat_coordinates=existing_coordinates,
    )
    return cats_app.CatsSolvedBoard(
        board_input=actual_input,
        logical_board=logical_board,
        successful_applications=7,
        click_plan=click_plan,
        status=status,
    )


def _diagnostics(
    state: CatsScreenState,
) -> CatsScreenStateDiagnostics:
    """Create valid public evidence for one fake state."""

    button = CatsScreenRect(20, 80, 40, 10)
    cards = (CatsScreenRect(20, 25, 40, 8), CatsScreenRect(20, 38, 40, 8))
    return CatsScreenStateDiagnostics(
        game_viewport_candidate=CatsScreenRect(5, 3, 60, 97),
        game_viewport_score=0.8,
        level_button_candidate=(
            button if state is CatsScreenState.LEVEL_COMPLETE else None
        ),
        level_button_score=(0.8 if state is CatsScreenState.LEVEL_COMPLETE else 0.0),
        ranking_card_candidates=(cards if state is CatsScreenState.RANKING else ()),
        ranking_score=(0.9 if state is CatsScreenState.RANKING else 0.0),
        board_candidate=(
            CatsScreenRect(10, 10, 60, 80) if state is CatsScreenState.BOARD else None
        ),
        board_confidence=(0.9 if state is CatsScreenState.BOARD else None),
        grid_confidence=(0.9 if state is CatsScreenState.BOARD else None),
        detected_rows=(4 if state is CatsScreenState.BOARD else None),
        detected_columns=(4 if state is CatsScreenState.BOARD else None),
        rejection_reasons=(
            ("synthetic unknown state",) if state is CatsScreenState.UNKNOWN else ()
        ),
    )


def _detection(
    state: CatsScreenState,
    *,
    action: CatsScreenPoint | None = None,
) -> CatsScreenStateDetection:
    """Return one accepted public state detection."""

    if action is None and state in (
        CatsScreenState.RANKING,
        CatsScreenState.LEVEL_COMPLETE,
    ):
        action = CatsScreenPoint(30, 70 if state is CatsScreenState.RANKING else 85)
    return CatsScreenStateDetection(
        state=state,
        confidence=0.0 if state is CatsScreenState.UNKNOWN else 0.9,
        action_point=action,
        diagnostics=_diagnostics(state),
    )


class FakeWindowLocator(WindowLocator):
    """Return configured windows and record every locate call."""

    def __init__(self, windows: tuple[WindowInfo, ...] = (_window(),)) -> None:
        self.windows = windows
        self.calls = 0

    def locate(self) -> WindowInfo:
        index = min(self.calls, len(self.windows) - 1)
        self.calls += 1
        return self.windows[index]


class FakeWindowCapturer(WindowCapturer):
    """Return one synthetic screenshot per poll without disk access."""

    def __init__(self) -> None:
        self.calls: list[WindowInfo] = []

    def capture(self, window: WindowInfo) -> Screenshot:
        self.calls.append(window)
        return _screenshot(len(self.calls) - 1)


class FakeCatsScreenStateDetector(CatsScreenStateDetector):
    """Return a deterministic sequence, repeating its final state if needed."""

    def __init__(self, detections: tuple[CatsScreenStateDetection, ...]) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, screenshot: Screenshot) -> CatsScreenStateDetection:
        del screenshot
        index = min(self.calls, len(self.detections) - 1)
        self.calls += 1
        return self.detections[index]


class FakeMouseController(MouseController):
    """Record safe portable clicks and optionally raise a typed failure."""

    def __init__(self, error: RuntimeError | None = None) -> None:
        self.error = error
        self.clicks: list[tuple[ScreenPoint, MouseButton]] = []

    def click(
        self,
        point: ScreenPoint,
        button: MouseButton = MouseButton.LEFT,
    ) -> None:
        self.clicks.append((point, button))
        if self.error is not None:
            raise self.error


class FakeClock:
    """Act as both monotonic clock and non-blocking sleep recorder."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeFailureRenderer:
    """Record explicit final failure saves without creating files."""

    def __init__(self) -> None:
        self.calls: list[tuple[Screenshot, CatsScreenStateDetection, Path]] = []

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        assert debug is True
        self.calls.append((screenshot, detection, destination))
        return destination


class FakeAnalyzer:
    """Return configured immutable board inputs in call order."""

    def __init__(self, inputs: tuple[BoardAnalysisOutcome, ...]) -> None:
        self.inputs = inputs
        self.calls = 0
        self.screenshots: list[Screenshot] = []

    def __call__(self, screenshot: Screenshot) -> cats_app.CatsBoardInput:
        self.screenshots.append(screenshot)
        index = min(self.calls, len(self.inputs) - 1)
        self.calls += 1
        outcome = self.inputs[index]
        if isinstance(outcome, RuntimeError):
            raise outcome
        return outcome


class FakeSolver:
    """Solve each analyzed board into the requested terminal status."""

    def __init__(
        self,
        statuses: tuple[CatsSolveStatus, ...] = (CatsSolveStatus.COMPLETE,),
    ) -> None:
        self.statuses = statuses
        self.calls: list[tuple[WindowInfo, cats_app.CatsBoardInput]] = []

    def __call__(
        self,
        window: WindowInfo,
        board_input: cats_app.CatsBoardInput,
    ) -> cats_app.CatsSolvedBoard:
        index = min(len(self.calls), len(self.statuses) - 1)
        self.calls.append((window, board_input))
        return _solved_board(board_input, window=window, status=self.statuses[index])


def _settings(
    *,
    execute: bool = True,
    max_levels: int = 1,
    poll_interval: float = 0.1,
    timeout: float = 2.0,
    board_retry: float = 3.0,
    overlay_retry: float = 0.75,
    max_overlay_retries: int = 3,
    new_board_delay: float = 0.0,
    click_delay: float = 0.01,
) -> autoplay.CatsAutoplaySettings:
    """Build concise valid settings for runner tests."""

    return autoplay.CatsAutoplaySettings(
        execute=execute,
        click_delay_seconds=click_delay,
        poll_interval_seconds=poll_interval,
        transition_timeout_seconds=timeout,
        board_analysis_retry_seconds=board_retry,
        overlay_retry_seconds=overlay_retry,
        max_overlay_retries=max_overlay_retries,
        new_board_delay_seconds=new_board_delay,
        max_levels=max_levels,
    )


def _runner(
    detections: tuple[CatsScreenStateDetection, ...],
    *,
    settings: autoplay.CatsAutoplaySettings | None = None,
    board_inputs: tuple[BoardAnalysisOutcome, ...] = (_board_input(),),
    statuses: tuple[CatsSolveStatus, ...] = (CatsSolveStatus.COMPLETE,),
    locator: FakeWindowLocator | None = None,
    mouse: FakeMouseController | None = None,
) -> tuple[
    autoplay.CatsAutoplayRunner,
    FakeWindowLocator,
    FakeWindowCapturer,
    FakeCatsScreenStateDetector,
    FakeMouseController,
    FakeClock,
    FakeFailureRenderer,
    FakeAnalyzer,
    FakeSolver,
]:
    """Compose a fully fake runner and expose every collaborator."""

    actual_locator = locator or FakeWindowLocator()
    capturer = FakeWindowCapturer()
    detector = FakeCatsScreenStateDetector(detections)
    actual_mouse = mouse or FakeMouseController()
    clock = FakeClock()
    renderer = FakeFailureRenderer()
    analyzer = FakeAnalyzer(board_inputs)
    solver = FakeSolver(statuses)
    runner = autoplay.CatsAutoplayRunner(
        settings=settings or _settings(),
        locator=actual_locator,
        capturer=capturer,
        detector=detector,
        renderer=renderer,
        mouse_controller=actual_mouse,
        sleep_function=clock.sleep,
        monotonic_function=clock.monotonic,
        analyze_board=analyzer,
        solve_board=solver,
    )
    return (
        runner,
        actual_locator,
        capturer,
        detector,
        actual_mouse,
        clock,
        renderer,
        analyzer,
        solver,
    )


def _board_analysis_failure_state(
    runner: autoplay.CatsAutoplayRunner,
) -> tuple[float | None, RuntimeError | None]:
    """Read retry state without retaining MyPy narrowing across runner calls."""

    return (
        runner._board_analysis_failure_started_at,
        runner._last_board_analysis_error,
    )


def _replace_board_values(
    solved: cats_app.CatsSolvedBoard,
    cats: tuple[tuple[int, int], ...],
) -> cats_app.CatsSolvedBoard:
    """Replace terminal values directly for focused invalid-solution tests."""

    for row in range(4):
        for column in range(4):
            solved.logical_board.cells[row][column] = (
                "K" if (row, column) in cats else "X"
            )
    return replace(solved, click_plan=())
