"""Deterministic tests for the complete Cats autoplay state machine."""

from dataclasses import replace
from datetime import UTC, datetime
from inspect import getsource
from pathlib import Path

import numpy as np
import pytest
from scripts import autoplay_bluestacks_cats as autoplay
from scripts import solve_bluestacks_cats as solve_script

from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.core import Board, BoardStateError
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
)
from logicforge.infrastructure.windows import MouseAutomationError
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
    WindowCaptureError,
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
            sample_inner_fraction=0.65,
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
) -> solve_script.CatsBoardInput:
    """Return complete immutable vision input for one board."""

    return solve_script.CatsBoardInput(
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
) -> solve_script.CatsBoardInput:
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
            sample_inner_fraction=0.65,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * (rows * columns),
            within_cell_spreads=(1.0,) * (rows * columns),
            cluster_centers_lab=tuple(
                (120.0 + index, 130.0, 140.0) for index in range(color_count)
            ),
            minimum_intercluster_distance=30.0,
        ),
    )
    return solve_script.CatsBoardInput(
        detected_board=BoardDetection(5, 5, 70, 90, 0.9),
        grid=grid,
        color_result=color_result,
    )


def _solved_board(
    board_input: solve_script.CatsBoardInput | None = None,
    *,
    window: WindowInfo | None = None,
    status: str = "COMPLETE",
    cat_columns: tuple[int, ...] | None = None,
) -> solve_script.CatsSolvedBoard:
    """Create one actual mutable Board and immutable solved wrapper."""

    actual_input = board_input or _board_input()
    actual_window = window or _window()
    logical_board = Board(actual_input.color_result)
    if status == "COMPLETE":
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
    click_plan = solve_script.build_cat_click_plan(
        logical_board,
        actual_input.grid,
        actual_window,
        existing_cat_coordinates=existing_coordinates,
    )
    return solve_script.CatsSolvedBoard(
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

    def __init__(self, inputs: tuple[solve_script.CatsBoardInput, ...]) -> None:
        self.inputs = inputs
        self.calls = 0

    def __call__(self, screenshot: Screenshot) -> solve_script.CatsBoardInput:
        del screenshot
        index = min(self.calls, len(self.inputs) - 1)
        self.calls += 1
        return self.inputs[index]


class FakeSolver:
    """Solve each analyzed board into the requested terminal status."""

    def __init__(self, statuses: tuple[str, ...] = ("COMPLETE",)) -> None:
        self.statuses = statuses
        self.calls: list[tuple[WindowInfo, solve_script.CatsBoardInput]] = []

    def __call__(
        self,
        window: WindowInfo,
        board_input: solve_script.CatsBoardInput,
    ) -> solve_script.CatsSolvedBoard:
        index = min(len(self.calls), len(self.statuses) - 1)
        self.calls.append((window, board_input))
        return _solved_board(board_input, window=window, status=self.statuses[index])


def _settings(
    *,
    execute: bool = True,
    max_levels: int = 1,
    poll_interval: float = 0.1,
    timeout: float = 2.0,
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
        overlay_retry_seconds=overlay_retry,
        max_overlay_retries=max_overlay_retries,
        new_board_delay_seconds=new_board_delay,
        max_levels=max_levels,
    )


def _runner(
    detections: tuple[CatsScreenStateDetection, ...],
    *,
    settings: autoplay.CatsAutoplaySettings | None = None,
    board_inputs: tuple[solve_script.CatsBoardInput, ...] = (_board_input(),),
    statuses: tuple[str, ...] = ("COMPLETE",),
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


@pytest.mark.parametrize(
    "state",
    (
        CatsScreenState.BOARD,
        CatsScreenState.RANKING,
        CatsScreenState.LEVEL_COMPLETE,
        CatsScreenState.UNKNOWN,
    ),
)
def test_dry_run_captures_and_classifies_exactly_once(state: CatsScreenState) -> None:
    """Never enter polling when explicit execution is absent."""

    components = _runner((_detection(state),), settings=_settings(execute=False))
    runner, _, capturer, detector, mouse, clock, _, analyzer, _ = components

    summary = runner.run()

    assert len(capturer.calls) == 1
    assert detector.calls == 1
    assert mouse.clicks == []
    assert clock.sleeps == []
    assert analyzer.calls == (1 if state is CatsScreenState.BOARD else 0)
    assert summary.final_screen_state is state


@pytest.mark.parametrize(
    ("state", "expected_text"),
    (
        (CatsScreenState.BOARD, "complete Cats click plan validated"),
        (CatsScreenState.RANKING, "RANKING single click"),
        (CatsScreenState.LEVEL_COMPLETE, "LEVEL_COMPLETE single click"),
        (CatsScreenState.UNKNOWN, "synthetic unknown state"),
    ),
)
def test_dry_run_prints_state_specific_plan(
    state: CatsScreenState,
    expected_text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explain one planned action without any mouse controller use."""

    runner, *_ = _runner((_detection(state),), settings=_settings(execute=False))

    runner.run()

    assert expected_text in capsys.readouterr().out


def test_board_ranking_level_board_sequence_solves_two_levels() -> None:
    """React to the actual optional-ranking path rather than a fixed screen count."""

    states = tuple(
        _detection(state)
        for state in (
            CatsScreenState.BOARD,
            CatsScreenState.RANKING,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        )
    )
    components = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )
    runner, _, capturer, _, mouse, _, _, analyzer, solver = components

    summary = runner.run()

    assert summary.solved_levels == 2
    assert summary.ranking_clicks == 1
    assert summary.level_button_clicks == 1
    assert summary.low_level_cat_clicks == 16
    assert len(capturer.calls) == 4
    assert analyzer.calls == 2
    assert len(solver.calls) == 2
    assert len(mouse.clicks) == 18


def test_board_level_board_sequence_works_without_ranking() -> None:
    """Allow LEVEL_COMPLETE to follow a solved board directly."""

    states = tuple(
        _detection(state)
        for state in (
            CatsScreenState.BOARD,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        )
    )
    runner, *_, analyzer, solver = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    summary = runner.run()

    assert summary.solved_levels == 2
    assert summary.ranking_clicks == 0
    assert summary.level_button_clicks == 1
    assert analyzer.calls == 2
    assert len(solver.calls) == 2


@pytest.mark.parametrize(
    "initial_states",
    (
        (
            CatsScreenState.RANKING,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        ),
        (CatsScreenState.LEVEL_COMPLETE, CatsScreenState.BOARD),
    ),
)
def test_autoplay_can_start_on_transition_overlay(
    initial_states: tuple[CatsScreenState, ...],
) -> None:
    """Start from either accepted transition state and reach the next board."""

    runner, *_ = _runner(tuple(_detection(state) for state in initial_states))

    summary = runner.run()

    assert summary.solved_levels == 1
    assert summary.level_button_clicks == 1
    assert summary.ranking_clicks == (
        1 if CatsScreenState.RANKING in initial_states else 0
    )


def test_repeated_board_waiting_for_transition_does_not_resolve() -> None:
    """Run the solver only once while the just-clicked board remains visible."""

    states = (
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, *_, analyzer, solver = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    runner.run()

    assert analyzer.calls == 2
    assert len(solver.calls) == 2


def test_new_board_delay_and_old_fingerprint_guard_solver() -> None:
    """Wait after level click and reject the old immutable matrix before solving."""

    states = (
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
    )
    clock_settings = _settings(max_levels=2, new_board_delay=0.2)
    runner, *_, clock, _, analyzer, solver = _runner(
        states,
        settings=clock_settings,
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    summary = runner.run()

    assert summary.solved_levels == 2
    assert clock.sleeps.count(0.1) >= 2
    assert analyzer.calls == 3
    assert len(solver.calls) == 2


@pytest.mark.parametrize("size", (5, 8, 9))
def test_cats_input_geometry_guard_accepts_consistent_square_sizes(size: int) -> None:
    """Permit arbitrary square Cats dimensions when color count and matrix agree."""

    autoplay.validate_cats_board_input_geometry(_geometry_board_input(size, size, size))


def test_cats_input_geometry_guard_rejects_rectangular_tile_lattice() -> None:
    """Keep square Cats validity separate from rectangular lattice geometry."""

    with pytest.raises(autoplay.CatsBoardGeometryMismatchError):
        autoplay.validate_cats_board_input_geometry(_geometry_board_input(6, 9, 9))


def test_generic_refined_9x9_reaches_guard_and_solver_without_autoplay_repair() -> None:
    """Consume final generic vision geometry exactly once through existing ports."""

    refined_input = _geometry_board_input(9, 9, 9)
    runner, _, _, _, mouse, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(refined_input,),
    )

    summary = runner.run()

    assert analyzer.calls == 1
    assert len(solver.calls) == 1
    assert solver.calls[0][1] is refined_input
    assert summary.solved_levels == 1
    assert len(mouse.clicks) == 18


def test_autoplay_uses_shared_tile_grid_analysis_without_copying_cv_pipeline() -> None:
    """Delegate board geometry to the reusable tile-grid-first solve analysis."""

    assert vars(autoplay)["analyze_captured_cats_board"] is (
        solve_script.analyze_captured_cats_board
    )
    source = getsource(autoplay).casefold()
    assert "opencvcatstilegriddetector" not in source
    assert "opencvboarddetector" not in source
    assert "opencvgriddetector" not in source


def test_transient_9x8_geometry_never_solves_clicks_or_updates_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Retry live-like 9x8/colors=9 evidence before Board construction."""

    invalid_input = _geometry_board_input(9, 8, 9)
    runner, _, _, _, mouse, _, renderer, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
        board_inputs=(invalid_input,),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()
    runner.save_failure_overlay()

    output = capsys.readouterr().out
    assert analyzer.calls > 1
    assert solver.calls == []
    assert mouse.clicks == []
    assert runner.summary().solved_levels == 0
    assert runner._last_solved_color_matrix is None
    assert runner._phase is autoplay.CatsAutomationPhase.READY_FOR_BOARD
    assert "rejected transient geometry: grid=9x8, colors=9" in output
    assert "new level accepted" not in output
    assert len(renderer.calls) == 1


def test_transient_9x8_then_9x9_solves_only_the_valid_frame(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Re-run the full poll and accept the corrected 9x9 geometry once."""

    invalid_input = _geometry_board_input(9, 8, 9)
    valid_input = _geometry_board_input(9, 9, 9)
    runner, _, capturer, _, mouse, _, _, analyzer, solver = _runner(
        (
            _detection(CatsScreenState.BOARD),
            _detection(CatsScreenState.BOARD),
        ),
        board_inputs=(invalid_input, valid_input),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 2
    assert analyzer.calls == 2
    assert len(solver.calls) == 1
    assert solver.calls[0][1] is valid_input
    assert len(mouse.clicks) == 18
    assert capsys.readouterr().out.count("new level accepted") == 1


def test_cats_input_geometry_guard_rejects_wrong_color_count() -> None:
    """Reject square 9x9 geometry when immutable vision reports eight colors."""

    with pytest.raises(
        autoplay.CatsBoardGeometryMismatchError,
        match=r"grid=9x9, colors=8",
    ):
        autoplay.validate_cats_board_input_geometry(_geometry_board_input(9, 9, 8))


def test_cats_input_geometry_guard_rejects_matrix_shape_before_solver() -> None:
    """Validate immutable matrix dimensions before any logical Board is created."""

    invalid_input = _geometry_board_input(9, 9, 9)
    object.__setattr__(
        invalid_input.color_result,
        "color_matrix",
        invalid_input.color_result.color_matrix[:-1],
    )
    runner, _, _, _, mouse, _, _, _, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.15),
        board_inputs=(invalid_input,),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    assert solver.calls == []
    assert mouse.clicks == []


def test_dry_run_9x8_returns_typed_validation_failure_without_click() -> None:
    """Surface controlled code-8-compatible geometry failure in one dry poll."""

    runner, _, capturer, _, mouse, _, _, _, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(execute=False),
        board_inputs=(_geometry_board_input(9, 8, 9),),
    )

    with pytest.raises(autoplay.CatsBoardGeometryMismatchError):
        runner.run()

    assert len(capturer.calls) == 1
    assert solver.calls == []
    assert mouse.clicks == []


def test_complete_validation_happens_before_first_cat_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never begin a click plan until full solution validation returns."""

    events: list[str] = []
    original_validate = autoplay.validate_complete_cats_solution

    def validate(solved: solve_script.CatsSolvedBoard) -> None:
        events.append("validate")
        original_validate(solved)

    class EventMouse(FakeMouseController):
        def click(
            self,
            point: ScreenPoint,
            button: MouseButton = MouseButton.LEFT,
        ) -> None:
            events.append("click")
            super().click(point, button)

    monkeypatch.setattr(autoplay, "validate_complete_cats_solution", validate)
    runner, *_ = _runner(
        (_detection(CatsScreenState.BOARD),),
        mouse=EventMouse(),
    )

    runner.run()

    assert events[0] == "validate"
    assert events[1] == "click"


def test_stalled_board_raises_before_any_click_and_saves_one_overlay() -> None:
    """Stop a partial fixed point without emitting a pointer action."""

    runner, _, _, _, mouse, _, renderer, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        statuses=("STALLED",),
    )

    with pytest.raises(autoplay.CatsSolutionValidationError, match="STALLED"):
        runner.run()
    runner.save_failure_overlay()
    runner.save_failure_overlay()

    assert mouse.clicks == []
    assert len(renderer.calls) == 1


def _replace_board_values(
    solved: solve_script.CatsSolvedBoard,
    cats: tuple[tuple[int, int], ...],
) -> solve_script.CatsSolvedBoard:
    """Replace terminal values directly for focused invalid-solution tests."""

    for row in range(4):
        for column in range(4):
            solved.logical_board.cells[row][column] = (
                "K" if (row, column) in cats else "X"
            )
    return replace(solved, click_plan=())


@pytest.mark.parametrize(
    "cats",
    (
        ((0, 1), (1, 3), (2, 0)),
        ((0, 1), (0, 2), (1, 3), (2, 0), (3, 2)),
        ((0, 1), (1, 3), (3, 2)),
        ((0, 1), (1, 3), (2, 0), (2, 2), (3, 2)),
    ),
)
def test_validation_rejects_missing_or_duplicate_row_or_column_cat(
    cats: tuple[tuple[int, int], ...],
) -> None:
    """Require exactly one K in every row and every column."""

    solved = _replace_board_values(_solved_board(), cats)

    with pytest.raises(autoplay.CatsSolutionValidationError):
        autoplay.validate_complete_cats_solution(solved)


def test_validation_rejects_duplicate_or_missing_original_color() -> None:
    """Require exactly one K for every immutable original color identifier."""

    matrix = [list(row) for row in COLUMN_COLOR_MATRIX]
    matrix[0][1] = "C0"
    matrix[0][0] = "C1"
    invalid_input = _board_input(tuple(tuple(row) for row in matrix))
    solved = _solved_board(invalid_input)

    with pytest.raises(autoplay.CatsSolutionValidationError, match="Original color"):
        autoplay.validate_complete_cats_solution(solved)


def test_validation_rejects_touching_cats() -> None:
    """Reject orthogonal or diagonal adjacency after row/column validity."""

    solved = _solved_board(cat_columns=(0, 1, 3, 2))

    with pytest.raises(autoplay.CatsSolutionValidationError, match="touch"):
        autoplay.validate_complete_cats_solution(solved)


@pytest.mark.parametrize("value", ("C0", "?"))
def test_validation_rejects_unresolved_or_unsupported_value(value: str) -> None:
    """Allow only K and X in a claimed complete solution."""

    solved = _solved_board()
    solved.logical_board.cells[0][0] = value

    with pytest.raises(autoplay.CatsSolutionValidationError):
        autoplay.validate_complete_cats_solution(solved)


def test_validation_rejects_click_plan_mismatch_and_duplicate() -> None:
    """Require a unique row-major plan exactly equal to all K coordinates."""

    solved = _solved_board()
    missing = replace(solved, click_plan=solved.click_plan[:-1])
    duplicate = replace(
        solved,
        click_plan=(*solved.click_plan, solved.click_plan[0]),
    )

    with pytest.raises(autoplay.CatsSolutionValidationError, match="match"):
        autoplay.validate_complete_cats_solution(missing)
    with pytest.raises(autoplay.CatsSolutionValidationError, match="duplicate"):
        autoplay.validate_complete_cats_solution(duplicate)


def test_complete_six_by_six_with_one_existing_cat_has_five_new_targets() -> None:
    board_input = _geometry_board_input(6, 6, 6)
    board_input = replace(
        board_input,
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(1, 0, 0.93),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    solved = _solved_board(board_input, cat_columns=SIX_CAT_COLUMNS)

    autoplay.validate_complete_cats_solution(solved)

    assert len(solve_script.collect_cat_coordinates(solved.logical_board)) == 6
    assert len(solved.click_plan) == 5
    assert (1, 0) not in tuple(
        (target.row, target.column) for target in solved.click_plan
    )


def test_autoplay_executes_only_new_cats_when_one_is_existing() -> None:
    board_input = replace(
        _geometry_board_input(6, 6, 6),
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(1, 0, 0.93),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(board_input,),
    )

    summary = runner.run()

    assert summary.low_level_cat_clicks == 10
    assert len(mouse.clicks) == 10
    existing_cell = board_input.grid.cells[1 * 6]
    existing_desktop = ScreenPoint(
        _window().bounds.x + existing_cell.center_x,
        _window().bounds.y + existing_cell.center_y,
    )
    assert all(point != existing_desktop for point, _ in mouse.clicks)


def test_several_existing_cats_reduce_autoplay_click_count() -> None:
    existing_coordinates = ((1, 0), (3, 1), (5, 3))
    board_input = replace(
        _geometry_board_input(6, 6, 6),
        existing_cat_detection=CatsExistingCatDetection(
            cats=tuple(
                CatsExistingCatObservation(row, column, 0.9)
                for row, column in existing_coordinates
            ),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(board_input,),
    )

    summary = runner.run()

    assert summary.low_level_cat_clicks == 6
    assert len(mouse.clicks) == 6


def test_existing_cat_missing_from_final_k_fails_validation() -> None:
    solved = _solved_board()
    invalid_input = replace(
        solved.board_input,
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(0, 0, 0.9),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    invalid = replace(solved, board_input=invalid_input)
    with pytest.raises(autoplay.CatsSolutionValidationError, match="not K"):
        autoplay.validate_complete_cats_solution(invalid)


def test_duplicate_existing_coordinates_fail_validation() -> None:
    solved = _solved_board()
    detection = CatsExistingCatDetection(
        cats=(CatsExistingCatObservation(0, 1, 0.9),),
        diagnostics=CatsExistingCatDiagnostics(cells=()),
    )
    object.__setattr__(
        detection,
        "cats",
        (
            CatsExistingCatObservation(0, 1, 0.9),
            CatsExistingCatObservation(0, 1, 0.9),
        ),
    )
    invalid = replace(
        solved,
        board_input=replace(solved.board_input, existing_cat_detection=detection),
    )
    with pytest.raises(autoplay.CatsSolutionValidationError, match="duplicate"):
        autoplay.validate_complete_cats_solution(invalid)


def test_existing_cat_detection_contradiction_emits_zero_clicks() -> None:
    locator = FakeWindowLocator()
    capturer = FakeWindowCapturer()
    detector = FakeCatsScreenStateDetector((_detection(CatsScreenState.BOARD),))
    mouse = FakeMouseController()
    clock = FakeClock()

    def reject_analysis(screenshot: Screenshot) -> solve_script.CatsBoardInput:
        del screenshot
        diagnostics = CatsExistingCatDiagnostics(
            cells=(),
            rejection_reasons=("multiple existing cats were detected in one row",),
        )
        raise CatsExistingCatDetectionError("synthetic contradiction", diagnostics)

    runner = autoplay.CatsAutoplayRunner(
        settings=_settings(),
        locator=locator,
        capturer=capturer,
        detector=detector,
        renderer=FakeFailureRenderer(),
        mouse_controller=mouse,
        sleep_function=clock.sleep,
        monotonic_function=clock.monotonic,
        analyze_board=reject_analysis,
    )

    with pytest.raises(CatsExistingCatDetectionError):
        runner.run()
    assert mouse.clicks == []


def test_validation_rejects_inconsistent_rows_columns_and_color_count() -> None:
    """Fail when one-cat row, column, and color counts cannot all agree."""

    solved = _solved_board()
    object.__setattr__(solved.board_input.color_result, "color_count", 3)

    with pytest.raises(autoplay.CatsSolutionValidationError, match="equal"):
        autoplay.validate_complete_cats_solution(solved)


def test_eight_cat_targets_emit_sixteen_row_major_low_level_clicks() -> None:
    """Retain existing two-click orchestration and target ordering."""

    first = _solved_board()
    second = _solved_board(_board_input(ROW_COLOR_MATRIX))
    targets = (*first.click_plan, *second.click_plan)
    mouse = FakeMouseController()
    clock = FakeClock()

    executed = solve_script.execute_cat_click_plan(
        targets,
        mouse,
        click_delay_seconds=0.01,
        sleep_function=clock.sleep,
    )

    assert executed == 8
    assert len(mouse.clicks) == 16
    assert [click[0] for click in mouse.clicks[::2]] == [
        ScreenPoint(target.desktop_x, target.desktop_y) for target in targets
    ]
    assert set(clock.sleeps) == {0.01}


@pytest.mark.parametrize(
    ("state", "expected_phase", "counter_name"),
    (
        (
            CatsScreenState.RANKING,
            autoplay.CatsAutomationPhase.WAITING_FOR_LEVEL_COMPLETE,
            "ranking_clicks",
        ),
        (
            CatsScreenState.LEVEL_COMPLETE,
            autoplay.CatsAutomationPhase.WAITING_FOR_NEXT_BOARD,
            "level_button_clicks",
        ),
    ),
)
def test_overlay_emits_one_left_click_and_converts_desktop_coordinates(
    state: CatsScreenState,
    expected_phase: autoplay.CatsAutomationPhase,
    counter_name: str,
) -> None:
    """Single-click accepted screenshot actions, including negative desktop x."""

    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(state),),
        settings=_settings(max_levels=0, timeout=0.2),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    expected_y = 70 if state is CatsScreenState.RANKING else 85
    assert mouse.clicks[0] == (
        ScreenPoint(-1470, 1 + expected_y),
        MouseButton.LEFT,
    )
    assert len(mouse.clicks) == 1
    assert getattr(runner.summary(), counter_name) == 1
    assert runner._phase is expected_phase


def test_bounds_change_skips_stale_click_and_recaptures() -> None:
    """Never translate old geometry by a window delta after BlueStacks moves."""

    old = _window()
    moved = _window(x=-1300)
    locator = FakeWindowLocator((old, moved, old, old, old, old))
    states = (
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, _, capturer, _, mouse, _, _, _, _ = _runner(
        states,
        locator=locator,
    )

    runner.run()

    assert len(capturer.calls) == 4
    assert mouse.clicks[0][0] == ScreenPoint(-1470, 71)
    assert all(point.x != -1270 for point, _ in mouse.clicks)


def test_every_low_level_cat_click_rechecks_window_bounds() -> None:
    """Guard every delegated click while retaining the existing plan executor."""

    runner, locator, capturer, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),)
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(mouse.clicks) == 8
    assert len(capturer.calls) == 1
    assert locator.calls == 1 + 1 + len(mouse.clicks)


def test_bounds_change_before_first_cat_click_forces_fresh_poll() -> None:
    """Discard the old plan when movement occurs between validation and click."""

    old = _window()
    moved = _window(x=-1300)
    locator = FakeWindowLocator((old, old, moved))
    runner, _, capturer, _, mouse, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        locator=locator,
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 2
    assert analyzer.calls == 2
    assert len(solver.calls) == 2
    assert len(mouse.clicks) == 8
    assert mouse.clicks[0][0].x >= moved.bounds.x


def test_stationary_overlay_retries_only_after_delay_and_stops_at_limit() -> None:
    """Throttle identical overlays and fail after bounded delayed retries."""

    runner, _, _, _, mouse, clock, renderer, _, _ = _runner(
        (_detection(CatsScreenState.RANKING),),
        settings=_settings(
            max_levels=0,
            timeout=10.0,
            poll_interval=0.1,
            overlay_retry=0.3,
            max_overlay_retries=2,
        ),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError, match="2 overlay retries"):
        runner.run()
    runner.save_failure_overlay()

    assert len(mouse.clicks) == 3
    assert clock.now >= 0.9
    assert len(renderer.calls) == 1


def test_unknown_never_clicks_or_solves_and_eventually_times_out() -> None:
    """Keep stationary UNKNOWN passive while enforcing progress timeout."""

    runner, _, _, _, mouse, clock, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.UNKNOWN),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    assert mouse.clicks == []
    assert analyzer.calls == 0
    assert solver.calls == []
    assert clock.sleeps == [0.1, 0.1, 0.1]


def test_unknown_can_be_interleaved_and_state_change_resets_timeout() -> None:
    """Treat recognized state changes as progress without clicking UNKNOWN."""

    states = (
        _detection(CatsScreenState.UNKNOWN),
        _detection(CatsScreenState.RANKING),
        _detection(CatsScreenState.UNKNOWN),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        states,
        settings=_settings(timeout=0.15, poll_interval=0.1),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert summary.ranking_clicks == 1
    assert summary.level_button_clicks == 1
    assert len(mouse.clicks) == 10


def test_stationary_board_waiting_for_transition_times_out() -> None:
    """Do not let an unchanged post-solve BOARD reset progress forever."""

    runner, _, _, _, _, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.25, poll_interval=0.1),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    assert analyzer.calls == 1
    assert len(solver.calls) == 1


def test_max_levels_one_stops_without_poll_sleep_or_transition_click() -> None:
    """Return immediately after the requested board count is clicked."""

    runner, _, capturer, _, mouse, clock, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=1),
    )

    summary = runner.run()

    assert summary.solved_levels == 1
    assert len(capturer.calls) == 1
    assert len(mouse.clicks) == 8
    assert 0.1 not in clock.sleeps


def test_max_levels_zero_continues_until_timeout() -> None:
    """Interpret zero as unlimited rather than immediate completion."""

    runner, _, capturer, _, _, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        settings=_settings(max_levels=0, timeout=0.15),
    )

    with pytest.raises(autoplay.CatsAutomationTimeoutError):
        runner.run()

    assert len(capturer.calls) > 1


def test_summary_prints_all_required_counts(capsys: pytest.CaptureFixture[str]) -> None:
    """Report levels, overlays, cat targets, low-level clicks, and final state."""

    autoplay.print_autoplay_summary(
        autoplay.CatsAutoplaySummary(2, 1, 1, 16, CatsScreenState.BOARD)
    )

    output = capsys.readouterr().out
    assert "Solved levels: 2" in output
    assert "Ranking clicks: 1" in output
    assert "Level-button clicks: 1" in output
    assert "Cat targets executed: 8" in output
    assert "Low-level cat clicks: 16" in output
    assert "Final screen state: BOARD" in output


def test_successful_session_does_not_save_failure_overlay() -> None:
    """Keep normal polling free from debug filesystem writes."""

    runner, _, _, _, _, _, renderer, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),)
    )

    runner.run()

    assert renderer.calls == []


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    (
        (WindowCaptureError("capture"), 1),
        (CatsScreenStateDetectionError("screen"), 2),
        (
            BoardDetectionError(
                "board",
                BoardDetectionDiagnostics(0, (), None, 0),
            ),
            3,
        ),
        (
            GridDetectionError(
                "grid",
                GridDetectionDiagnostics(
                    0, 0, 1, 1, (), (), (), (), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, ()
                ),
            ),
            4,
        ),
        (
            ColorDetectionError(
                "color",
                ColorDetectionDiagnostics(1, 1, 0.5, 1.0, (), (), (), None, ("error",)),
            ),
            5,
        ),
        (BoardStateError("board state"), 6),
        (solve_script.CatClickPlanError("plan"), 7),
        (autoplay.CatsSolutionValidationError("validation"), 8),
        (autoplay.CatsBoardGeometryMismatchError("geometry"), 8),
        (MouseAutomationError("mouse"), 9),
        (solve_script.CatClickExecutionError("execution"), 9),
        (autoplay.CatsAutomationTimeoutError("timeout"), 10),
    ),
)
def test_main_maps_expected_errors_to_documented_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    exception: RuntimeError,
    expected_code: int,
) -> None:
    """Translate expected operational failures without broad exception handling."""

    class FailingRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self) -> autoplay.CatsAutoplaySummary:
            raise exception

        def summary(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(0, 0, 0, 0, CatsScreenState.UNKNOWN)

        def save_failure_overlay(self) -> None:
            return None

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", FailingRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", object)

    assert autoplay.main(("--execute",)) == expected_code


def test_keyboard_interrupt_returns_130_without_click(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch Ctrl+C only at main and print the partial summary."""

    mouse = FakeMouseController()

    class InterruptedRunner:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self) -> autoplay.CatsAutoplaySummary:
            raise KeyboardInterrupt

        def summary(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(1, 0, 0, 8, CatsScreenState.BOARD)

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", InterruptedRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", lambda: mouse)

    assert autoplay.main(("--execute",)) == 130
    assert mouse.clicks == []
    assert "Cats autoplay stopped by user." in capsys.readouterr().out


def test_dry_main_does_not_construct_win32_mouse_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep native mouse composition strictly behind --execute."""

    class SuccessfulRunner:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["mouse_controller"] is None

        def run(self) -> autoplay.CatsAutoplaySummary:
            return autoplay.CatsAutoplaySummary(0, 0, 0, 0, CatsScreenState.UNKNOWN)

    def reject_mouse() -> MouseController:
        raise AssertionError("dry-run must not construct Win32MouseController")

    monkeypatch.setattr(autoplay, "CatsAutoplayRunner", SuccessfulRunner)
    monkeypatch.setattr(autoplay, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(autoplay, "MssWindowCapturer", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDetector", object)
    monkeypatch.setattr(autoplay, "OpenCvCatsScreenStateDebugRenderer", object)
    monkeypatch.setattr(autoplay, "Win32MouseController", reject_mouse)

    assert autoplay.main(()) == 0


def test_cli_defaults_and_validation() -> None:
    """Expose the documented defaults and reject every unsafe boundary."""

    arguments = autoplay.parse_arguments(())

    assert arguments.execute is False
    assert arguments.click_delay_ms == 10
    assert arguments.poll_interval_ms == 100
    assert arguments.transition_timeout_seconds == 20.0
    assert arguments.overlay_retry_ms == 750
    assert arguments.max_overlay_retries == 3
    assert arguments.new_board_delay_ms == 300
    assert arguments.max_levels == 0


@pytest.mark.parametrize(
    "arguments",
    (
        ("--click-delay-ms", "-1"),
        ("--poll-interval-ms", "9"),
        ("--transition-timeout-seconds", "0"),
        ("--overlay-retry-ms", "99"),
        ("--max-overlay-retries", "0"),
        ("--new-board-delay-ms", "-1"),
        ("--max-levels", "-1"),
    ),
)
def test_cli_rejects_invalid_values(arguments: tuple[str, str]) -> None:
    """Use argparse failures for invalid timing and count options."""

    with pytest.raises(SystemExit) as error:
        autoplay.parse_arguments(arguments)

    assert error.value.code == 2


def test_source_uses_only_accepted_screen_detection() -> None:
    """Forbid duplicated vision heuristics and unsafe external technologies."""

    source = getsource(autoplay).casefold()
    forbidden = (
        "cv2.inrange",
        "cv2.findcontours",
        "pytesseract",
        "easyocr",
        "matchtemplate",
        "subprocess",
        "pyautogui",
        "pynput",
    )

    for text in forbidden:
        assert text not in source
    assert "opencvcatsscreenstatedetector" in source
    assert "detection.action_point" in source
