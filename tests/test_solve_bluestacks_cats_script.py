"""Deterministic tests for the manual BlueStacks Cats solving script."""

from datetime import UTC, datetime
from inspect import getsource

import numpy as np
import pytest
from scripts import solve_bluestacks_cats as solve_script

from logicforge.core import Board, BoardStateError
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
    WindowInfo,
)


def _screenshot() -> Screenshot:
    """Create an immutable in-memory capture without accessing a desktop."""

    return Screenshot(
        image=np.zeros((100, 120, 3), dtype=np.uint8),
        width=120,
        height=100,
        timestamp=datetime(2026, 8, 6, tzinfo=UTC),
    )


def _board_detection() -> BoardDetection:
    """Return stable board metadata for successful script output."""

    return BoardDetection(x=10, y=20, width=40, height=40, confidence=0.95)


def _grid_detection() -> GridDetection:
    """Build a valid 2x2 grid whose geometry remains irrelevant to fake colors."""

    horizontal_lines = (20, 40, 60)
    vertical_lines = (10, 30, 50)
    cells = tuple(
        CellBounds(
            row=row,
            column=column,
            x=vertical_lines[column],
            y=horizontal_lines[row],
            width=20,
            height=20,
            center_x=vertical_lines[column] + 10,
            center_y=horizontal_lines[row] + 10,
        )
        for row in range(2)
        for column in range(2)
    )
    return GridDetection(
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        rows=2,
        columns=2,
        cells=cells,
        confidence=0.94,
    )


def _rectangular_grid_detection() -> GridDetection:
    """Build the documented 2x3 screenshot-space geometry fixture."""

    horizontal_lines = (100, 200, 300)
    vertical_lines = (50, 150, 250, 350)
    cells = tuple(
        CellBounds(
            row=row,
            column=column,
            x=vertical_lines[column],
            y=horizontal_lines[row],
            width=vertical_lines[column + 1] - vertical_lines[column],
            height=horizontal_lines[row + 1] - horizontal_lines[row],
            center_x=(vertical_lines[column] + vertical_lines[column + 1]) // 2,
            center_y=(horizontal_lines[row] + horizontal_lines[row + 1]) // 2,
        )
        for row in range(2)
        for column in range(3)
    )
    return GridDetection(
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        rows=2,
        columns=3,
        cells=cells,
        confidence=0.94,
    )


def _offset_window(*, x: int = 400, y: int = 300) -> WindowInfo:
    """Return deterministic virtual-desktop window geometry for mapping tests."""

    return WindowInfo(
        title="BlueStacks App Player",
        bounds=WindowBounds(x=x, y=y, width=800, height=600),
    )


def _color_result(
    matrix: tuple[tuple[str, ...], ...] = (("C0", "C1"), ("C2", "C3")),
) -> ColorDetectionResult:
    """Build one complete immutable color result with contiguous logical IDs."""

    rows = len(matrix)
    columns = len(matrix[0])
    color_count = len({value for row in matrix for value in row})
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
    return ColorDetectionResult(
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


def _logical_board() -> Board:
    """Return a fresh actual Board for helper-level state assertions."""

    return Board(_color_result())


def _rectangular_logical_board() -> Board:
    """Return a mutable 2x3 Board aligned with the rectangular grid fixture."""

    return Board(_color_result((("C0", "C1", "C2"), ("C3", "C4", "C5"))))


def _board_error() -> BoardDetectionError:
    """Create a typed synthetic board-localization failure."""

    return BoardDetectionError(
        "synthetic board failure",
        BoardDetectionDiagnostics(
            contour_count=0,
            candidates=(),
            selected_candidate=None,
            competitive_candidate_count=0,
        ),
    )


def _grid_error() -> GridDetectionError:
    """Create a typed synthetic grid-extraction failure."""

    return GridDetectionError(
        "synthetic grid failure",
        GridDetectionDiagnostics(
            board_x=10,
            board_y=20,
            board_width=40,
            board_height=40,
            normalized_horizontal_positions=(),
            normalized_vertical_positions=(),
            horizontal_lines=(),
            vertical_lines=(),
            estimated_rows=0,
            estimated_columns=0,
            horizontal_spacing_coefficient_of_variation=0.0,
            vertical_spacing_coefficient_of_variation=0.0,
            horizontal_coverage=0.0,
            vertical_coverage=0.0,
            grid_evidence_score=0.0,
            rejection_reasons=("synthetic rejection",),
        ),
    )


def _color_error() -> ColorDetectionError:
    """Create a typed synthetic color-classification failure."""

    return ColorDetectionError(
        "synthetic color failure",
        ColorDetectionDiagnostics(
            rows=2,
            columns=2,
            sample_inner_fraction=0.65,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(),
            within_cell_spreads=(),
            cluster_centers_lab=(),
            minimum_intercluster_distance=None,
            rejection_reasons=("synthetic rejection",),
        ),
    )


class _CaptureService:
    """Expose injected window/capture behavior through the production API."""

    screenshot = _screenshot()
    error: WindowCaptureError | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Accept real composition arguments without using OS adapters."""

    def locate_window(self) -> WindowInfo:
        """Return deterministic metadata or raise the configured capture error."""

        if self.error is not None:
            raise self.error
        return WindowInfo(
            title="BlueStacks App Player",
            bounds=WindowBounds(0, 0, self.screenshot.width, self.screenshot.height),
        )

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        """Return the in-memory screenshot without desktop or filesystem access."""

        return self.screenshot


class _BoardDetector:
    """Return stable board geometry or an injected typed failure."""

    error: BoardDetectionError | None = None

    def __init__(self, settings: object) -> None:
        """Accept production settings while keeping the fake deterministic."""

    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Resolve the configured board success or failure branch."""

        if self.error is not None:
            raise self.error
        return _board_detection()


class _GridDetector:
    """Return stable cell geometry or an injected typed failure."""

    error: GridDetectionError | None = None

    def __init__(self, settings: object) -> None:
        """Accept production settings while keeping the fake deterministic."""

    def detect(self, screenshot: Screenshot, board: BoardDetection) -> GridDetection:
        """Resolve the configured grid success or failure branch."""

        if self.error is not None:
            raise self.error
        return _grid_detection()


class _ColorDetector:
    """Return immutable logical colors or an injected typed failure."""

    error: ColorDetectionError | None = None
    result = _color_result()

    def __init__(self, settings: object) -> None:
        """Accept production settings while keeping the fake deterministic."""

    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
    ) -> ColorDetectionResult:
        """Resolve the configured color success or failure branch."""

        if self.error is not None:
            raise self.error
        return self.result


def _configure_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    capture_error: WindowCaptureError | None = None,
    board_error: BoardDetectionError | None = None,
    grid_error: GridDetectionError | None = None,
    color_error: ColorDetectionError | None = None,
) -> None:
    """Replace every OS and OpenCV collaborator with deterministic local fakes."""

    _CaptureService.error = capture_error
    _BoardDetector.error = board_error
    _GridDetector.error = grid_error
    _ColorDetector.error = color_error
    _ColorDetector.result = _color_result()

    monkeypatch.setattr(solve_script, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(solve_script, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(solve_script, "MssWindowCapturer", object)
    monkeypatch.setattr(solve_script, "OpenCvBoardDetector", _BoardDetector)
    monkeypatch.setattr(solve_script, "OpenCvGridDetector", _GridDetector)
    monkeypatch.setattr(solve_script, "OpenCvColorDetector", _ColorDetector)


def _set_complete_result(board: Board) -> int:
    """Mutate all cells to terminal states and report one fake application."""

    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    board.set_blocked(1, 0)
    board.set_cat(1, 1)
    return 1


def _set_stalled_result(board: Board) -> int:
    """Mutate part of the board while deliberately retaining unresolved cells."""

    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    return 2


def test_format_matrix_supports_immutable_tuple_matrix() -> None:
    """Format the immutable color-result representation without conversion."""

    assert solve_script.format_matrix((("C0", "C1"), ("C2", "C3"))) == ("C0 C1\nC2 C3")


def test_format_matrix_supports_mutable_list_matrix() -> None:
    """Format the Board's mutable nested-list representation directly."""

    assert solve_script.format_matrix([["K", "X"], ["C0", "C1"]]) == (" K  X\nC0 C1")


def test_format_matrix_aligns_c2_with_c10() -> None:
    """Right-align short values to the widest logical identifier."""

    assert solve_script.format_matrix((("C2", "C10"), ("X", "K"))) == (
        " C2 C10\n  X   K"
    )


def test_collect_cat_coordinates_returns_only_cats() -> None:
    """Exclude unresolved and blocked values from the diagnostic coordinate list."""

    board = _logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 0)

    assert solve_script.collect_cat_coordinates(board) == ((0, 1),)


def test_collect_cat_coordinates_uses_row_major_order() -> None:
    """Return zero-based coordinates deterministically regardless of mutation order."""

    board = _logical_board()
    board.set_cat(1, 1)
    board.set_cat(0, 1)
    board.set_cat(1, 0)

    assert solve_script.collect_cat_coordinates(board) == ((0, 1), (1, 0), (1, 1))


def test_count_unresolved_cells_counts_only_color_ids() -> None:
    """Count every remaining C<n> entry through the Board query contract."""

    board = _logical_board()
    board.set_cat(0, 0)

    assert solve_script.count_unresolved_cells(board) == 3


def test_cat_and_blocked_cells_are_not_unresolved() -> None:
    """Exclude both terminal Board states from the unresolved count."""

    board = _logical_board()
    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    board.set_cat(1, 0)
    board.set_blocked(1, 1)

    assert solve_script.count_unresolved_cells(board) == 0


def test_classify_result_returns_complete_for_terminal_board() -> None:
    """Report COMPLETE only when no logical color candidate remains."""

    board = _logical_board()
    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    board.set_blocked(1, 0)
    board.set_cat(1, 1)

    assert solve_script.classify_result(board) == "COMPLETE"


def test_classify_result_returns_stalled_with_unknown_cell() -> None:
    """Report STALLED whenever at least one C<n> remains after deductions."""

    board = _logical_board()
    board.set_cat(0, 0)

    assert solve_script.classify_result(board) == "STALLED"


def test_get_grid_cell_returns_first_cell() -> None:
    """Resolve row zero, column zero directly from row-major geometry."""

    grid = _rectangular_grid_detection()

    assert solve_script.get_grid_cell(grid, 0, 0) is grid.cells[0]


def test_get_grid_cell_returns_last_cell() -> None:
    """Resolve the final logical coordinate without scanning all cells."""

    grid = _rectangular_grid_detection()

    assert solve_script.get_grid_cell(grid, 1, 2) is grid.cells[-1]


def test_get_grid_cell_uses_row_major_index() -> None:
    """Map coordinates through row * columns + column."""

    grid = _rectangular_grid_detection()

    assert solve_script.get_grid_cell(grid, 1, 1) is grid.cells[4]


def test_get_grid_cell_rejects_negative_row() -> None:
    """Reject negative logical rows with requested coordinates in the error."""

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=-1, column=0",
    ):
        solve_script.get_grid_cell(_rectangular_grid_detection(), -1, 0)


def test_get_grid_cell_rejects_negative_column() -> None:
    """Reject negative logical columns with requested coordinates in the error."""

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=0, column=-1",
    ):
        solve_script.get_grid_cell(_rectangular_grid_detection(), 0, -1)


def test_get_grid_cell_rejects_row_outside_grid() -> None:
    """Reject a row equal to the public row count."""

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=2, column=0",
    ):
        solve_script.get_grid_cell(_rectangular_grid_detection(), 2, 0)


def test_get_grid_cell_rejects_column_outside_grid() -> None:
    """Reject a column equal to the public column count."""

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=0, column=3",
    ):
        solve_script.get_grid_cell(_rectangular_grid_detection(), 0, 3)


def test_get_grid_cell_rejects_inconsistent_cell_coordinates() -> None:
    """Fail closed if supposedly row-major public geometry is corrupted."""

    grid = _rectangular_grid_detection()
    object.__setattr__(grid.cells[4], "column", 0)

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=1, column=1",
    ):
        solve_script.get_grid_cell(grid, 1, 1)


def test_get_grid_cell_rejects_missing_row_major_entry() -> None:
    """Translate defensive tuple corruption into the local typed error."""

    grid = _rectangular_grid_detection()
    object.__setattr__(grid, "cells", grid.cells[:-1])

    with pytest.raises(
        solve_script.CatClickPlanError,
        match=r"row=1, column=2",
    ):
        solve_script.get_grid_cell(grid, 1, 2)


def test_create_target_preserves_logical_coordinates() -> None:
    """Carry zero-based row and column into the immutable dry-run target."""

    target = solve_script.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert (target.row, target.column) == (1, 2)


def test_create_target_uses_exact_cell_center_x() -> None:
    """Use CellBounds.center_x instead of recalculating board geometry."""

    target = solve_script.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.screenshot_x == 300


def test_create_target_uses_exact_cell_center_y() -> None:
    """Use CellBounds.center_y instead of recalculating board geometry."""

    target = solve_script.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.screenshot_y == 250


def test_create_target_adds_window_x_to_screenshot_center() -> None:
    """Translate screenshot x through the window's virtual-desktop origin."""

    target = solve_script.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.desktop_x == 400 + 300


def test_create_target_adds_window_y_to_screenshot_center() -> None:
    """Translate screenshot y through the window's virtual-desktop origin."""

    target = solve_script.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.desktop_y == 300 + 250


def test_create_target_maps_shifted_window() -> None:
    """Map a non-origin BlueStacks window using direct offset addition."""

    target = solve_script.create_cat_click_target(
        _offset_window(x=300, y=200),
        _rectangular_grid_detection(),
        1,
        1,
    )

    assert (target.screenshot_x, target.screenshot_y) == (200, 250)
    assert (target.desktop_x, target.desktop_y) == (500, 450)


def test_create_target_accepts_negative_window_position() -> None:
    """Preserve valid negative desktop coordinates on a secondary monitor."""

    target = solve_script.create_cat_click_target(
        _offset_window(x=-1000, y=50),
        _rectangular_grid_detection(),
        0,
        0,
    )

    assert (target.desktop_x, target.desktop_y) == (-900, 200)


def test_build_click_plan_contains_only_cat_cells() -> None:
    """Map confirmed K while excluding every other logical state."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 0)

    plan = solve_script.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple((target.row, target.column) for target in plan) == ((0, 1),)


def test_build_click_plan_excludes_blocked_cells() -> None:
    """Never create a future target for X."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 2)

    plan = solve_script.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert all((target.row, target.column) != (1, 2) for target in plan)


def test_build_click_plan_excludes_unresolved_colors() -> None:
    """Never create a future target for C<n>."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)

    plan = solve_script.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert all((target.row, target.column) != (1, 2) for target in plan)


def test_build_click_plan_orders_multiple_cats_row_major() -> None:
    """Retain collect_cat_coordinates ordering regardless of mutation order."""

    board = _rectangular_logical_board()
    board.set_cat(1, 2)
    board.set_cat(0, 1)
    board.set_cat(1, 0)

    plan = solve_script.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple((target.row, target.column) for target in plan) == (
        (0, 1),
        (1, 0),
        (1, 2),
    )


def test_build_click_plan_returns_empty_tuple_without_cats() -> None:
    """Represent a valid no-click dry run without sentinel values."""

    plan = solve_script.build_cat_click_plan(
        _rectangular_logical_board(),
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert plan == ()


def test_build_click_plan_rejects_board_row_count_mismatch() -> None:
    """Validate all Board dimensions before producing any targets."""

    board = Board(_color_result((("C0", "C1", "C2"),)))

    with pytest.raises(solve_script.CatClickPlanError, match="1 rows"):
        solve_script.build_cat_click_plan(
            board,
            _rectangular_grid_detection(),
            _offset_window(),
        )


def test_build_click_plan_rejects_board_column_count_mismatch() -> None:
    """Reject rows that do not match the detected public grid width."""

    with pytest.raises(solve_script.CatClickPlanError, match="2 columns"):
        solve_script.build_cat_click_plan(
            _logical_board(),
            _rectangular_grid_detection(),
            _offset_window(),
        )


def test_dimension_error_creates_no_partial_click_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish complete dimension validation before mapping the first K."""

    board = _rectangular_logical_board()
    board.set_cat(0, 0)
    board.cells[1].pop()
    mapping_calls: list[tuple[int, int]] = []

    def record_mapping(
        window: WindowInfo,
        grid: GridDetection,
        row: int,
        column: int,
    ) -> solve_script.CatClickTarget:
        """Fail the test if mapping starts before shape validation ends."""

        del window, grid
        mapping_calls.append((row, column))
        raise AssertionError("partial click plan was started")

    monkeypatch.setattr(solve_script, "create_cat_click_target", record_mapping)

    with pytest.raises(solve_script.CatClickPlanError):
        solve_script.build_cat_click_plan(
            board,
            _rectangular_grid_detection(),
            _offset_window(),
        )
    assert mapping_calls == []


def test_build_click_plan_does_not_mutate_board() -> None:
    """Keep the sole logical Board unchanged during coordinate mapping."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    expected = tuple(tuple(row) for row in board.cells)

    solve_script.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple(tuple(row) for row in board.cells) == expected


def test_build_click_plan_does_not_mutate_grid() -> None:
    """Treat immutable GridDetection as read-only geometry input."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    grid = _rectangular_grid_detection()
    expected = grid

    solve_script.build_cat_click_plan(board, grid, _offset_window())

    assert grid == expected


def test_print_click_plan_outputs_target_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the number of planned future cat targets."""

    target = solve_script.CatClickTarget(1, 2, 300, 250, 700, 550)

    solve_script.print_cat_click_plan((target,))

    assert "Planned cat click targets: 1" in capsys.readouterr().out


def test_print_click_plan_outputs_logical_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print zero-based row and column for inspection."""

    target = solve_script.CatClickTarget(1, 2, 300, 250, 700, 550)

    solve_script.print_cat_click_plan((target,))

    assert "row=1, column=2" in capsys.readouterr().out


def test_print_click_plan_outputs_screenshot_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the exact CellBounds center in screenshot space."""

    target = solve_script.CatClickTarget(1, 2, 300, 250, 700, 550)

    solve_script.print_cat_click_plan((target,))

    assert "screenshot=(300, 250)" in capsys.readouterr().out


def test_print_click_plan_outputs_desktop_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the offset absolute virtual-desktop position."""

    target = solve_script.CatClickTarget(1, 2, 300, 250, 700, 550)

    solve_script.print_cat_click_plan((target,))

    assert "desktop=(700, 550)" in capsys.readouterr().out


def test_print_empty_click_plan_outputs_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print only a zero-count summary when no K exists."""

    solve_script.print_cat_click_plan(())

    assert capsys.readouterr().out.strip() == "Planned cat click targets: 0"


def test_main_creates_one_board_and_calls_rule_loop_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the sole newly constructed Board to exactly one rule-loop invocation."""

    _configure_pipeline(monkeypatch)
    created_boards: list[Board] = []
    solved_boards: list[Board] = []

    def create_board(result: ColorDetectionResult) -> Board:
        board = Board(result)
        created_boards.append(board)
        return board

    def solve(board: Board) -> int:
        solved_boards.append(board)
        return 0

    monkeypatch.setattr(solve_script, "Board", create_board)
    monkeypatch.setattr(solve_script, "apply_cats_rules_until_stalled", solve)

    assert solve_script.main() == 0
    assert len(created_boards) == 1
    assert solved_boards == created_boards


def test_main_builds_click_plan_after_deduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map final K state only after the single Cats rule-loop invocation."""

    _configure_pipeline(monkeypatch)
    events: list[str] = []

    def solve(board: Board) -> int:
        """Record deduction and create one confirmed cat for the plan."""

        events.append("deduction")
        board.set_cat(0, 0)
        return 1

    def build_plan(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Record that mapping observes the already-mutated Board."""

        del grid, window
        events.append("click-plan")
        assert board.is_cat(0, 0)
        return ()

    monkeypatch.setattr(solve_script, "apply_cats_rules_until_stalled", solve)
    monkeypatch.setattr(solve_script, "build_cat_click_plan", build_plan)

    assert solve_script.main() == 0
    assert events == ["deduction", "click-plan"]


def test_main_reuses_same_window_and_grid_for_click_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass pipeline geometry objects directly to mapping without re-detection."""

    _configure_pipeline(monkeypatch)
    located_windows: list[WindowInfo] = []
    detected_grids: list[GridDetection] = []

    def locate_window(service: _CaptureService) -> WindowInfo:
        """Return and retain one unique WindowInfo instance."""

        del service
        window = _offset_window(x=300, y=200)
        located_windows.append(window)
        return window

    def detect_grid(
        detector: _GridDetector,
        screenshot: Screenshot,
        board: BoardDetection,
    ) -> GridDetection:
        """Return and retain one unique GridDetection instance."""

        del detector, screenshot, board
        grid = _grid_detection()
        detected_grids.append(grid)
        return grid

    def inspect_plan_inputs(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Verify identity of the already-produced pipeline geometry."""

        del board
        assert window is located_windows[0]
        assert grid is detected_grids[0]
        return ()

    monkeypatch.setattr(_CaptureService, "locate_window", locate_window)
    monkeypatch.setattr(_GridDetector, "detect", detect_grid)
    monkeypatch.setattr(solve_script, "apply_cats_rules_until_stalled", lambda board: 0)
    monkeypatch.setattr(solve_script, "build_cat_click_plan", inspect_plan_inputs)

    assert solve_script.main() == 0
    assert len(located_windows) == 1
    assert len(detected_grids) == 1


def test_main_prints_dry_run_click_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Include mapped K centers in successful script diagnostics."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    assert solve_script.main() == 0

    output = capsys.readouterr().out
    assert "Planned cat click targets: 2" in output
    assert "CLICK: row=0, column=0, screenshot=(20, 30), desktop=(20, 30)" in output


def test_complete_main_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a fully finalized deduction result as successful execution."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    assert solve_script.main() == 0


def test_stalled_main_also_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a valid fixed point with unresolved cells as operational success."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )

    assert solve_script.main() == 0


def test_output_contains_initial_immutable_color_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the detector result directly even after the Board has been mutated."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Initial board:\nC0 C1\nC2 C3" in output


def test_output_contains_final_mutable_board_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print finalized K/X values from the same Board consumed by the rule loop."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Final board:\nK X\nX K" in output


def test_output_contains_successful_application_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report successful apply calls rather than inferring a cell-change count."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )

    solve_script.main()

    assert "Successful rule applications: 2" in capsys.readouterr().out


def test_output_contains_all_cat_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print every K using stable zero-based row-major coordinates."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Cats found: 2" in output
    assert "K: row=0, column=0" in output
    assert "K: row=1, column=1" in output


def test_window_capture_error_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated capture-stage exit code and message."""

    _configure_pipeline(
        monkeypatch,
        capture_error=WindowCaptureError("synthetic capture failure"),
    )

    assert solve_script.main() == 1
    assert "synthetic capture failure" in capsys.readouterr().err


def test_board_detection_error_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated board-stage exit code and message."""

    _configure_pipeline(monkeypatch, board_error=_board_error())

    assert solve_script.main() == 2
    assert "synthetic board failure" in capsys.readouterr().err


def test_grid_detection_error_returns_three(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated grid-stage exit code and message."""

    _configure_pipeline(monkeypatch, grid_error=_grid_error())

    assert solve_script.main() == 3
    assert "synthetic grid failure" in capsys.readouterr().err


def test_color_detection_error_returns_four(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated color-stage exit code and message."""

    _configure_pipeline(monkeypatch, color_error=_color_error())

    assert solve_script.main() == 4
    assert "synthetic color failure" in capsys.readouterr().err


def test_board_state_error_from_rules_returns_five(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate only the expected deduction contradiction at the CLI boundary."""

    _configure_pipeline(monkeypatch)

    def fail_deduction(board: Board) -> int:
        raise BoardStateError(f"synthetic contradiction at {board.get(0, 0)}")

    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        fail_deduction,
    )

    assert solve_script.main() == 5
    assert "Cats deduction failed" in capsys.readouterr().err


def test_board_state_error_does_not_print_false_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Avoid presenting a contradiction as either successful terminal status."""

    _configure_pipeline(monkeypatch)

    def fail_deduction(board: Board) -> int:
        raise BoardStateError(f"synthetic contradiction at {board.get(0, 0)}")

    monkeypatch.setattr(
        solve_script,
        "apply_cats_rules_until_stalled",
        fail_deduction,
    )

    solve_script.main()
    captured = capsys.readouterr()

    assert "Status: COMPLETE" not in captured.out
    assert "Status: STALLED" not in captured.out


def test_click_plan_error_returns_six(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose geometry inconsistency through its dedicated script exit code."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(solve_script, "apply_cats_rules_until_stalled", lambda board: 0)

    def fail_mapping(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Raise the expected local mapping error at the script boundary."""

        del board, grid, window
        raise solve_script.CatClickPlanError("synthetic click-plan failure")

    monkeypatch.setattr(solve_script, "build_cat_click_plan", fail_mapping)

    assert solve_script.main() == 6


def test_click_plan_error_prints_actionable_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Describe the failed mapping without masking unexpected exceptions."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(solve_script, "apply_cats_rules_until_stalled", lambda board: 0)

    def fail_mapping(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Raise the typed failure used by the CLI error branch."""

        del board, grid, window
        raise solve_script.CatClickPlanError("synthetic click-plan failure")

    monkeypatch.setattr(solve_script, "build_cat_click_plan", fail_mapping)

    solve_script.main()

    error_output = capsys.readouterr().err
    assert "Cats click-plan mapping failed" in error_output
    assert "synthetic click-plan failure" in error_output


def test_script_imports_no_mouse_or_click_adapter() -> None:
    """Keep this milestone diagnostic-only with no automation dependency."""

    source = getsource(solve_script)

    assert "MouseController" not in source
    assert "logicforge.automation" not in source


def test_script_contains_no_pointer_event_technology() -> None:
    """Keep source free from all explicitly forbidden mouse emitters."""

    source = getsource(solve_script)

    for forbidden_text in (
        "pyautogui",
        "pynput",
        "SetCursorPos",
        "mouse_event",
        "MouseController",
        ".click(",
    ):
        assert forbidden_text not in source


def test_script_imports_no_mouse_event_modules() -> None:
    """Keep dry-run coordinate mapping independent from automation packages."""

    source = getsource(solve_script)

    assert "logicforge.automation" not in source
    assert "import pyautogui" not in source
    assert "import pynput" not in source
