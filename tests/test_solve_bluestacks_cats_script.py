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


def test_script_imports_no_mouse_or_click_adapter() -> None:
    """Keep this milestone diagnostic-only with no automation dependency."""

    source = getsource(solve_script)

    assert "MouseController" not in source
    assert "logicforge.automation" not in source
