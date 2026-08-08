"""Shared fixtures for Cats solve application and CLI tests."""

from datetime import UTC, datetime

import numpy as np
import pytest
from scripts import solve_bluestacks_cats as solve_cli

from logicforge.application import cats as cats_app
from logicforge.application.cats import solving as cats_solving
from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.core import Board
from logicforge.infrastructure.windows import MouseAutomationError
from logicforge.plugins.cats.exact_search import (
    CatsExactSearchResult,
    CatsExactSearchStatus,
)
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDiagnostics,
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


def _square_grid_detection(size: int) -> GridDetection:
    """Build row-major square geometry for exact-search orchestration tests."""

    lines = tuple(index * 20 for index in range(size + 1))
    return GridDetection(
        horizontal_lines=lines,
        vertical_lines=lines,
        rows=size,
        columns=size,
        cells=tuple(
            CellBounds(
                row=row,
                column=column,
                x=lines[column],
                y=lines[row],
                width=20,
                height=20,
                center_x=lines[column] + 10,
                center_y=lines[row] + 10,
            )
            for row in range(size)
            for column in range(size)
        ),
        confidence=0.95,
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


def _click_targets() -> tuple[cats_app.CatClickTarget, ...]:
    """Return two immutable targets in deterministic logical row-major order."""

    return (
        cats_app.CatClickTarget(0, 1, 20, 30, 420, 330),
        cats_app.CatClickTarget(1, 2, 40, 50, 440, 350),
    )


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


class _ExistingCatDetector:
    """Return no already-present cats for legacy solve fixtures."""

    calls = 0

    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        colors: ColorDetectionResult,
    ) -> CatsExistingCatDetection:
        del screenshot, grid, colors
        type(self).calls += 1
        return CatsExistingCatDetection(
            cats=(),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        )


class _FakeMouseController(MouseController):
    """Record portable click calls without touching the real native pointer."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        """Optionally fail on one zero-based attempted click index."""

        self.fail_on_call = fail_on_call
        self.clicks: list[tuple[ScreenPoint, MouseButton]] = []

    def click(
        self,
        point: ScreenPoint,
        button: MouseButton = MouseButton.LEFT,
    ) -> None:
        """Record or fail one click deterministically."""

        click_index = len(self.clicks)
        self.clicks.append((point, button))
        if click_index == self.fail_on_call:
            raise MouseAutomationError("synthetic native click failure")


class _FakeSleep:
    """Record requested delays without pausing the test process."""

    def __init__(self) -> None:
        """Start with no recorded sleeps."""

        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        """Retain one requested delay value."""

        self.calls.append(seconds)


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
    _ExistingCatDetector.calls = 0

    monkeypatch.setattr(solve_cli, "WindowCaptureService", _CaptureService)
    monkeypatch.setattr(solve_cli, "Win32BlueStacksWindowLocator", object)
    monkeypatch.setattr(solve_cli, "MssWindowCapturer", object)
    monkeypatch.setattr(solve_cli, "OpenCvBoardDetector", _BoardDetector)
    monkeypatch.setattr(solve_cli, "OpenCvGridDetector", _GridDetector)
    monkeypatch.setattr(solve_cli, "OpenCvColorDetector", _ColorDetector)
    monkeypatch.setattr(
        solve_cli,
        "OpenCvCatsExistingCatDetector",
        _ExistingCatDetector,
    )
    monkeypatch.setattr(
        cats_solving,
        "solve_cats_exact",
        lambda *args, **kwargs: _ambiguous_exact_result(),
    )


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


def _ambiguous_exact_result() -> CatsExactSearchResult:
    """Return a controlled fail-closed fallback result for legacy fake boards."""

    return CatsExactSearchResult(
        status=CatsExactSearchStatus.AMBIGUOUS,
        solution=None,
        solutions_found=2,
        search_nodes=4,
        propagation_steps=1,
    )
