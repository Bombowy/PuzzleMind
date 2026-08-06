"""Deterministic tests for public screenshot-space grid and cell geometry."""

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from logicforge.config.settings import GridExtractionSettings
from logicforge.infrastructure.opencv_grid_detection_renderer import (
    OpenCvGridDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.infrastructure.opencv_internal_grid_evidence import (
    InternalGridEvidence,
)
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import CellBounds, GridDetectionError
from logicforge.vision.screenshot import Screenshot
from synthetic_vision import (
    advertisement_like_screenshot,
    custom_grid_screenshot,
)


def _screenshot(image: NDArray[np.uint8]) -> Screenshot:
    """Wrap deterministic BGR pixels in the production immutable transport model."""

    height, width = image.shape[:2]
    return Screenshot(
        image=image,
        width=width,
        height=height,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _grid_case(
    rows: int,
    columns: int,
    *,
    screenshot_width: int = 900,
    screenshot_height: int = 700,
    board_x: int = 170,
    board_y: int = 120,
    board_width: int = 560,
    board_height: int = 420,
) -> tuple[Screenshot, BoardDetection]:
    """Create one axis-aligned grid and its exact full-screenshot board contract."""

    image = np.full(
        (screenshot_height, screenshot_width, 3),
        32,
        dtype=np.uint8,
    )
    right = board_x + board_width
    bottom = board_y + board_height
    cv2.rectangle(
        image,
        (board_x, board_y),
        (right, bottom),
        (225, 225, 225),
        -1,
    )
    cv2.rectangle(
        image,
        (board_x, board_y),
        (right, bottom),
        (245, 245, 245),
        6,
    )
    for row in range(1, rows):
        y = board_y + round(board_height * row / rows)
        cv2.line(image, (board_x, y), (right, y), (70, 70, 70), 3)
    for column in range(1, columns):
        x = board_x + round(board_width * column / columns)
        cv2.line(image, (x, board_y), (x, bottom), (70, 70, 70), 3)
    return _screenshot(image), BoardDetection(
        x=board_x,
        y=board_y,
        width=board_width,
        height=board_height,
        confidence=0.90,
    )


class _StubGridAnalyzer:
    """Provide controlled primitive evidence for pixel-conversion failure tests."""

    def __init__(
        self,
        evidence: InternalGridEvidence,
        rejection_reasons: tuple[str, ...] = (),
    ) -> None:
        """Store deterministic evidence without invoking any OpenCV line algorithm."""

        self._evidence = evidence
        self._rejection_reasons = rejection_reasons

    def analyze(self, grayscale_roi: NDArray[np.uint8]) -> InternalGridEvidence:
        """Return configured evidence while accepting the production ROI shape."""

        del grayscale_roi
        return self._evidence

    def rejection_reasons(
        self,
        evidence: InternalGridEvidence,
    ) -> tuple[str, ...]:
        """Return configured hard failures for the supplied evidence record."""

        del evidence
        return self._rejection_reasons


def _stub_evidence(
    horizontal_positions: tuple[float, ...],
    vertical_positions: tuple[float, ...],
) -> InternalGridEvidence:
    """Build otherwise-valid shared evidence around controlled normalized lines."""

    return InternalGridEvidence(
        horizontal_line_positions=horizontal_positions,
        vertical_line_positions=vertical_positions,
        horizontal_line_count=len(horizontal_positions),
        vertical_line_count=len(vertical_positions),
        estimated_rows=len(horizontal_positions) - 1,
        estimated_columns=len(vertical_positions) - 1,
        horizontal_spacing_coefficient_of_variation=0.0,
        vertical_spacing_coefficient_of_variation=0.0,
        horizontal_spacing_regularity=1.0,
        vertical_spacing_regularity=1.0,
        horizontal_line_coverage=1.0,
        vertical_line_coverage=1.0,
        score=1.0,
    )


@pytest.mark.parametrize(
    ("rows", "columns"),
    ((3, 3), (8, 8), (4, 7), (5, 10), (9, 6)),
)
def test_detects_regular_square_and_rectangular_grids(
    rows: int,
    columns: int,
) -> None:
    """Support multiple regular dimensions without square or 8x8 assumptions."""

    screenshot, board = _grid_case(rows, columns)

    grid = OpenCvGridDetector().detect(screenshot, board)

    assert grid.rows == rows
    assert grid.columns == columns
    assert len(grid.horizontal_lines) == rows + 1
    assert len(grid.vertical_lines) == columns + 1
    assert len(grid.cells) == rows * columns
    assert 0.0 <= grid.confidence <= 1.0


@pytest.mark.parametrize(
    (
        "screenshot_width",
        "screenshot_height",
        "board_x",
        "board_y",
        "board_width",
        "board_height",
    ),
    (
        (640, 480, 90, 70, 420, 300),
        (1000, 800, 220, 160, 630, 455),
        (1400, 1000, 310, 180, 840, 630),
    ),
)
def test_coordinates_remain_relative_to_different_full_screenshots(
    screenshot_width: int,
    screenshot_height: int,
    board_x: int,
    board_y: int,
    board_width: int,
    board_height: int,
) -> None:
    """Preserve full-screenshot offsets and exact outer board boundaries."""

    screenshot, board = _grid_case(
        4,
        7,
        screenshot_width=screenshot_width,
        screenshot_height=screenshot_height,
        board_x=board_x,
        board_y=board_y,
        board_width=board_width,
        board_height=board_height,
    )

    grid = OpenCvGridDetector().detect(screenshot, board)

    assert grid.horizontal_lines[0] == board.y
    assert grid.horizontal_lines[-1] == board.y + board.height
    assert grid.vertical_lines[0] == board.x
    assert grid.vertical_lines[-1] == board.x + board.width
    assert all(board.x <= cell.x < board.x + board.width for cell in grid.cells)
    assert all(board.y <= cell.y < board.y + board.height for cell in grid.cells)


def test_cells_are_stable_row_major_half_open_rectangles() -> None:
    """Guarantee ordering, centers, complete coverage, adjacency, and no overlap."""

    screenshot, board = _grid_case(5, 7)

    grid = OpenCvGridDetector().detect(screenshot, board)

    for index, cell in enumerate(grid.cells):
        expected_row, expected_column = divmod(index, grid.columns)
        assert (cell.row, cell.column) == (expected_row, expected_column)
        assert board.x <= cell.x < cell.x + cell.width <= board.x + board.width
        assert board.y <= cell.y < cell.y + cell.height <= board.y + board.height
        assert cell.x <= cell.center_x < cell.x + cell.width
        assert cell.y <= cell.center_y < cell.y + cell.height

    for row in range(grid.rows):
        row_cells = grid.cells[row * grid.columns : (row + 1) * grid.columns]
        assert row_cells[0].x == board.x
        assert row_cells[-1].x + row_cells[-1].width == board.x + board.width
        assert all(
            left.x + left.width == right.x for left, right in pairwise(row_cells)
        )
    for row in range(grid.rows - 1):
        upper_cells = grid.cells[row * grid.columns : (row + 1) * grid.columns]
        lower_cells = grid.cells[(row + 1) * grid.columns : (row + 2) * grid.columns]
        assert all(
            upper.y + upper.height == lower.y
            for upper, lower in zip(upper_cells, lower_cells, strict=True)
        )
    assert sum(cell.width * cell.height for cell in grid.cells) == (
        board.width * board.height
    )


def test_repeated_detection_is_exactly_deterministic() -> None:
    """Return identical immutable lines and cells across repeated analyzer runs."""

    screenshot, board = _grid_case(8, 8)
    detector = OpenCvGridDetector()

    results = tuple(detector.detect(screenshot, board) for _ in range(3))

    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize(
    "board",
    (
        BoardDetection(x=-1, y=20, width=300, height=300, confidence=0.8),
        BoardDetection(x=20, y=-1, width=300, height=300, confidence=0.8),
        BoardDetection(x=20, y=20, width=0, height=300, confidence=0.8),
        BoardDetection(x=20, y=20, width=300, height=0, confidence=0.8),
        BoardDetection(x=20, y=20, width=300, height=300, confidence=-0.1),
        BoardDetection(x=20, y=20, width=300, height=300, confidence=1.1),
    ),
)
def test_rejects_invalid_board_detection_before_cropping(board: BoardDetection) -> None:
    """Fail before ROI access for invalid position, size, or confidence contracts."""

    screenshot = _screenshot(np.zeros((500, 500, 3), dtype=np.uint8))

    with pytest.raises(GridDetectionError) as raised:
        OpenCvGridDetector().detect(screenshot, board)

    assert raised.value.diagnostics.rejection_reasons


def test_rejects_board_outside_screenshot() -> None:
    """Reject a positive rectangle whose right or bottom edge exceeds the image."""

    screenshot = _screenshot(np.zeros((400, 500, 3), dtype=np.uint8))
    board = BoardDetection(x=300, y=250, width=250, height=200, confidence=0.8)

    with pytest.raises(GridDetectionError, match="exceeds screenshot"):
        OpenCvGridDetector().detect(screenshot, board)


def test_rejects_normalized_boundaries_that_round_to_duplicate_pixels() -> None:
    """Fail instead of adjusting collapsed separators or creating zero-width cells."""

    screenshot, board = _grid_case(
        3,
        3,
        screenshot_width=100,
        screenshot_height=100,
        board_x=10,
        board_y=10,
        board_width=20,
        board_height=20,
    )
    evidence = _stub_evidence(
        (0.0, 0.01, 0.50, 1.0),
        (0.0, 0.01, 0.50, 1.0),
    )
    detector = OpenCvGridDetector(analyzer=_StubGridAnalyzer(evidence))

    with pytest.raises(GridDetectionError, match="collapse") as raised:
        detector.detect(screenshot, board)

    assert any(
        "duplicate" in reason for reason in raised.value.diagnostics.rejection_reasons
    )


def test_rejects_cells_below_extraction_pixel_minimum() -> None:
    """Apply extraction-only minimum dimensions without moving public boundaries."""

    screenshot, board = _grid_case(8, 8, board_width=400, board_height=400)
    detector = OpenCvGridDetector(
        extraction_settings=GridExtractionSettings(
            minimum_cell_width_pixels=60,
            minimum_cell_height_pixels=60,
        )
    )

    with pytest.raises(GridDetectionError, match="minimum pixel"):
        detector.detect(screenshot, board)


def test_cell_bounds_reject_zero_sized_geometry() -> None:
    """Protect the public model even if a future adapter bypasses detector checks."""

    with pytest.raises(ValueError, match="positive"):
        CellBounds(
            row=0,
            column=0,
            x=10,
            y=10,
            width=0,
            height=5,
            center_x=10,
            center_y=12,
        )


def test_rejects_irregular_grid() -> None:
    """Reuse shared spacing rules rather than exporting unreliable cell geometry."""

    screenshot = custom_grid_screenshot(
        rows=4,
        columns=4,
        horizontal_positions=(0.12, 0.47, 0.86),
        vertical_positions=(0.10, 0.38, 0.88),
    )
    board = BoardDetection(x=200, y=140, width=400, height=320, confidence=0.9)

    with pytest.raises(GridDetectionError) as raised:
        OpenCvGridDetector().detect(screenshot, board)

    assert any(
        "irregular" in reason for reason in raised.value.diagnostics.rejection_reasons
    )


def test_rejects_insufficient_line_coverage() -> None:
    """Reuse shared line coverage as a mandatory public extraction condition."""

    screenshot = custom_grid_screenshot(
        rows=5,
        columns=5,
        horizontal_coverage=0.42,
        vertical_coverage=0.42,
    )
    board = BoardDetection(x=200, y=140, width=400, height=320, confidence=0.9)

    with pytest.raises(GridDetectionError) as raised:
        OpenCvGridDetector().detect(screenshot, board)

    assert any(
        "coverage" in reason for reason in raised.value.diagnostics.rejection_reasons
    )


def test_advertisement_regression_cannot_produce_public_grid_detection() -> None:
    """Keep the shared synthetic advertisement rejected by both detector layers."""

    screenshot = advertisement_like_screenshot()
    board = BoardDetection(x=200, y=140, width=400, height=320, confidence=0.9)

    with pytest.raises(GridDetectionError) as raised:
        OpenCvGridDetector().detect(screenshot, board)

    assert raised.value.diagnostics.grid_evidence_score < 0.65
    assert raised.value.diagnostics.rejection_reasons


def test_grid_confidence_does_not_depend_on_board_confidence() -> None:
    """Represent grid reliability alone rather than blending board confidence."""

    screenshot, board = _grid_case(4, 7)
    low_board = BoardDetection(
        x=board.x,
        y=board.y,
        width=board.width,
        height=board.height,
        confidence=0.1,
    )
    high_board = BoardDetection(
        x=board.x,
        y=board.y,
        width=board.width,
        height=board.height,
        confidence=0.9,
    )
    detector = OpenCvGridDetector()

    assert (
        detector.detect(screenshot, low_board).confidence
        == detector.detect(
            screenshot,
            high_board,
        ).confidence
    )


def test_debug_false_creates_no_file_or_directory(tmp_path: Path) -> None:
    """Keep normal grid extraction and disabled rendering free of filesystem writes."""

    screenshot, board = _grid_case(4, 7)
    grid = OpenCvGridDetector().detect(screenshot, board)
    output_path = tmp_path / "missing" / "grid_detection.png"

    result = OpenCvGridDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        board,
        grid,
        output_path,
        debug=False,
    )

    assert result is None
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_debug_true_creates_readable_overlay_with_boundaries_and_centers(
    tmp_path: Path,
) -> None:
    """Write one explicit PNG containing all public geometry visualization layers."""

    screenshot, board = _grid_case(4, 7)
    grid = OpenCvGridDetector().detect(screenshot, board)
    output_path = tmp_path / "vision" / "grid_detection.png"

    saved_path = OpenCvGridDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        board,
        grid,
        output_path,
        debug=True,
        detailed=True,
    )
    loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert saved_path == output_path.resolve()
    assert loaded is not None
    assert loaded.shape == screenshot.image.shape
    assert not np.array_equal(loaded, screenshot.image)


@pytest.mark.skip(reason="No anonymized real puzzle screenshot fixture is available.")
def test_public_grid_detection_on_anonymized_real_fixture() -> None:
    """Reserve integration coverage until a safe real screenshot is contributed."""
