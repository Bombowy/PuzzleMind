"""Puzzle-neutral contracts for public grid and cell geometry extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite

from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class CellBounds:
    """Describe one zero-based logical cell in full-screenshot pixel coordinates.

    ``x`` and ``y`` are inclusive. ``x + width`` and ``y + height`` are exclusive.
    Adjacent cells therefore share a boundary without sharing pixels. The center is
    an integer pixel position guaranteed to lie inside the same half-open cell.
    """

    row: int
    column: int
    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int

    def __post_init__(self) -> None:
        """Reject invalid indices, geometry, or centers at the public boundary."""

        if self.row < 0 or self.column < 0:
            raise ValueError(
                "Cell row and column must be zero-based non-negative values."
            )
        if self.x < 0 or self.y < 0:
            raise ValueError("Cell coordinates must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Cell width and height must be positive.")
        if not self.x <= self.center_x < self.x + self.width:
            raise ValueError("Cell center_x must lie inside its half-open x interval.")
        if not self.y <= self.center_y < self.y + self.height:
            raise ValueError("Cell center_y must lie inside its half-open y interval.")


@dataclass(frozen=True, slots=True)
class GridDetection:
    """Expose complete grid boundaries and cells in screenshot coordinates.

    Boundary tuples include both outer board edges. Cells are ordered row-major and
    use the same half-open coordinate semantics as ``CellBounds``. Confidence
    measures only public grid-geometry reliability and excludes board confidence.
    """

    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]
    rows: int
    columns: int
    cells: tuple[CellBounds, ...]
    confidence: float

    def __post_init__(self) -> None:
        """Enforce complete, monotonic, row-major public geometry invariants."""

        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Grid rows and columns must be positive.")
        if len(self.horizontal_lines) != self.rows + 1:
            raise ValueError("Horizontal line count must equal rows plus one.")
        if len(self.vertical_lines) != self.columns + 1:
            raise ValueError("Vertical line count must equal columns plus one.")
        if any(line < 0 for line in (*self.horizontal_lines, *self.vertical_lines)):
            raise ValueError("Grid line coordinates must be non-negative.")
        if any(
            current <= previous for previous, current in pairwise(self.horizontal_lines)
        ):
            raise ValueError("Horizontal lines must be strictly increasing.")
        if any(
            current <= previous for previous, current in pairwise(self.vertical_lines)
        ):
            raise ValueError("Vertical lines must be strictly increasing.")
        if len(self.cells) != self.rows * self.columns:
            raise ValueError("Cell count must equal rows times columns.")
        for index, cell in enumerate(self.cells):
            expected_row, expected_column = divmod(index, self.columns)
            if (cell.row, cell.column) != (expected_row, expected_column):
                raise ValueError("Cells must use stable zero-based row-major ordering.")
            expected_left = self.vertical_lines[expected_column]
            expected_right = self.vertical_lines[expected_column + 1]
            expected_top = self.horizontal_lines[expected_row]
            expected_bottom = self.horizontal_lines[expected_row + 1]
            if (
                cell.x != expected_left
                or cell.y != expected_top
                or cell.width != expected_right - expected_left
                or cell.height != expected_bottom - expected_top
            ):
                raise ValueError(
                    "Every cell must exactly tile its adjacent public boundaries."
                )
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Grid confidence must be finite within 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class GridDetectionDiagnostics:
    """Carry primitive evidence and failure details without backend-specific types."""

    board_x: int
    board_y: int
    board_width: int
    board_height: int
    normalized_horizontal_positions: tuple[float, ...]
    normalized_vertical_positions: tuple[float, ...]
    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]
    estimated_rows: int
    estimated_columns: int
    horizontal_spacing_coefficient_of_variation: float
    vertical_spacing_coefficient_of_variation: float
    horizontal_coverage: float
    vertical_coverage: float
    grid_evidence_score: float
    rejection_reasons: tuple[str, ...]


class GridDetectionError(RuntimeError):
    """Report fail-closed grid extraction with actionable primitive diagnostics."""

    def __init__(self, message: str, diagnostics: GridDetectionDiagnostics) -> None:
        """Retain structured evidence while presenting a concise operational error."""

        super().__init__(message)
        self.diagnostics = diagnostics


class GridDetector(ABC):
    """Define the port converting a validated board into explicit cell geometry."""

    @abstractmethod
    def detect(self, screenshot: Screenshot, board: BoardDetection) -> GridDetection:
        """Return complete screenshot-space geometry or raise ``GridDetectionError``."""

        raise NotImplementedError
