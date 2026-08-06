"""Backend-neutral Cats tile-grid detection contracts and diagnostics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class CatsTileComponentDiagnostic:
    """Describe one colored connected component using primitive image evidence."""

    x: int
    y: int
    width: int
    height: int
    center_x: float
    center_y: float
    area: int
    fill_ratio: float
    aspect_ratio: float
    mean_lab_chroma: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep component geometry and numeric evidence finite and meaningful."""

        if min(self.x, self.y, self.area) < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Cats tile component geometry must be non-negative.")
        if not isfinite(self.center_x) or not isfinite(self.center_y):
            raise ValueError("Cats tile component centers must be finite.")
        if not isfinite(self.fill_ratio) or not 0.0 <= self.fill_ratio <= 1.0:
            raise ValueError("Cats tile component fill_ratio must be within 0..1.")
        if not isfinite(self.aspect_ratio) or self.aspect_ratio <= 0.0:
            raise ValueError("Cats tile component aspect_ratio must be positive.")
        if not isfinite(self.mean_lab_chroma) or self.mean_lab_chroma < 0.0:
            raise ValueError("Cats tile component LAB chroma must be non-negative.")


@dataclass(frozen=True, slots=True)
class CatsTileGridDiagnostics:
    """Expose primitive tile-family, lattice, and slot-assignment evidence."""

    component_count: int
    candidate_tile_count: int
    selected_tile_count: int
    row_count: int
    column_count: int
    row_centers: tuple[float, ...]
    column_centers: tuple[float, ...]
    horizontal_pitch: float
    vertical_pitch: float
    horizontal_pitch_cv: float
    vertical_pitch_cv: float
    median_tile_width: float
    median_tile_height: float
    tile_width_cv: float
    tile_height_cv: float
    occupancy_ratio: float
    missing_slot_coordinates: tuple[tuple[int, int], ...]
    row_component_counts: tuple[int, ...]
    column_component_counts: tuple[int, ...]
    row_support_ratios: tuple[float, ...]
    column_support_ratios: tuple[float, ...]
    minimum_row_support_ratio: float
    minimum_column_support_ratio: float
    mean_slot_residual: float
    grid_score: float
    components: tuple[CatsTileComponentDiagnostic, ...]
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject backend objects indirectly by validating primitive diagnostics."""

        counts = (
            self.component_count,
            self.candidate_tile_count,
            self.selected_tile_count,
            self.row_count,
            self.column_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("Cats tile-grid diagnostic counts must be non-negative.")
        if len(self.row_centers) != self.row_count:
            raise ValueError("row_centers must match row_count.")
        if len(self.column_centers) != self.column_count:
            raise ValueError("column_centers must match column_count.")
        if len(self.row_component_counts) != self.row_count:
            raise ValueError("row_component_counts must match row_count.")
        if len(self.column_component_counts) != self.column_count:
            raise ValueError("column_component_counts must match column_count.")
        if len(self.row_support_ratios) != self.row_count:
            raise ValueError("row_support_ratios must match row_count.")
        if len(self.column_support_ratios) != self.column_count:
            raise ValueError("column_support_ratios must match column_count.")
        if any(
            row < 0
            or column < 0
            or row >= self.row_count
            or column >= self.column_count
            for row, column in self.missing_slot_coordinates
        ):
            raise ValueError("missing_slot_coordinates must lie inside the lattice.")
        if len(self.missing_slot_coordinates) != len(
            set(self.missing_slot_coordinates)
        ):
            raise ValueError("missing_slot_coordinates cannot contain duplicates.")
        if any(
            count < 0
            for count in (*self.row_component_counts, *self.column_component_counts)
        ):
            raise ValueError("Row and column component counts must be non-negative.")
        finite_non_negative = (
            *self.row_centers,
            *self.column_centers,
            self.horizontal_pitch,
            self.vertical_pitch,
            self.median_tile_width,
            self.median_tile_height,
        )
        if any(not isfinite(value) or value < 0.0 for value in finite_non_negative):
            raise ValueError("Cats tile-grid geometry diagnostics must be finite.")
        unit_scores = (
            self.horizontal_pitch_cv,
            self.vertical_pitch_cv,
            self.tile_width_cv,
            self.tile_height_cv,
            self.occupancy_ratio,
            *self.row_support_ratios,
            *self.column_support_ratios,
            self.minimum_row_support_ratio,
            self.minimum_column_support_ratio,
            self.mean_slot_residual,
            self.grid_score,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_scores):
            raise ValueError("Cats tile-grid scores must be finite within 0..1.")


@dataclass(frozen=True, slots=True)
class CatsTileGridDetection:
    """Pair fitted Cats board/grid geometry with its component diagnostics."""

    board: BoardDetection
    grid: GridDetection
    diagnostics: CatsTileGridDiagnostics


class CatsTileGridDetectionError(RuntimeError):
    """Report fail-closed tile-grid fitting with complete primitive diagnostics."""

    def __init__(self, message: str, diagnostics: CatsTileGridDiagnostics) -> None:
        """Retain structured diagnostics alongside one actionable message."""

        super().__init__(message)
        self.diagnostics = diagnostics


class CatsTileGridDetector(ABC):
    """Define Cats board and grid detection directly from colored tile lattices."""

    @abstractmethod
    def detect(self, screenshot: Screenshot) -> CatsTileGridDetection:
        """Return fitted board/grid geometry or raise CatsTileGridDetectionError."""

        raise NotImplementedError
