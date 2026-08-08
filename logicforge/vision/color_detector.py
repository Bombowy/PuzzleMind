"""Puzzle-neutral contracts for cell-color classification and diagnostics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot

type LabColor = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ColorObservation:
    """Describe one classified cell in zero-based row-major coordinates.

    ``color_id`` is a logical equality class, not a human color name. The LAB
    tuple uses OpenCV's 8-bit LAB scale (each component is in ``[0, 255]``) and is
    retained only as puzzle-neutral diagnostic evidence.
    """

    row: int
    column: int
    color_id: str
    confidence: float
    representative_lab: LabColor

    def __post_init__(self) -> None:
        """Reject incomplete coordinates, identifiers, and numeric evidence."""

        if self.row < 0 or self.column < 0:
            raise ValueError("Color observation coordinates must be non-negative.")
        if not self.color_id.startswith("C") or not self.color_id[1:].isdigit():
            raise ValueError("Color identifiers must use the C0, C1, ... format.")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Color confidence must be finite within 0.0 and 1.0.")
        if len(self.representative_lab) != 3 or any(
            not isfinite(component) or not 0.0 <= component <= 255.0
            for component in self.representative_lab
        ):
            raise ValueError("Representative LAB components must be within 0..255.")


@dataclass(frozen=True, slots=True)
class ColorDetectionDiagnostics:
    """Expose primitive sampling and clustering evidence without OpenCV objects.

    ``sample_pixel_counts`` records all four corner-patch pixels before corner-level
    outlier rejection.
    """

    rows: int
    columns: int
    cluster_distance_threshold: float
    sample_pixel_counts: tuple[int, ...]
    within_cell_spreads: tuple[float, ...]
    cluster_centers_lab: tuple[LabColor, ...]
    minimum_intercluster_distance: float | None
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate primitive measurements while permitting partial failure data."""

        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("Color diagnostics dimensions must be positive.")
        if not isfinite(self.cluster_distance_threshold) or (
            self.cluster_distance_threshold <= 0.0
        ):
            raise ValueError("Diagnostic cluster threshold must be positive.")
        if any(pixel_count < 0 for pixel_count in self.sample_pixel_counts):
            raise ValueError("Diagnostic sample pixel counts cannot be negative.")
        if any(
            not isfinite(spread) or spread < 0.0 for spread in self.within_cell_spreads
        ):
            raise ValueError("Diagnostic within-cell spreads must be non-negative.")
        if any(
            len(center) != 3
            or any(
                not isfinite(component) or not 0.0 <= component <= 255.0
                for component in center
            )
            for center in self.cluster_centers_lab
        ):
            raise ValueError("Diagnostic LAB centers must contain valid components.")
        if self.minimum_intercluster_distance is not None and (
            not isfinite(self.minimum_intercluster_distance)
            or self.minimum_intercluster_distance < 0.0
        ):
            raise ValueError(
                "Diagnostic minimum intercluster distance must be non-negative."
            )


@dataclass(frozen=True, slots=True)
class ColorDetectionResult:
    """Return complete row-major color equality classes for a detected grid.

    ``color_matrix`` has exactly ``rows`` rows and ``columns`` entries per row.
    It repeats the logical identifiers from ``observations`` so consumers receive
    a direct immutable board-shaped input without reconstructing geometry.
    """

    observations: tuple[ColorObservation, ...]
    color_count: int
    color_matrix: tuple[tuple[str, ...], ...]
    mean_confidence: float
    diagnostics: ColorDetectionDiagnostics

    def __post_init__(self) -> None:
        """Enforce completeness, row-major ordering, and consistent class IDs."""

        rows = self.diagnostics.rows
        columns = self.diagnostics.columns
        if rows <= 0 or columns <= 0:
            raise ValueError("Color result rows and columns must be positive.")
        if len(self.observations) != rows * columns:
            raise ValueError("Color observations must cover every grid cell.")
        for index, observation in enumerate(self.observations):
            expected_row, expected_column = divmod(index, columns)
            if (observation.row, observation.column) != (
                expected_row,
                expected_column,
            ):
                raise ValueError("Color observations must use stable row-major order.")
        if len(self.color_matrix) != rows or any(
            len(row) != columns for row in self.color_matrix
        ):
            raise ValueError("Color matrix dimensions must match diagnostics.")
        flattened_matrix = tuple(
            color_id for row in self.color_matrix for color_id in row
        )
        observed_ids = tuple(observation.color_id for observation in self.observations)
        if flattened_matrix != observed_ids:
            raise ValueError("Color matrix must exactly mirror observations.")
        unique_ids = set(observed_ids)
        expected_ids = {f"C{index}" for index in range(self.color_count)}
        if self.color_count <= 0 or unique_ids != expected_ids:
            raise ValueError("Color classes must be contiguous identifiers C0..Cn.")
        if len(self.diagnostics.sample_pixel_counts) != len(self.observations):
            raise ValueError("Successful diagnostics must cover every cell sample.")
        if len(self.diagnostics.within_cell_spreads) != len(self.observations):
            raise ValueError("Successful diagnostics must include every cell spread.")
        if len(self.diagnostics.cluster_centers_lab) != self.color_count:
            raise ValueError("Diagnostic center count must match color_count.")
        if self.diagnostics.rejection_reasons:
            raise ValueError("A successful color result cannot contain rejections.")
        if not isfinite(self.mean_confidence) or not (
            0.0 <= self.mean_confidence <= 1.0
        ):
            raise ValueError("Mean color confidence must be within 0.0 and 1.0.")


class ColorDetectionError(RuntimeError):
    """Report fail-closed sampling or classification with primitive diagnostics."""

    def __init__(self, message: str, diagnostics: ColorDetectionDiagnostics) -> None:
        """Retain structured context while exposing an actionable message."""

        super().__init__(message)
        self.diagnostics = diagnostics


class ColorDetector(ABC):
    """Define the port that classifies complete grid cells by color similarity."""

    @abstractmethod
    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
    ) -> ColorDetectionResult:
        """Return complete logical color classes or raise ``ColorDetectionError``."""

        raise NotImplementedError
