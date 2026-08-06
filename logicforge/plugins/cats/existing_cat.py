"""Backend-neutral contracts for Cats already present in detected grid cells."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from logicforge.vision.color_detector import ColorDetectionResult
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class CatsExistingCatObservation:
    """Identify one accepted existing cat with normalized confidence."""

    row: int
    column: int
    confidence: float

    def __post_init__(self) -> None:
        if self.row < 0 or self.column < 0:
            raise ValueError("Existing cat coordinates must be non-negative.")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Existing cat confidence must be finite within 0..1.")


@dataclass(frozen=True, slots=True)
class CatsExistingCatCellDiagnostic:
    """Expose primitive foreground and connected-component evidence for one cell."""

    row: int
    column: int
    roi_x: int
    roi_y: int
    roi_width: int
    roi_height: int
    foreground_ratio: float
    largest_component_ratio: float
    component_width_ratio: float
    component_height_ratio: float
    center_offset_ratio: float
    score: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if min(self.row, self.column, self.roi_x, self.roi_y) < 0:
            raise ValueError(
                "Existing-cat diagnostic coordinates must be non-negative."
            )
        if self.roi_width <= 0 or self.roi_height <= 0:
            raise ValueError("Existing-cat diagnostic ROI must be positive.")
        ratios = (
            self.foreground_ratio,
            self.largest_component_ratio,
            self.component_width_ratio,
            self.component_height_ratio,
            self.center_offset_ratio,
            self.score,
        )
        if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError(
                "Existing-cat ratios and score must be finite within 0..1."
            )
        if self.accepted and self.rejection_reasons:
            raise ValueError("An accepted existing-cat cell cannot have rejections.")


@dataclass(frozen=True, slots=True)
class CatsExistingCatDiagnostics:
    """Retain complete row-major primitive cell evidence and failure context."""

    cells: tuple[CatsExistingCatCellDiagnostic, ...]
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CatsExistingCatDetection:
    """Return zero or more validated existing cats and complete diagnostics."""

    cats: tuple[CatsExistingCatObservation, ...]
    diagnostics: CatsExistingCatDiagnostics

    def __post_init__(self) -> None:
        coordinates = tuple((cat.row, cat.column) for cat in self.cats)
        if coordinates != tuple(sorted(coordinates)):
            raise ValueError("Existing cats must use deterministic row-major order.")
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("Existing cat coordinates cannot contain duplicates.")
        if self.diagnostics.rejection_reasons:
            raise ValueError(
                "Successful existing-cat detection cannot have rejections."
            )


class CatsExistingCatDetectionError(RuntimeError):
    """Fail closed on malformed inputs or contradictory accepted cat evidence."""

    def __init__(self, message: str, diagnostics: CatsExistingCatDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class CatsExistingCatDetector(ABC):
    """Detect existing Cats only inside public grid CellBounds."""

    @abstractmethod
    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        colors: ColorDetectionResult,
    ) -> CatsExistingCatDetection:
        """Return validated cat occupancy or raise a typed fail-closed error."""

        raise NotImplementedError
