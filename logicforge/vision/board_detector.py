"""Boundary and diagnostic models for locating a puzzle board in a screenshot."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite

from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class BoardDetection:
    """Describe a rectangular board hypothesis in screenshot pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass(frozen=True, slots=True)
class BoardCandidateDiagnostic:
    """Record puzzle-neutral measurements and filtering decisions for one rectangle.

    These values support deterministic debugging without exposing contours, OpenCV
    matrices, or backend-specific objects outside the infrastructure layer.
    """

    x: int
    y: int
    width: int
    height: int
    relative_area: float
    aspect_ratio: float
    rectangularity: float
    edge_density: float
    location_score: float
    geometry_score: float
    horizontal_grid_line_positions: tuple[float, ...]
    vertical_grid_line_positions: tuple[float, ...]
    horizontal_grid_line_count: int
    vertical_grid_line_count: int
    estimated_rows: int
    estimated_columns: int
    horizontal_spacing_coefficient_of_variation: float
    vertical_spacing_coefficient_of_variation: float
    horizontal_spacing_regularity: float
    vertical_spacing_regularity: float
    horizontal_line_coverage: float
    vertical_line_coverage: float
    grid_evidence_score: float
    confidence: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BoardEnvelopeRefinementDiagnostic:
    """Describe one primitive, backend-neutral grid-envelope extension attempt."""

    seed_x: int
    seed_y: int
    seed_width: int
    seed_height: int
    refined_x: int
    refined_y: int
    refined_width: int
    refined_height: int
    direction: str
    added_pixels: int
    seed_rows: int
    seed_columns: int
    refined_rows: int
    refined_columns: int
    old_border_match_score: float
    separator_continuation_score: float
    supported_separator_fraction: float
    spacing_score: float
    refined_grid_score: float
    refinement_score: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Keep public refinement diagnostics finite, bounded, and primitive."""

        if self.direction not in {"left", "right", "top", "bottom"}:
            raise ValueError("Board-envelope refinement direction is invalid.")
        scores = (
            self.old_border_match_score,
            self.separator_continuation_score,
            self.supported_separator_fraction,
            self.spacing_score,
            self.refined_grid_score,
            self.refinement_score,
        )
        if any(not isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError(
                "Board-envelope refinement scores must be finite within 0.0 and 1.0."
            )


@dataclass(frozen=True, slots=True)
class BoardDetectionDiagnostics:
    """Summarize contour seeds, envelope refinement, and final selection."""

    contour_count: int
    candidates: tuple[BoardCandidateDiagnostic, ...]
    selected_candidate: BoardCandidateDiagnostic | None
    competitive_candidate_count: int
    envelope_refinements: tuple[BoardEnvelopeRefinementDiagnostic, ...] = ()
    selected_refinement: BoardEnvelopeRefinementDiagnostic | None = None


@dataclass(frozen=True, slots=True)
class BoardDetectionAnalysis:
    """Pair the selected public result with diagnostics used by debug tooling."""

    detection: BoardDetection
    diagnostics: BoardDetectionDiagnostics


class BoardDetectionError(RuntimeError):
    """Report that no candidate met reliability thresholds with useful diagnostics."""

    def __init__(self, message: str, diagnostics: BoardDetectionDiagnostics) -> None:
        """Retain structured diagnostics while exposing an actionable error message."""

        super().__init__(message)
        self.diagnostics = diagnostics


class BoardDetector(ABC):
    """Define the application-facing port for puzzle-board localization.

    Implementations inspect an immutable screenshot and return only the selected
    rectangle. Backend diagnostics may be exposed through additional adapter APIs.
    """

    @abstractmethod
    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Locate a reliable board or raise ``BoardDetectionError``."""

        raise NotImplementedError
