"""Boundary and diagnostic models for locating a puzzle board in a screenshot."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    confidence: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BoardDetectionDiagnostics:
    """Summarize contour evaluation and ambiguity for one detector invocation."""

    contour_count: int
    candidates: tuple[BoardCandidateDiagnostic, ...]
    selected_candidate: BoardCandidateDiagnostic | None
    competitive_candidate_count: int


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
