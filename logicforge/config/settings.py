"""Dependency-free settings records for framework composition."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BoardDetectionSettings:
    """Configure scale-independent classical board localization heuristics.

    Every geometric threshold is relative to screenshot dimensions so the detector
    remains usable across emulator positions, window sizes, and puzzle levels.
    """

    minimum_relative_area: float = 0.06
    maximum_relative_area: float = 0.72
    preferred_relative_area: float = 0.28
    minimum_aspect_ratio: float = 0.55
    maximum_aspect_ratio: float = 1.80
    minimum_rectangularity: float = 0.68
    border_exclusion: float = 0.02
    top_content_exclusion: float = 0.05
    right_toolbar_exclusion: float = 0.035
    bottom_content_exclusion: float = 0.02
    gaussian_blur_kernel_size: int = 5
    canny_lower_threshold: int = 50
    canny_upper_threshold: int = 160
    morphology_kernel_relative_size: float = 0.012
    morphology_iterations: int = 2
    edge_envelope_kernel_relative_size: float = 0.007
    edge_envelope_iterations: int = 1
    minimum_edge_density: float = 0.015
    preferred_edge_density: float = 0.08
    maximum_edge_density: float = 0.28
    expected_center_x: float = 0.55
    expected_center_y: float = 0.55
    polygon_epsilon_ratio: float = 0.02
    minimum_confidence: float = 0.48
    ambiguity_score_delta: float = 0.03
    duplicate_iou_threshold: float = 0.90
    debug_rejected_candidates: bool = False

    def __post_init__(self) -> None:
        """Reject contradictory or unsafe thresholds at composition time."""

        unit_interval_fields = {
            "minimum_relative_area": self.minimum_relative_area,
            "maximum_relative_area": self.maximum_relative_area,
            "preferred_relative_area": self.preferred_relative_area,
            "minimum_rectangularity": self.minimum_rectangularity,
            "border_exclusion": self.border_exclusion,
            "top_content_exclusion": self.top_content_exclusion,
            "right_toolbar_exclusion": self.right_toolbar_exclusion,
            "bottom_content_exclusion": self.bottom_content_exclusion,
            "morphology_kernel_relative_size": self.morphology_kernel_relative_size,
            "edge_envelope_kernel_relative_size": (
                self.edge_envelope_kernel_relative_size
            ),
            "minimum_edge_density": self.minimum_edge_density,
            "preferred_edge_density": self.preferred_edge_density,
            "maximum_edge_density": self.maximum_edge_density,
            "expected_center_x": self.expected_center_x,
            "expected_center_y": self.expected_center_y,
            "polygon_epsilon_ratio": self.polygon_epsilon_ratio,
            "minimum_confidence": self.minimum_confidence,
            "ambiguity_score_delta": self.ambiguity_score_delta,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
        }
        for field_name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within 0.0 and 1.0.")

        if not (
            self.minimum_relative_area
            < self.preferred_relative_area
            < self.maximum_relative_area
        ):
            raise ValueError(
                "Relative area thresholds must satisfy minimum < preferred < maximum."
            )
        if not 0.0 < self.minimum_aspect_ratio < 1.0:
            raise ValueError("minimum_aspect_ratio must be greater than 0 and below 1.")
        if self.maximum_aspect_ratio <= 1.0:
            raise ValueError("maximum_aspect_ratio must be greater than 1.")
        if not (
            self.minimum_edge_density
            < self.preferred_edge_density
            < self.maximum_edge_density
        ):
            raise ValueError(
                "Edge density thresholds must satisfy minimum < preferred < maximum."
            )
        if self.gaussian_blur_kernel_size < 1 or not (
            self.gaussian_blur_kernel_size % 2
        ):
            raise ValueError("gaussian_blur_kernel_size must be a positive odd value.")
        if not 0 <= self.canny_lower_threshold < self.canny_upper_threshold <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= lower < upper <= 255.")
        if self.morphology_iterations < 1:
            raise ValueError("morphology_iterations must be positive.")
        if self.edge_envelope_iterations < 1:
            raise ValueError("edge_envelope_iterations must be positive.")


@dataclass(frozen=True, slots=True)
class VisionSettings:
    """Hold typed puzzle-neutral configuration for vision adapters."""

    debug_artifacts_directory: Path = Path("artifacts/vision")
    board_detection: BoardDetectionSettings = field(
        default_factory=BoardDetectionSettings
    )


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Hold safety limits for future deterministic solver orchestration.

    TODO: Add iteration and timeout limits in v0.4 together with explicit failure
    results, cancellation behavior, and reproducibility guarantees.
    """

    explanations_enabled: bool = True


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Describe logging policy without configuring process-global state.

    TODO: Add structured output, redaction, and correlation-id options when the
    application composition layer introduces an operational logging adapter.
    """

    level: str = "INFO"


@dataclass(frozen=True, slots=True)
class LogicForgeSettings:
    """Aggregate immutable settings passed through the composition root.

    Nested settings keep subsystems independent and make future configuration
    sources replaceable without making domain code aware of environment variables.

    TODO: Add an external settings loader after precedence rules for files,
    environment variables, and command-line options are documented and tested.
    """

    vision: VisionSettings = field(default_factory=VisionSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
