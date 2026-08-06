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
    minimum_horizontal_grid_line_count: int = 4
    minimum_vertical_grid_line_count: int = 4
    minimum_estimated_rows: int = 3
    minimum_estimated_columns: int = 3
    maximum_horizontal_spacing_coefficient_of_variation: float = 0.22
    maximum_vertical_spacing_coefficient_of_variation: float = 0.22
    minimum_horizontal_line_coverage: float = 0.50
    minimum_vertical_line_coverage: float = 0.50
    minimum_grid_evidence_score: float = 0.65
    grid_line_cluster_distance_relative: float = 0.02
    grid_border_line_exclusion_tolerance: float = 0.05
    horizontal_line_kernel_relative_length: float = 0.06
    vertical_line_kernel_relative_length: float = 0.06
    minimum_grid_line_response: float = 0.30
    grid_adaptive_block_relative_size: float = 0.061
    grid_adaptive_constant: float = 5.0
    geometry_confidence_weight: float = 0.40
    grid_confidence_weight: float = 0.60
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
            "maximum_horizontal_spacing_coefficient_of_variation": (
                self.maximum_horizontal_spacing_coefficient_of_variation
            ),
            "maximum_vertical_spacing_coefficient_of_variation": (
                self.maximum_vertical_spacing_coefficient_of_variation
            ),
            "minimum_horizontal_line_coverage": (self.minimum_horizontal_line_coverage),
            "minimum_vertical_line_coverage": self.minimum_vertical_line_coverage,
            "minimum_grid_evidence_score": self.minimum_grid_evidence_score,
            "grid_line_cluster_distance_relative": (
                self.grid_line_cluster_distance_relative
            ),
            "grid_border_line_exclusion_tolerance": (
                self.grid_border_line_exclusion_tolerance
            ),
            "horizontal_line_kernel_relative_length": (
                self.horizontal_line_kernel_relative_length
            ),
            "vertical_line_kernel_relative_length": (
                self.vertical_line_kernel_relative_length
            ),
            "minimum_grid_line_response": self.minimum_grid_line_response,
            "grid_adaptive_block_relative_size": (
                self.grid_adaptive_block_relative_size
            ),
            "geometry_confidence_weight": self.geometry_confidence_weight,
            "grid_confidence_weight": self.grid_confidence_weight,
            "minimum_confidence": self.minimum_confidence,
            "ambiguity_score_delta": self.ambiguity_score_delta,
            "duplicate_iou_threshold": self.duplicate_iou_threshold,
        }
        for field_name, value in unit_interval_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within 0.0 and 1.0.")

        strictly_positive_unit_fields = {
            "maximum_horizontal_spacing_coefficient_of_variation": (
                self.maximum_horizontal_spacing_coefficient_of_variation
            ),
            "maximum_vertical_spacing_coefficient_of_variation": (
                self.maximum_vertical_spacing_coefficient_of_variation
            ),
            "minimum_horizontal_line_coverage": (self.minimum_horizontal_line_coverage),
            "minimum_vertical_line_coverage": self.minimum_vertical_line_coverage,
            "minimum_grid_evidence_score": self.minimum_grid_evidence_score,
            "grid_line_cluster_distance_relative": (
                self.grid_line_cluster_distance_relative
            ),
            "grid_border_line_exclusion_tolerance": (
                self.grid_border_line_exclusion_tolerance
            ),
            "horizontal_line_kernel_relative_length": (
                self.horizontal_line_kernel_relative_length
            ),
            "vertical_line_kernel_relative_length": (
                self.vertical_line_kernel_relative_length
            ),
            "minimum_grid_line_response": self.minimum_grid_line_response,
            "grid_adaptive_block_relative_size": (
                self.grid_adaptive_block_relative_size
            ),
            "geometry_confidence_weight": self.geometry_confidence_weight,
            "grid_confidence_weight": self.grid_confidence_weight,
        }
        for field_name, value in strictly_positive_unit_fields.items():
            if value <= 0.0:
                raise ValueError(f"{field_name} must be greater than 0.0.")

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
        positive_count_fields = {
            "minimum_horizontal_grid_line_count": (
                self.minimum_horizontal_grid_line_count
            ),
            "minimum_vertical_grid_line_count": self.minimum_vertical_grid_line_count,
            "minimum_estimated_rows": self.minimum_estimated_rows,
            "minimum_estimated_columns": self.minimum_estimated_columns,
        }
        for field_name, value in positive_count_fields.items():
            if value < 1:
                raise ValueError(f"{field_name} must be positive.")
        if self.minimum_horizontal_grid_line_count < 4:
            raise ValueError("minimum_horizontal_grid_line_count must be at least 4.")
        if self.minimum_vertical_grid_line_count < 4:
            raise ValueError("minimum_vertical_grid_line_count must be at least 4.")
        if self.minimum_estimated_rows < 3:
            raise ValueError("minimum_estimated_rows must be at least 3.")
        if self.minimum_estimated_columns < 3:
            raise ValueError("minimum_estimated_columns must be at least 3.")
        if self.grid_adaptive_constant < 0.0:
            raise ValueError("grid_adaptive_constant must not be negative.")
        if self.grid_line_cluster_distance_relative >= 0.5:
            raise ValueError("grid_line_cluster_distance_relative must be below 0.5.")
        if self.grid_border_line_exclusion_tolerance >= 0.5:
            raise ValueError("grid_border_line_exclusion_tolerance must be below 0.5.")
        if (
            not abs(self.geometry_confidence_weight + self.grid_confidence_weight - 1.0)
            < 1e-9
        ):
            raise ValueError("Confidence weights must sum to 1.0.")
        if self.grid_confidence_weight < self.geometry_confidence_weight:
            raise ValueError(
                "grid_confidence_weight must be at least geometry_confidence_weight."
            )


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
