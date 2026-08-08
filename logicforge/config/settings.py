"""Typed settings for active vision and Cats detector adapters."""

from dataclasses import dataclass
from math import isfinite


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
    grid_missing_line_recovery_enabled: bool = True
    grid_weak_horizontal_line_kernel_relative_length: float = 0.03
    grid_weak_vertical_line_kernel_relative_length: float = 0.03
    grid_missing_line_minimum_gap_factor: float = 1.55
    grid_missing_line_maximum_gap_factor: float = 2.45
    grid_missing_line_search_half_width_fraction: float = 0.25
    grid_missing_line_minimum_weak_response: float = 0.10
    grid_missing_line_maximum_other_gap_deviation: float = 0.30
    grid_missing_line_minimum_cv_improvement: float = 0.05
    grid_missing_line_maximum_recovered_per_axis: int = 1
    grid_envelope_refinement_enabled: bool = True
    grid_envelope_maximum_added_cells_per_side: int = 1
    grid_envelope_minimum_seed_rows: int = 3
    grid_envelope_minimum_seed_columns: int = 3
    grid_envelope_minimum_added_size_ratio: float = 0.75
    grid_envelope_maximum_added_size_ratio: float = 1.25
    grid_envelope_separator_position_tolerance_ratio: float = 0.18
    grid_envelope_continuation_probe_thickness_ratio: float = 0.08
    grid_envelope_minimum_line_continuation_response: float = 0.12
    grid_envelope_minimum_supported_separator_fraction: float = 0.65
    grid_envelope_maximum_spacing_cv_increase: float = 0.03
    grid_envelope_maximum_grid_score_drop: float = 0.08
    grid_envelope_minimum_refinement_score: float = 0.63
    grid_envelope_ambiguity_delta: float = 0.03
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
            "grid_weak_horizontal_line_kernel_relative_length": (
                self.grid_weak_horizontal_line_kernel_relative_length
            ),
            "grid_weak_vertical_line_kernel_relative_length": (
                self.grid_weak_vertical_line_kernel_relative_length
            ),
            "grid_missing_line_search_half_width_fraction": (
                self.grid_missing_line_search_half_width_fraction
            ),
            "grid_missing_line_minimum_weak_response": (
                self.grid_missing_line_minimum_weak_response
            ),
            "grid_missing_line_maximum_other_gap_deviation": (
                self.grid_missing_line_maximum_other_gap_deviation
            ),
            "grid_missing_line_minimum_cv_improvement": (
                self.grid_missing_line_minimum_cv_improvement
            ),
            "grid_envelope_separator_position_tolerance_ratio": (
                self.grid_envelope_separator_position_tolerance_ratio
            ),
            "grid_envelope_continuation_probe_thickness_ratio": (
                self.grid_envelope_continuation_probe_thickness_ratio
            ),
            "grid_envelope_minimum_line_continuation_response": (
                self.grid_envelope_minimum_line_continuation_response
            ),
            "grid_envelope_minimum_supported_separator_fraction": (
                self.grid_envelope_minimum_supported_separator_fraction
            ),
            "grid_envelope_maximum_spacing_cv_increase": (
                self.grid_envelope_maximum_spacing_cv_increase
            ),
            "grid_envelope_maximum_grid_score_drop": (
                self.grid_envelope_maximum_grid_score_drop
            ),
            "grid_envelope_minimum_refinement_score": (
                self.grid_envelope_minimum_refinement_score
            ),
            "grid_envelope_ambiguity_delta": self.grid_envelope_ambiguity_delta,
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
            "grid_weak_horizontal_line_kernel_relative_length": (
                self.grid_weak_horizontal_line_kernel_relative_length
            ),
            "grid_weak_vertical_line_kernel_relative_length": (
                self.grid_weak_vertical_line_kernel_relative_length
            ),
            "grid_missing_line_search_half_width_fraction": (
                self.grid_missing_line_search_half_width_fraction
            ),
            "grid_envelope_separator_position_tolerance_ratio": (
                self.grid_envelope_separator_position_tolerance_ratio
            ),
            "grid_envelope_continuation_probe_thickness_ratio": (
                self.grid_envelope_continuation_probe_thickness_ratio
            ),
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
        gap_factors = (
            self.grid_missing_line_minimum_gap_factor,
            self.grid_missing_line_maximum_gap_factor,
        )
        if any(not isfinite(value) for value in gap_factors):
            raise ValueError("Grid missing-line gap factors must be finite.")
        if not 1.0 < gap_factors[0] < gap_factors[1] <= 4.0:
            raise ValueError(
                "Grid missing-line gap factors must satisfy "
                "1 < minimum < maximum <= 4."
            )
        if self.grid_missing_line_search_half_width_fraction > 0.5:
            raise ValueError(
                "grid_missing_line_search_half_width_fraction must not exceed 0.5."
            )
        if not isinstance(
            self.grid_missing_line_maximum_recovered_per_axis, int
        ) or isinstance(self.grid_missing_line_maximum_recovered_per_axis, bool):
            raise ValueError(
                "grid_missing_line_maximum_recovered_per_axis must be an integer."
            )
        if not 0 <= self.grid_missing_line_maximum_recovered_per_axis <= 1:
            raise ValueError(
                "grid_missing_line_maximum_recovered_per_axis must be 0 or 1."
            )
        added_size_ratios = (
            self.grid_envelope_minimum_added_size_ratio,
            self.grid_envelope_maximum_added_size_ratio,
        )
        if any(not isfinite(value) for value in added_size_ratios):
            raise ValueError("Grid-envelope added-size ratios must be finite.")
        if not 0.0 < added_size_ratios[0] < 1.0 < added_size_ratios[1]:
            raise ValueError(
                "Grid-envelope added-size ratios must satisfy "
                "0 < minimum < 1 < maximum."
            )
        if not isinstance(
            self.grid_envelope_maximum_added_cells_per_side, int
        ) or isinstance(self.grid_envelope_maximum_added_cells_per_side, bool):
            raise ValueError(
                "grid_envelope_maximum_added_cells_per_side must be an integer."
            )
        if not 0 <= self.grid_envelope_maximum_added_cells_per_side <= 1:
            raise ValueError(
                "grid_envelope_maximum_added_cells_per_side must be 0 or 1."
            )
        seed_counts = {
            "grid_envelope_minimum_seed_rows": self.grid_envelope_minimum_seed_rows,
            "grid_envelope_minimum_seed_columns": (
                self.grid_envelope_minimum_seed_columns
            ),
        }
        for field_name, value in seed_counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer.")
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
class CatsTileGridDetectionSettings:
    """Configure Cats tile-component and regular-lattice geometry fitting."""

    tile_minimum_hsv_saturation: int = 45
    tile_minimum_lab_chroma: float = 12.0
    tile_minimum_component_area_ratio: float = 0.00035
    tile_maximum_component_area_ratio: float = 0.035
    tile_minimum_aspect_ratio: float = 0.72
    tile_maximum_aspect_ratio: float = 1.38
    tile_minimum_fill_ratio: float = 0.72
    tile_size_family_tolerance_ratio: float = 0.18
    tile_center_cluster_tolerance_ratio: float = 0.32
    tile_pitch_cv_maximum: float = 0.12
    tile_size_cv_maximum: float = 0.16
    tile_slot_residual_ratio: float = 0.30
    tile_grid_minimum_rows: int = 4
    tile_grid_minimum_columns: int = 4
    tile_grid_minimum_occupancy_ratio: float = 0.90
    tile_grid_minimum_row_support_ratio: float = 0.75
    tile_grid_minimum_column_support_ratio: float = 0.75
    tile_grid_minimum_score: float = 0.78
    tile_mask_kernel_relative_size: float = 0.003

    def __post_init__(self) -> None:
        """Reject unsafe, non-finite, or contradictory tile-grid thresholds."""

        if not isinstance(self.tile_minimum_hsv_saturation, int) or isinstance(
            self.tile_minimum_hsv_saturation, bool
        ):
            raise ValueError("tile_minimum_hsv_saturation must be an integer.")
        if not 0 <= self.tile_minimum_hsv_saturation <= 255:
            raise ValueError("tile_minimum_hsv_saturation must be within 0..255.")
        unit_fields = {
            "tile_minimum_component_area_ratio": (
                self.tile_minimum_component_area_ratio
            ),
            "tile_maximum_component_area_ratio": (
                self.tile_maximum_component_area_ratio
            ),
            "tile_minimum_fill_ratio": self.tile_minimum_fill_ratio,
            "tile_size_family_tolerance_ratio": (self.tile_size_family_tolerance_ratio),
            "tile_center_cluster_tolerance_ratio": (
                self.tile_center_cluster_tolerance_ratio
            ),
            "tile_pitch_cv_maximum": self.tile_pitch_cv_maximum,
            "tile_size_cv_maximum": self.tile_size_cv_maximum,
            "tile_slot_residual_ratio": self.tile_slot_residual_ratio,
            "tile_grid_minimum_occupancy_ratio": (
                self.tile_grid_minimum_occupancy_ratio
            ),
            "tile_grid_minimum_row_support_ratio": (
                self.tile_grid_minimum_row_support_ratio
            ),
            "tile_grid_minimum_column_support_ratio": (
                self.tile_grid_minimum_column_support_ratio
            ),
            "tile_grid_minimum_score": self.tile_grid_minimum_score,
            "tile_mask_kernel_relative_size": self.tile_mask_kernel_relative_size,
        }
        for field_name, value in unit_fields.items():
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite within 0.0 and 1.0.")
        if not isfinite(self.tile_minimum_lab_chroma) or not (
            0.0 <= self.tile_minimum_lab_chroma <= 181.0
        ):
            raise ValueError("tile_minimum_lab_chroma must be finite within 0..181.")
        if not (
            0.0
            < self.tile_minimum_component_area_ratio
            < self.tile_maximum_component_area_ratio
            <= 1.0
        ):
            raise ValueError(
                "Tile component area ratios must satisfy 0 < minimum < maximum <= 1."
            )
        if not (
            isfinite(self.tile_minimum_aspect_ratio)
            and isfinite(self.tile_maximum_aspect_ratio)
            and 0.0 < self.tile_minimum_aspect_ratio < 1.0
            and self.tile_maximum_aspect_ratio > 1.0
            and self.tile_minimum_aspect_ratio < self.tile_maximum_aspect_ratio
        ):
            raise ValueError(
                "Tile aspect ratios must satisfy 0 < minimum < 1 < maximum."
            )
        positive_unit_fields = {
            "tile_minimum_fill_ratio": self.tile_minimum_fill_ratio,
            "tile_size_family_tolerance_ratio": (self.tile_size_family_tolerance_ratio),
            "tile_center_cluster_tolerance_ratio": (
                self.tile_center_cluster_tolerance_ratio
            ),
            "tile_pitch_cv_maximum": self.tile_pitch_cv_maximum,
            "tile_size_cv_maximum": self.tile_size_cv_maximum,
            "tile_slot_residual_ratio": self.tile_slot_residual_ratio,
            "tile_grid_minimum_occupancy_ratio": (
                self.tile_grid_minimum_occupancy_ratio
            ),
            "tile_grid_minimum_row_support_ratio": (
                self.tile_grid_minimum_row_support_ratio
            ),
            "tile_grid_minimum_column_support_ratio": (
                self.tile_grid_minimum_column_support_ratio
            ),
            "tile_grid_minimum_score": self.tile_grid_minimum_score,
            "tile_mask_kernel_relative_size": self.tile_mask_kernel_relative_size,
        }
        for field_name, value in positive_unit_fields.items():
            if value <= 0.0:
                raise ValueError(f"{field_name} must be greater than 0.0.")
        for field_name, value in {
            "tile_grid_minimum_rows": self.tile_grid_minimum_rows,
            "tile_grid_minimum_columns": self.tile_grid_minimum_columns,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{field_name} must be an integer of at least 2.")


@dataclass(frozen=True, slots=True)
class CatsExistingCatDetectionSettings:
    """Configure scale-relative Cats foreground occupancy detection per cell."""

    cat_roi_horizontal_inset_ratio: float = 0.08
    cat_roi_vertical_inset_ratio: float = 0.06
    cat_foreground_lab_distance_threshold: float = 32.0
    cat_mask_kernel_relative_size: float = 0.035
    cat_minimum_foreground_ratio: float = 0.26
    cat_minimum_largest_component_ratio: float = 0.24
    cat_minimum_component_width_ratio: float = 0.38
    cat_minimum_component_height_ratio: float = 0.38
    cat_maximum_center_offset_ratio: float = 0.18
    cat_minimum_score: float = 0.40

    def __post_init__(self) -> None:
        """Reject non-finite or contradictory occupancy thresholds."""

        inset_fields = {
            "cat_roi_horizontal_inset_ratio": self.cat_roi_horizontal_inset_ratio,
            "cat_roi_vertical_inset_ratio": self.cat_roi_vertical_inset_ratio,
        }
        for field_name, value in inset_fields.items():
            if not isfinite(value) or not 0.0 <= value < 0.5:
                raise ValueError(f"{field_name} must be finite within [0.0, 0.5).")
        unit_fields = {
            "cat_mask_kernel_relative_size": self.cat_mask_kernel_relative_size,
            "cat_minimum_foreground_ratio": self.cat_minimum_foreground_ratio,
            "cat_minimum_largest_component_ratio": (
                self.cat_minimum_largest_component_ratio
            ),
            "cat_minimum_component_width_ratio": (
                self.cat_minimum_component_width_ratio
            ),
            "cat_minimum_component_height_ratio": (
                self.cat_minimum_component_height_ratio
            ),
            "cat_maximum_center_offset_ratio": self.cat_maximum_center_offset_ratio,
            "cat_minimum_score": self.cat_minimum_score,
        }
        for field_name, value in unit_fields.items():
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{field_name} must be finite within (0.0, 1.0].")
        if not isfinite(self.cat_foreground_lab_distance_threshold) or not (
            0.0 < self.cat_foreground_lab_distance_threshold <= 441.7
        ):
            raise ValueError(
                "cat_foreground_lab_distance_threshold must be finite within "
                "(0.0, 441.7]."
            )


@dataclass(frozen=True, slots=True)
class GridExtractionSettings:
    """Configure only normalized-boundary to pixel-cell conversion constraints."""

    minimum_cell_width_pixels: int = 1
    minimum_cell_height_pixels: int = 1

    def __post_init__(self) -> None:
        """Reject extraction constraints that could permit empty cell geometry."""

        if self.minimum_cell_width_pixels < 1:
            raise ValueError("minimum_cell_width_pixels must be positive.")
        if self.minimum_cell_height_pixels < 1:
            raise ValueError("minimum_cell_height_pixels must be positive.")


@dataclass(frozen=True, slots=True)
class ColorDetectionSettings:
    """Configure robust cell sampling and deterministic LAB color clustering.

    LAB distances use OpenCV's 8-bit representation. The threshold therefore has
    implementation-calibrated units and does not claim to be a CIE Delta-E value.
    """

    corner_sample_patch_fraction: float = 0.12
    corner_sample_offset_fraction: float = 0.1
    corner_sample_minimum_consistent_patches: int = 3
    outlier_trim_fraction: float = 0.15
    cluster_distance_threshold: float = 18.0
    maximum_within_cell_spread: float = 24.0
    minimum_sample_pixels: int = 25
    homogeneity_confidence_weight: float = 0.70
    cluster_fit_confidence_weight: float = 0.30

    def __post_init__(self) -> None:
        """Reject settings that could create empty samples or invalid confidence."""

        if not isfinite(self.corner_sample_patch_fraction) or not (
            0.0 < self.corner_sample_patch_fraction < 0.5
        ):
            raise ValueError(
                "corner_sample_patch_fraction must be finite within (0.0, 0.5)."
            )
        if not isfinite(self.corner_sample_offset_fraction) or not (
            0.0 <= self.corner_sample_offset_fraction < 0.5
        ):
            raise ValueError(
                "corner_sample_offset_fraction must be finite within [0.0, 0.5)."
            )
        if self.corner_sample_offset_fraction + self.corner_sample_patch_fraction > 0.5:
            raise ValueError(
                "Corner sample offset plus patch size must not cross cell center."
            )
        if not isinstance(
            self.corner_sample_minimum_consistent_patches, int
        ) or isinstance(self.corner_sample_minimum_consistent_patches, bool):
            raise ValueError(
                "corner_sample_minimum_consistent_patches must be an integer."
            )
        if not 2 <= self.corner_sample_minimum_consistent_patches <= 4:
            raise ValueError(
                "corner_sample_minimum_consistent_patches must be within 2..4."
            )
        if not 0.0 <= self.outlier_trim_fraction < 0.5:
            raise ValueError("outlier_trim_fraction must be within [0.0, 0.5).")
        if self.cluster_distance_threshold <= 0.0:
            raise ValueError("cluster_distance_threshold must be positive.")
        if self.maximum_within_cell_spread <= 0.0:
            raise ValueError("maximum_within_cell_spread must be positive.")
        if self.minimum_sample_pixels < 1:
            raise ValueError("minimum_sample_pixels must be positive.")
        weights = (
            self.homogeneity_confidence_weight,
            self.cluster_fit_confidence_weight,
        )
        if any(weight < 0.0 or weight > 1.0 for weight in weights):
            raise ValueError("Color confidence weights must be within 0.0 and 1.0.")
        if not abs(sum(weights) - 1.0) < 1e-9:
            raise ValueError("Color confidence weights must sum to 1.0.")
