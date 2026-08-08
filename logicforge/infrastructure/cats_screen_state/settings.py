"""Validated settings for the OpenCV Cats screen-state adapter."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import cast


@dataclass(frozen=True, slots=True)
class CatsScreenStateDetectionSettings:
    """Configure scale-relative viewport and transition-overlay heuristics."""

    viewport_minimum_height_ratio: float = 0.88
    viewport_minimum_aspect_ratio: float = 0.48
    viewport_preferred_aspect_ratio: float = 9.0 / 16.0
    viewport_maximum_aspect_ratio: float = 0.66
    viewport_content_top_search_ratio: float = 0.10
    viewport_content_top_minimum_ratio: float = 0.01
    viewport_content_top_maximum_ratio: float = 0.08
    viewport_content_top_fallback_ratio: float = 0.032
    viewport_minimum_content_top_boundary_score: float = 0.12
    viewport_boundary_probe_ratio: float = 0.006
    viewport_boundary_smoothing_ratio: float = 0.006
    viewport_minimum_boundary_score: float = 0.08
    viewport_minimum_internal_activity: float = 0.12
    viewport_minimum_score: float = 0.54

    # Warm CTA button: red/orange low-hue range plus HSV red wrap near 179.
    level_warm_cta_hue_minimum: int = 0
    level_warm_cta_hue_maximum: int = 28
    level_warm_cta_red_wrap_hue_minimum: int = 170
    level_warm_cta_saturation_minimum: int = 145
    level_warm_cta_value_minimum: int = 120
    level_region_start_y_ratio: float = 0.60
    level_button_minimum_width_ratio: float = 0.55
    level_button_maximum_width_ratio: float = 0.92
    level_button_minimum_height_ratio: float = 0.055
    level_button_maximum_height_ratio: float = 0.15
    level_button_minimum_center_y_ratio: float = 0.78
    level_button_maximum_center_y_ratio: float = 0.98
    level_button_minimum_aspect_ratio: float = 3.8
    level_button_maximum_aspect_ratio: float = 10.5
    level_button_minimum_area_ratio: float = 0.03
    level_button_maximum_area_ratio: float = 0.14
    level_button_minimum_warm_fill_ratio: float = 0.48
    level_button_minimum_rectangularity: float = 0.50
    level_button_acceptance_score: float = 0.60
    level_morphology_kernel_ratio: float = 0.009
    level_morphology_iterations: int = 1

    ranking_brightness_minimum: int = 170
    ranking_saturation_maximum: int = 105
    ranking_warm_hue_maximum: int = 35
    ranking_warm_saturation_minimum: int = 20
    ranking_warm_saturation_maximum: int = 190
    ranking_warm_brightness_minimum: int = 160
    ranking_region_left_ratio: float = 0.03
    ranking_region_right_ratio: float = 0.97
    ranking_region_top_ratio: float = 0.20
    ranking_region_bottom_ratio: float = 0.80
    ranking_card_minimum_width_ratio: float = 0.55
    ranking_card_maximum_width_ratio: float = 0.95
    ranking_card_minimum_height_ratio: float = 0.05
    ranking_card_maximum_height_ratio: float = 0.17
    ranking_card_minimum_area_ratio: float = 0.025
    ranking_card_maximum_area_ratio: float = 0.15
    ranking_card_minimum_aspect_ratio: float = 3.0
    ranking_card_maximum_aspect_ratio: float = 13.0
    ranking_card_minimum_fill_ratio: float = 0.42
    ranking_card_minimum_contrast_score: float = 0.18
    ranking_maximum_width_variation: float = 0.18
    ranking_maximum_edge_alignment_ratio: float = 0.055
    ranking_minimum_gap_ratio: float = 0.006
    ranking_maximum_gap_ratio: float = 0.14
    ranking_maximum_gap_coefficient_of_variation: float = 0.55
    ranking_minimum_card_count: int = 2
    ranking_maximum_card_count: int = 3
    ranking_acceptance_score: float = 0.64
    ranking_morphology_horizontal_kernel_ratio: float = 0.055
    ranking_morphology_vertical_kernel_ratio: float = 0.003
    ranking_action_margin_ratio: float = 0.035
    ranking_action_minimum_y_ratio: float = 0.80
    ranking_action_maximum_y_ratio: float = 0.93

    def __post_init__(self) -> None:
        """Reject invalid colors, ratios, scores, ranges, and divisor settings."""

        if not (
            0
            <= self.level_warm_cta_hue_minimum
            < self.level_warm_cta_hue_maximum
            <= 179
        ):
            raise ValueError(
                "Warm CTA low-hue thresholds must satisfy 0 <= min < max <= 179."
            )
        if not (
            self.level_warm_cta_hue_maximum
            < self.level_warm_cta_red_wrap_hue_minimum
            <= 179
        ):
            raise ValueError(
                "level_warm_cta_red_wrap_hue_minimum must be above the low-hue "
                "CTA range and within 0..179."
            )
        if not 0 <= self.ranking_warm_hue_maximum <= 179:
            raise ValueError("ranking_warm_hue_maximum must be within 0 and 179.")
        byte_fields = {
            "level_warm_cta_saturation_minimum": (
                self.level_warm_cta_saturation_minimum
            ),
            "level_warm_cta_value_minimum": self.level_warm_cta_value_minimum,
            "ranking_brightness_minimum": self.ranking_brightness_minimum,
            "ranking_saturation_maximum": self.ranking_saturation_maximum,
            "ranking_warm_saturation_minimum": (self.ranking_warm_saturation_minimum),
            "ranking_warm_saturation_maximum": (self.ranking_warm_saturation_maximum),
            "ranking_warm_brightness_minimum": (self.ranking_warm_brightness_minimum),
        }
        for name, byte_value in byte_fields.items():
            if not 0 <= byte_value <= 255:
                raise ValueError(f"{name} must be within 0 and 255.")
        if self.ranking_warm_saturation_minimum >= self.ranking_warm_saturation_maximum:
            raise ValueError("Warm saturation thresholds must be ordered.")

        unit_fields = {
            field.name: cast(float, getattr(self, field.name))
            for field in fields(self)
            if (field.name.endswith("_ratio") and "aspect_ratio" not in field.name)
            or field.name.endswith("_score")
        }
        for name, unit_value in unit_fields.items():
            if not isfinite(unit_value) or not 0.0 <= unit_value <= 1.0:
                raise ValueError(f"{name} must be finite within 0.0 and 1.0.")

        additional_unit_fields = {
            "ranking_maximum_width_variation": self.ranking_maximum_width_variation,
            "ranking_maximum_gap_coefficient_of_variation": (
                self.ranking_maximum_gap_coefficient_of_variation
            ),
        }
        for name, tolerance_value in additional_unit_fields.items():
            if not isfinite(tolerance_value) or not 0.0 <= tolerance_value <= 1.0:
                raise ValueError(f"{name} must be finite within 0.0 and 1.0.")

        ordered_unit_ranges = (
            (
                self.viewport_content_top_minimum_ratio,
                self.viewport_content_top_maximum_ratio,
                "viewport content-top search bounds",
            ),
            (
                self.level_button_minimum_width_ratio,
                self.level_button_maximum_width_ratio,
                "level button width",
            ),
            (
                self.level_button_minimum_height_ratio,
                self.level_button_maximum_height_ratio,
                "level button height",
            ),
            (
                self.level_button_minimum_center_y_ratio,
                self.level_button_maximum_center_y_ratio,
                "level button center y",
            ),
            (
                self.level_button_minimum_area_ratio,
                self.level_button_maximum_area_ratio,
                "level button area",
            ),
            (
                self.ranking_region_left_ratio,
                self.ranking_region_right_ratio,
                "ranking horizontal region",
            ),
            (
                self.ranking_region_top_ratio,
                self.ranking_region_bottom_ratio,
                "ranking vertical region",
            ),
            (
                self.ranking_card_minimum_width_ratio,
                self.ranking_card_maximum_width_ratio,
                "ranking card width",
            ),
            (
                self.ranking_card_minimum_height_ratio,
                self.ranking_card_maximum_height_ratio,
                "ranking card height",
            ),
            (
                self.ranking_card_minimum_area_ratio,
                self.ranking_card_maximum_area_ratio,
                "ranking card area",
            ),
            (
                self.ranking_minimum_gap_ratio,
                self.ranking_maximum_gap_ratio,
                "ranking gap",
            ),
            (
                self.ranking_action_minimum_y_ratio,
                self.ranking_action_maximum_y_ratio,
                "ranking action y",
            ),
        )
        for minimum, maximum, name in ordered_unit_ranges:
            if minimum >= maximum:
                raise ValueError(f"{name} thresholds must satisfy minimum < maximum.")

        if not (
            isfinite(self.viewport_minimum_aspect_ratio)
            and isfinite(self.viewport_preferred_aspect_ratio)
            and isfinite(self.viewport_maximum_aspect_ratio)
            and 0.0
            < self.viewport_minimum_aspect_ratio
            < self.viewport_preferred_aspect_ratio
            < self.viewport_maximum_aspect_ratio
        ):
            raise ValueError(
                "Viewport aspect ratios must satisfy 0 < minimum < preferred < maximum."
            )
        aspect_ranges = (
            (
                self.level_button_minimum_aspect_ratio,
                self.level_button_maximum_aspect_ratio,
                "Level button",
            ),
            (
                self.ranking_card_minimum_aspect_ratio,
                self.ranking_card_maximum_aspect_ratio,
                "Ranking card",
            ),
        )
        for minimum, maximum, name in aspect_ranges:
            if not (
                isfinite(minimum) and isfinite(maximum) and 0.0 < minimum < maximum
            ):
                raise ValueError(
                    f"{name} aspect thresholds must be positive and ordered."
                )

        positive_ratios = {
            "viewport_minimum_height_ratio": self.viewport_minimum_height_ratio,
            "viewport_content_top_search_ratio": (
                self.viewport_content_top_search_ratio
            ),
            "viewport_boundary_probe_ratio": self.viewport_boundary_probe_ratio,
            "viewport_boundary_smoothing_ratio": (
                self.viewport_boundary_smoothing_ratio
            ),
            "level_morphology_kernel_ratio": self.level_morphology_kernel_ratio,
            "ranking_morphology_horizontal_kernel_ratio": (
                self.ranking_morphology_horizontal_kernel_ratio
            ),
            "ranking_morphology_vertical_kernel_ratio": (
                self.ranking_morphology_vertical_kernel_ratio
            ),
            "ranking_maximum_edge_alignment_ratio": (
                self.ranking_maximum_edge_alignment_ratio
            ),
            "ranking_maximum_gap_coefficient_of_variation": (
                self.ranking_maximum_gap_coefficient_of_variation
            ),
        }
        for name, positive_ratio in positive_ratios.items():
            if positive_ratio <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.viewport_content_top_maximum_ratio > (
            self.viewport_content_top_search_ratio
        ):
            raise ValueError(
                "Content-top maximum must not exceed its search-region ratio."
            )
        if self.level_morphology_iterations < 1:
            raise ValueError("Level morphology iterations must be positive.")
        if self.ranking_minimum_card_count < 2:
            raise ValueError("Ranking requires at least two cards.")
        if self.ranking_maximum_card_count < self.ranking_minimum_card_count:
            raise ValueError("Ranking maximum card count must not be below minimum.")
