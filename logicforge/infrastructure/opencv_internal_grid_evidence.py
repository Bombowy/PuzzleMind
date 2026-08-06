"""Internal OpenCV regular-grid evidence used only to validate board candidates.

This module deliberately does not implement the public ``GridDetector`` port. It
measures whether a rectangular board candidate contains a credible axis-aligned
grid, but it does not expose cells or grid geometry to parsing code.
"""

from dataclasses import dataclass
from itertools import pairwise
from math import floor
from statistics import fmean, median, pstdev
from typing import Protocol, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings


def _clamp_unit(value: float) -> float:
    """Clamp a normalized grid metric into the inclusive unit interval."""

    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class InternalGridEvidence:
    """Carry primitive, backend-neutral measurements from one candidate ROI.

    Line positions are normalized to the candidate dimensions. The first and last
    positions represent the candidate's logical outer boundaries; detected
    internal separator responses occupy the positions between them.
    """

    horizontal_line_positions: tuple[float, ...]
    vertical_line_positions: tuple[float, ...]
    horizontal_line_count: int
    vertical_line_count: int
    estimated_rows: int
    estimated_columns: int
    horizontal_spacing_coefficient_of_variation: float
    vertical_spacing_coefficient_of_variation: float
    horizontal_spacing_regularity: float
    vertical_spacing_regularity: float
    horizontal_line_coverage: float
    vertical_line_coverage: float
    score: float


@dataclass(frozen=True, slots=True)
class _AxisLineEvidence:
    """Keep backend-only strong and weak evidence for one separator axis."""

    positions: tuple[float, ...]
    coverages: tuple[float, ...]
    recovered_positions: tuple[float, ...]
    strong_projection: NDArray[np.float64]
    weak_projection: NDArray[np.float64]


class InternalGridEvidenceAnalyzer(Protocol):
    """Type the shared analysis path for board validation and public extraction."""

    def analyze(self, grayscale_roi: NDArray[np.uint8]) -> InternalGridEvidence:
        """Measure normalized primitive grid evidence from one grayscale ROI."""

        ...

    def rejection_reasons(
        self,
        evidence: InternalGridEvidence,
    ) -> tuple[str, ...]:
        """Return the shared mandatory validation failures for measured evidence."""

        ...


class OpenCvInternalGridEvidenceAnalyzer:
    """Measure regular row-and-column evidence inside a geometric candidate.

    The analyzer combines CLAHE-normalized Canny edges with edges from an adaptive
    threshold. Directional morphological opening removes short decorative strokes.
    Projection profiles locate long horizontal and vertical responses, and nearby
    responses are clustered so a thick separator contributes exactly one boundary.

    Grid evidence is ``0.25 * boundary adequacy + 0.35 * spacing regularity +
    0.40 * line coverage``. Each paired component is the mean of its horizontal
    and vertical values and every value is clamped to ``[0.0, 1.0]``.
    """

    def __init__(self, settings: BoardDetectionSettings) -> None:
        """Receive immutable thresholds shared with the owning board detector."""

        self._settings = settings

    def analyze(self, grayscale_roi: NDArray[np.uint8]) -> InternalGridEvidence:
        """Return deterministic primitive evidence for one grayscale candidate ROI."""

        height, width = grayscale_roi.shape
        normalized = cast(
            NDArray[np.uint8],
            cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale_roi),
        )
        edges = cast(
            NDArray[np.uint8],
            cv2.Canny(
                normalized,
                self._settings.canny_lower_threshold,
                self._settings.canny_upper_threshold,
            ),
        )
        adaptive_block_size = self._adaptive_block_size(width, height)
        adaptive = cast(
            NDArray[np.uint8],
            cv2.adaptiveThreshold(
                normalized,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                adaptive_block_size,
                self._settings.grid_adaptive_constant,
            ),
        )
        adaptive_edges = cast(
            NDArray[np.uint8],
            cv2.Canny(
                adaptive,
                self._settings.canny_lower_threshold,
                self._settings.canny_upper_threshold,
            ),
        )
        combined_edges = cast(
            NDArray[np.uint8],
            cv2.bitwise_or(edges, adaptive_edges),
        )

        horizontal_mask = self._extract_directional_lines(
            combined_edges,
            horizontal=True,
        )
        vertical_mask = self._extract_directional_lines(
            combined_edges,
            horizontal=False,
        )
        horizontal_internal, horizontal_coverages = self._cluster_line_responses(
            horizontal_mask,
            horizontal=True,
        )
        vertical_internal, vertical_coverages = self._cluster_line_responses(
            vertical_mask,
            horizontal=False,
        )
        horizontal_axis = _AxisLineEvidence(
            positions=horizontal_internal,
            coverages=horizontal_coverages,
            recovered_positions=(),
            strong_projection=self._directional_projection(
                horizontal_mask,
                horizontal=True,
            ),
            weak_projection=self._weak_line_projection(
                normalized,
                combined_edges,
                horizontal=True,
            ),
        )
        vertical_axis = _AxisLineEvidence(
            positions=vertical_internal,
            coverages=vertical_coverages,
            recovered_positions=(),
            strong_projection=self._directional_projection(
                vertical_mask,
                horizontal=False,
            ),
            weak_projection=self._weak_line_projection(
                normalized,
                combined_edges,
                horizontal=False,
            ),
        )
        if (
            self._settings.grid_missing_line_recovery_enabled
            and self._settings.grid_missing_line_maximum_recovered_per_axis > 0
        ):
            horizontal_axis = self._recover_single_missing_line(
                horizontal_axis,
                axis_length=height,
                maximum_spacing_cv=(
                    self._settings.maximum_horizontal_spacing_coefficient_of_variation
                ),
            )
            vertical_axis = self._recover_single_missing_line(
                vertical_axis,
                axis_length=width,
                maximum_spacing_cv=(
                    self._settings.maximum_vertical_spacing_coefficient_of_variation
                ),
            )

        horizontal_positions = (0.0, *horizontal_axis.positions, 1.0)
        vertical_positions = (0.0, *vertical_axis.positions, 1.0)
        horizontal_cv, horizontal_regularity = self._spacing_metrics(
            horizontal_positions,
            self._settings.maximum_horizontal_spacing_coefficient_of_variation,
        )
        vertical_cv, vertical_regularity = self._spacing_metrics(
            vertical_positions,
            self._settings.maximum_vertical_spacing_coefficient_of_variation,
        )
        horizontal_coverage = (
            fmean(horizontal_axis.coverages) if horizontal_axis.coverages else 0.0
        )
        vertical_coverage = (
            fmean(vertical_axis.coverages) if vertical_axis.coverages else 0.0
        )
        horizontal_count = len(horizontal_positions)
        vertical_count = len(vertical_positions)
        estimated_rows = max(0, horizontal_count - 1)
        estimated_columns = max(0, vertical_count - 1)
        score = self._grid_score(
            horizontal_count=horizontal_count,
            vertical_count=vertical_count,
            horizontal_regularity=horizontal_regularity,
            vertical_regularity=vertical_regularity,
            horizontal_coverage=horizontal_coverage,
            vertical_coverage=vertical_coverage,
        )

        return InternalGridEvidence(
            horizontal_line_positions=horizontal_positions,
            vertical_line_positions=vertical_positions,
            horizontal_line_count=horizontal_count,
            vertical_line_count=vertical_count,
            estimated_rows=estimated_rows,
            estimated_columns=estimated_columns,
            horizontal_spacing_coefficient_of_variation=horizontal_cv,
            vertical_spacing_coefficient_of_variation=vertical_cv,
            horizontal_spacing_regularity=horizontal_regularity,
            vertical_spacing_regularity=vertical_regularity,
            horizontal_line_coverage=_clamp_unit(horizontal_coverage),
            vertical_line_coverage=_clamp_unit(vertical_coverage),
            score=score,
        )

    def rejection_reasons(
        self,
        evidence: InternalGridEvidence,
    ) -> tuple[str, ...]:
        """Apply shared mandatory grid rules for every analyzer consumer."""

        reasons: list[str] = []
        if (
            evidence.horizontal_line_count
            < self._settings.minimum_horizontal_grid_line_count
        ):
            reasons.append("insufficient horizontal grid lines")
        if (
            evidence.vertical_line_count
            < self._settings.minimum_vertical_grid_line_count
        ):
            reasons.append("insufficient vertical grid lines")
        if evidence.estimated_rows < self._settings.minimum_estimated_rows:
            reasons.append("too few estimated rows")
        if evidence.estimated_columns < self._settings.minimum_estimated_columns:
            reasons.append("too few estimated columns")
        if (
            evidence.horizontal_spacing_coefficient_of_variation
            > self._settings.maximum_horizontal_spacing_coefficient_of_variation
        ):
            reasons.append("irregular horizontal grid spacing")
        if (
            evidence.vertical_spacing_coefficient_of_variation
            > self._settings.maximum_vertical_spacing_coefficient_of_variation
        ):
            reasons.append("irregular vertical grid spacing")
        if (
            evidence.horizontal_line_coverage
            < self._settings.minimum_horizontal_line_coverage
        ):
            reasons.append("insufficient horizontal grid coverage")
        if (
            evidence.vertical_line_coverage
            < self._settings.minimum_vertical_line_coverage
        ):
            reasons.append("insufficient vertical grid coverage")
        if evidence.score < self._settings.minimum_grid_evidence_score:
            reasons.append("grid evidence below required threshold")
        return tuple(reasons)

    def _adaptive_block_size(self, width: int, height: int) -> int:
        """Build a valid odd adaptive-threshold neighborhood for the current ROI."""

        shorter_side = min(width, height)
        requested = max(
            3,
            round(shorter_side * self._settings.grid_adaptive_block_relative_size),
        )
        if requested % 2 == 0:
            requested += 1
        maximum = shorter_side if shorter_side % 2 == 1 else shorter_side - 1
        return max(3, min(requested, maximum))

    def _extract_directional_lines(
        self,
        edges: NDArray[np.uint8],
        *,
        horizontal: bool,
        weak: bool = False,
    ) -> NDArray[np.uint8]:
        """Retain axis-aligned edge runs long enough to represent separators."""

        height, width = edges.shape
        if horizontal:
            relative_length = (
                self._settings.grid_weak_horizontal_line_kernel_relative_length
                if weak
                else self._settings.horizontal_line_kernel_relative_length
            )
            length = max(
                3,
                round(width * relative_length),
            )
            kernel_shape = (length, 1)
        else:
            relative_length = (
                self._settings.grid_weak_vertical_line_kernel_relative_length
                if weak
                else self._settings.vertical_line_kernel_relative_length
            )
            length = max(
                3,
                round(height * relative_length),
            )
            kernel_shape = (1, length)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        return cast(
            NDArray[np.uint8],
            cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel),
        )

    def _weak_line_projection(
        self,
        normalized: NDArray[np.uint8],
        combined_edges: NDArray[np.uint8],
        *,
        horizontal: bool,
    ) -> NDArray[np.float64]:
        """Combine normalized short-run morphology with an axis Sobel profile."""

        weak_mask = self._extract_directional_lines(
            combined_edges,
            horizontal=horizontal,
            weak=True,
        )
        directional_projection = self._directional_projection(
            weak_mask,
            horizontal=horizontal,
        )
        derivative_x, derivative_y = (0, 1) if horizontal else (1, 0)
        gradient = np.abs(
            cv2.Sobel(
                normalized,
                cv2.CV_64F,
                derivative_x,
                derivative_y,
                ksize=3,
            )
        )
        maximum_gradient = float(np.max(gradient)) if gradient.size else 0.0
        if maximum_gradient > 0.0:
            normalized_gradient = gradient / maximum_gradient
        else:
            normalized_gradient = np.zeros_like(gradient, dtype=np.float64)
        gradient_projection = np.mean(
            normalized_gradient,
            axis=1 if horizontal else 0,
            dtype=np.float64,
        )
        return cast(
            NDArray[np.float64],
            np.clip(
                np.maximum(directional_projection, gradient_projection),
                0.0,
                1.0,
            ).astype(np.float64),
        )

    @staticmethod
    def _directional_projection(
        line_mask: NDArray[np.uint8],
        *,
        horizontal: bool,
    ) -> NDArray[np.float64]:
        """Normalize directional line coverage into one bounded axis profile."""

        height, width = line_mask.shape
        if horizontal:
            raw_projection = np.count_nonzero(line_mask, axis=1)
            normalization_length = width
        else:
            raw_projection = np.count_nonzero(line_mask, axis=0)
            normalization_length = height
        return cast(
            NDArray[np.float64],
            np.clip(
                raw_projection.astype(np.float64) / max(1, normalization_length),
                0.0,
                1.0,
            ).astype(np.float64),
        )

    def _cluster_line_responses(
        self,
        line_mask: NDArray[np.uint8],
        *,
        horizontal: bool,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Merge nearby projection peaks and remove candidate-border responses."""

        height, width = line_mask.shape
        if horizontal:
            axis_length = height
        else:
            axis_length = width
        projection = self._directional_projection(
            line_mask,
            horizontal=horizontal,
        )
        response_indices = np.flatnonzero(
            projection >= self._settings.minimum_grid_line_response
        ).tolist()
        cluster_distance = max(
            1,
            round(axis_length * self._settings.grid_line_cluster_distance_relative),
        )

        clusters: list[list[int]] = []
        for raw_index in response_indices:
            index = int(raw_index)
            if not clusters or index - clusters[-1][-1] > cluster_distance:
                clusters.append([index])
            else:
                clusters[-1].append(index)

        border_tolerance = (
            axis_length * self._settings.grid_border_line_exclusion_tolerance
        )
        normalized_positions: list[float] = []
        coverages: list[float] = []
        denominator = max(1, axis_length - 1)
        for cluster in clusters:
            weights = tuple(float(projection[index]) for index in cluster)
            weight_sum = sum(weights)
            if weight_sum <= 0.0:
                continue
            center = (
                sum(
                    index * weight
                    for index, weight in zip(cluster, weights, strict=True)
                )
                / weight_sum
            )
            if center <= border_tolerance or center >= denominator - border_tolerance:
                continue
            normalized_positions.append(_clamp_unit(center / denominator))
            coverages.append(_clamp_unit(max(weights)))
        return tuple(normalized_positions), tuple(coverages)

    def _recover_single_missing_line(
        self,
        axis: _AxisLineEvidence,
        *,
        axis_length: int,
        maximum_spacing_cv: float,
    ) -> _AxisLineEvidence:
        """Recover one image-supported separator from one otherwise regular gap."""

        if len(axis.positions) < 3 or axis_length < 3:
            return axis
        boundaries = (0.0, *axis.positions, 1.0)
        gaps = tuple(current - previous for previous, current in pairwise(boundaries))
        typical_spacing = median(gaps)
        if typical_spacing <= 0.0:
            return axis

        large_gap_indices = tuple(
            index
            for index, gap in enumerate(gaps)
            if (
                self._settings.grid_missing_line_minimum_gap_factor
                <= gap / typical_spacing
                <= self._settings.grid_missing_line_maximum_gap_factor
            )
        )
        if len(large_gap_indices) != 1:
            return axis
        large_gap_index = large_gap_indices[0]
        if large_gap_index in (0, len(gaps) - 1):
            return axis
        other_gaps = tuple(
            gap for index, gap in enumerate(gaps) if index != large_gap_index
        )
        if any(
            abs(gap - typical_spacing) / typical_spacing
            > self._settings.grid_missing_line_maximum_other_gap_deviation
            for gap in other_gaps
        ):
            return axis

        gap_left = boundaries[large_gap_index]
        gap_right = boundaries[large_gap_index + 1]
        expected = (gap_left + gap_right) / 2.0
        border_tolerance = self._settings.grid_border_line_exclusion_tolerance
        if expected <= border_tolerance or expected >= 1.0 - border_tolerance:
            return axis
        denominator = axis_length - 1
        expected_pixel = expected * denominator
        search_radius = max(
            1,
            round(
                typical_spacing
                * self._settings.grid_missing_line_search_half_width_fraction
                * denominator
            ),
        )
        search_start = max(1, floor(expected_pixel - search_radius))
        search_end = min(axis_length - 2, floor(expected_pixel + search_radius))
        local_peaks = tuple(
            index
            for index in range(search_start, search_end + 1)
            if (
                axis.weak_projection[index] >= axis.weak_projection[index - 1]
                and axis.weak_projection[index] >= axis.weak_projection[index + 1]
            )
        )
        if not local_peaks:
            return axis
        peak_index = min(
            local_peaks,
            key=lambda index: (
                -float(axis.weak_projection[index]),
                abs(index - expected_pixel),
                index,
            ),
        )
        peak_response = float(axis.weak_projection[peak_index])
        if peak_response < self._settings.grid_missing_line_minimum_weak_response:
            return axis
        recovered_position = peak_index / denominator
        if min(abs(recovered_position - position) for position in boundaries) <= (
            self._settings.grid_line_cluster_distance_relative
        ):
            return axis

        combined = sorted(
            (
                *zip(axis.positions, axis.coverages, strict=True),
                (recovered_position, _clamp_unit(peak_response)),
            ),
            key=lambda item: item[0],
        )
        recovered_positions = tuple(position for position, _ in combined)
        recovered_coverages = tuple(coverage for _, coverage in combined)
        final_boundaries = (0.0, *recovered_positions, 1.0)
        converted_pixels = tuple(
            floor(position * axis_length + 0.5) for position in final_boundaries
        )
        if len(converted_pixels) != len(set(converted_pixels)):
            return axis

        old_cv, _ = self._spacing_metrics(boundaries, maximum_spacing_cv)
        new_cv, _ = self._spacing_metrics(final_boundaries, maximum_spacing_cv)
        if new_cv >= old_cv:
            return axis
        if old_cv - new_cv < self._settings.grid_missing_line_minimum_cv_improvement:
            return axis
        if new_cv > maximum_spacing_cv:
            return axis
        return _AxisLineEvidence(
            positions=recovered_positions,
            coverages=recovered_coverages,
            recovered_positions=(recovered_position,),
            strong_projection=axis.strong_projection,
            weak_projection=axis.weak_projection,
        )

    @staticmethod
    def _spacing_metrics(
        positions: tuple[float, ...],
        maximum_coefficient_of_variation: float,
    ) -> tuple[float, float]:
        """Measure normalized separator-spacing variance and derived regularity."""

        if len(positions) < 3:
            return 1.0, 0.0
        spacings = tuple(
            current - previous for previous, current in pairwise(positions)
        )
        mean_spacing = fmean(spacings)
        coefficient = pstdev(spacings) / mean_spacing if mean_spacing > 0.0 else 1.0
        regularity = _clamp_unit(1.0 - coefficient / maximum_coefficient_of_variation)
        return coefficient, regularity

    def _grid_score(
        self,
        *,
        horizontal_count: int,
        vertical_count: int,
        horizontal_regularity: float,
        vertical_regularity: float,
        horizontal_coverage: float,
        vertical_coverage: float,
    ) -> float:
        """Combine boundary quantity, spacing, and coverage into documented evidence."""

        required_horizontal = max(
            self._settings.minimum_horizontal_grid_line_count,
            self._settings.minimum_estimated_rows + 1,
        )
        required_vertical = max(
            self._settings.minimum_vertical_grid_line_count,
            self._settings.minimum_estimated_columns + 1,
        )
        horizontal_count_score = _clamp_unit(
            (horizontal_count - 2) / max(1, required_horizontal - 2)
        )
        vertical_count_score = _clamp_unit(
            (vertical_count - 2) / max(1, required_vertical - 2)
        )
        count_score = (horizontal_count_score + vertical_count_score) / 2.0
        regularity_score = (horizontal_regularity + vertical_regularity) / 2.0
        coverage_score = (horizontal_coverage + vertical_coverage) / 2.0
        return _clamp_unit(
            0.25 * count_score + 0.35 * regularity_score + 0.40 * coverage_score
        )
