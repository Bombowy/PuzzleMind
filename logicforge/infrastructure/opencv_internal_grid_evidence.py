"""Internal OpenCV regular-grid evidence used only to validate board candidates.

This module deliberately does not implement the public ``GridDetector`` port. It
measures whether a rectangular board candidate contains a credible axis-aligned
grid, but it does not expose cells or grid geometry to parsing code.
"""

from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean, pstdev
from typing import cast

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
        normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            grayscale_roi
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

        horizontal_positions = (0.0, *horizontal_internal, 1.0)
        vertical_positions = (0.0, *vertical_internal, 1.0)
        horizontal_cv, horizontal_regularity = self._spacing_metrics(
            horizontal_positions,
            self._settings.maximum_horizontal_spacing_coefficient_of_variation,
        )
        vertical_cv, vertical_regularity = self._spacing_metrics(
            vertical_positions,
            self._settings.maximum_vertical_spacing_coefficient_of_variation,
        )
        horizontal_coverage = (
            fmean(horizontal_coverages) if horizontal_coverages else 0.0
        )
        vertical_coverage = fmean(vertical_coverages) if vertical_coverages else 0.0
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
    ) -> NDArray[np.uint8]:
        """Retain axis-aligned edge runs long enough to represent separators."""

        height, width = edges.shape
        if horizontal:
            length = max(
                3,
                round(width * self._settings.horizontal_line_kernel_relative_length),
            )
            kernel_shape = (length, 1)
        else:
            length = max(
                3,
                round(height * self._settings.vertical_line_kernel_relative_length),
            )
            kernel_shape = (1, length)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
        return cast(
            NDArray[np.uint8],
            cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel),
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
            raw_projection = np.count_nonzero(line_mask, axis=1)
            normalization_length = width
            axis_length = height
        else:
            raw_projection = np.count_nonzero(line_mask, axis=0)
            normalization_length = height
            axis_length = width
        projection = raw_projection.astype(np.float64) / normalization_length
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
