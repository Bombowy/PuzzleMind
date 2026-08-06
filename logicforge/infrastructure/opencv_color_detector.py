"""Classical OpenCV adapter for puzzle-neutral cell-color classification."""

from dataclasses import dataclass
from itertools import combinations
from math import ceil, dist
from statistics import median
from typing import Never

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import ColorDetectionSettings
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionError,
    ColorDetectionResult,
    ColorDetector,
    ColorObservation,
    LabColor,
)
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class _CellColorSample:
    """Keep one adapter-internal robust LAB estimate and sampling evidence."""

    row: int
    column: int
    representative_lab: LabColor
    pixel_count: int
    spread: float


class _CornerSampleGeometryError(RuntimeError):
    """Report a cell too small for four positive inset patch rectangles."""


class _CornerSampleConsensusError(RuntimeError):
    """Report insufficient agreement among primitive corner representatives."""


class OpenCvColorDetector(ColorDetector):
    """Classify cell backgrounds by robust LAB sampling and complete-link clusters.

    The adapter samples four inset corner patches in every cell, away from both
    cell edges and the symbol-prone center. Each patch uses the existing robust LAB
    estimator. One corner-level outlier is removed before the retained three are
    combined, without interpreting symbol semantics.

    Clustering is deterministic complete-link agglomeration. Two clusters merge
    only if every cross-cluster representative distance is at most the configured
    threshold. Final clusters are sorted lexicographically by LAB center before
    receiving ``C0..Cn`` identifiers, so traversal or merge order cannot rename
    otherwise identical output.
    """

    def __init__(self, settings: ColorDetectionSettings | None = None) -> None:
        """Receive immutable thresholds or use documented production defaults."""

        self._settings = settings or ColorDetectionSettings()

    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
    ) -> ColorDetectionResult:
        """Return complete logical color classes for every public grid cell."""

        self._validate_grid_bounds(screenshot, grid)
        samples = self._sample_cells(screenshot, grid)
        clusters = self._cluster_samples(samples)
        centers = tuple(self._cluster_center(cluster, samples) for cluster in clusters)
        observation_by_index = self._build_observations(samples, clusters, centers)
        observations = tuple(
            observation_by_index[index] for index in range(len(samples))
        )
        matrix = tuple(
            tuple(
                observations[row * grid.columns + column].color_id
                for column in range(grid.columns)
            )
            for row in range(grid.rows)
        )
        mean_confidence = float(
            sum(observation.confidence for observation in observations)
            / len(observations)
        )
        diagnostics = self._diagnostics(
            grid,
            samples,
            centers,
            minimum_intercluster_distance=self._minimum_center_distance(centers),
        )
        return ColorDetectionResult(
            observations=observations,
            color_count=len(clusters),
            color_matrix=matrix,
            mean_confidence=self._clamp(mean_confidence),
            diagnostics=diagnostics,
        )

    def _validate_grid_bounds(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
    ) -> None:
        """Fail before cropping if public cell geometry exceeds the screenshot."""

        reasons: list[str] = []
        if len(grid.cells) != grid.rows * grid.columns:
            reasons.append("grid does not contain one cell per row and column")
        for cell in grid.cells:
            if cell.x + cell.width > screenshot.width:
                reasons.append(
                    f"cell ({cell.row}, {cell.column}) exceeds screenshot width"
                )
            if cell.y + cell.height > screenshot.height:
                reasons.append(
                    f"cell ({cell.row}, {cell.column}) exceeds screenshot height"
                )
        if reasons:
            diagnostics = self._empty_diagnostics(grid, tuple(dict.fromkeys(reasons)))
            raise ColorDetectionError(
                "Color detection rejected invalid grid geometry: "
                + "; ".join(diagnostics.rejection_reasons),
                diagnostics,
            )

    def _sample_cells(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
    ) -> tuple[_CellColorSample, ...]:
        """Produce one robust representative from four inset patches per cell."""

        samples: list[_CellColorSample] = []
        for cell in grid.cells:
            try:
                bounds = self._corner_sample_bounds(cell)
            except _CornerSampleGeometryError as error:
                self._raise_sampling_error(grid, samples, cell, str(error))
            representatives: list[LabColor] = []
            spreads: list[float] = []
            pixel_count = 0
            for left, top, right, bottom in bounds:
                bgr_roi = screenshot.image[top:bottom, left:right]
                pixel_count += int(bgr_roi.shape[0] * bgr_roi.shape[1])
                representative, spread = self._representative_lab(bgr_roi)
                representatives.append(representative)
                spreads.append(spread)
            if pixel_count < self._settings.minimum_sample_pixels:
                self._raise_sampling_error(
                    grid,
                    samples,
                    cell,
                    (
                        f"provides only {pixel_count} corner-patch pixels; "
                        f"minimum is {self._settings.minimum_sample_pixels}"
                    ),
                )
            try:
                representative, spread = self._combine_corner_representatives(
                    tuple(representatives),
                    tuple(spreads),
                )
            except _CornerSampleConsensusError as error:
                self._raise_sampling_error(grid, samples, cell, str(error))
            samples.append(
                _CellColorSample(
                    row=cell.row,
                    column=cell.column,
                    representative_lab=representative,
                    pixel_count=pixel_count,
                    spread=spread,
                )
            )
        return tuple(samples)

    def _corner_sample_bounds(
        self,
        cell: CellBounds,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Return TL, TR, BL, BR half-open inset patches for one logical cell."""

        patch_width = round(cell.width * self._settings.corner_sample_patch_fraction)
        patch_height = round(cell.height * self._settings.corner_sample_patch_fraction)
        offset_x = round(cell.width * self._settings.corner_sample_offset_fraction)
        offset_y = round(cell.height * self._settings.corner_sample_offset_fraction)
        left = cell.x + offset_x
        right = cell.x + cell.width - offset_x - patch_width
        top = cell.y + offset_y
        bottom = cell.y + cell.height - offset_y - patch_height
        bounds = (
            (left, top, left + patch_width, top + patch_height),
            (right, top, right + patch_width, top + patch_height),
            (left, bottom, left + patch_width, bottom + patch_height),
            (right, bottom, right + patch_width, bottom + patch_height),
        )
        cell_right = cell.x + cell.width
        cell_bottom = cell.y + cell.height
        center_x = cell.x + cell.width / 2.0
        center_y = cell.y + cell.height / 2.0
        if patch_width <= 0 or patch_height <= 0:
            raise _CornerSampleGeometryError(
                "has degenerate corner-patch dimensions after relative rounding"
            )
        if any(
            patch_left < cell.x
            or patch_top < cell.y
            or patch_right > cell_right
            or patch_bottom > cell_bottom
            or patch_right <= patch_left
            or patch_bottom <= patch_top
            for patch_left, patch_top, patch_right, patch_bottom in bounds
        ):
            raise _CornerSampleGeometryError(
                "has a corner patch outside its half-open cell bounds"
            )
        if not (
            bounds[0][2] <= center_x
            and bounds[2][2] <= center_x
            and bounds[1][0] >= center_x
            and bounds[3][0] >= center_x
            and bounds[0][3] <= center_y
            and bounds[1][3] <= center_y
            and bounds[2][1] >= center_y
            and bounds[3][1] >= center_y
        ):
            raise _CornerSampleGeometryError(
                "has a rounded corner patch crossing the cell center"
            )
        return bounds

    def _combine_corner_representatives(
        self,
        representatives: tuple[LabColor, ...],
        spreads: tuple[float, ...],
    ) -> tuple[LabColor, float]:
        """Reject one corner outlier and combine the retained three by median."""

        if len(representatives) != 4 or len(spreads) != 4:
            raise ValueError("Exactly four corner representatives are required.")
        colors = np.asarray(representatives, dtype=np.float64)
        corner_median = np.median(colors, axis=0)
        distances = np.linalg.norm(colors - corner_median, axis=1)
        rejected_index = max(
            range(4),
            key=lambda index: (float(distances[index]), -index),
        )
        retained_indices = tuple(index for index in range(4) if index != rejected_index)
        minimum_consistent = self._settings.corner_sample_minimum_consistent_patches
        consensus_indices = (
            tuple(range(4)) if minimum_consistent == 4 else retained_indices
        )
        maximum_consistent = max(
            (
                len(indices)
                for size in range(2, len(consensus_indices) + 1)
                for indices in combinations(consensus_indices, size)
                if all(
                    dist(representatives[left], representatives[right])
                    <= self._settings.cluster_distance_threshold
                    for left, right in combinations(indices, 2)
                )
            ),
            default=1,
        )
        if maximum_consistent < minimum_consistent:
            raise _CornerSampleConsensusError(
                "corner patches did not provide the configured LAB consensus"
            )

        retained_colors = colors[list(retained_indices)]
        final_array = np.median(retained_colors, axis=0)
        retained_distances = np.linalg.norm(retained_colors - final_array, axis=1)
        retained_patch_spread = float(
            median(spreads[index] for index in retained_indices)
        )
        consensus_spread = float(np.median(retained_distances))
        rejected_outlier_penalty = min(
            float(distances[rejected_index]),
            self._settings.cluster_distance_threshold,
        )
        representative: LabColor = (
            float(final_array[0]),
            float(final_array[1]),
            float(final_array[2]),
        )
        return representative, max(
            retained_patch_spread,
            consensus_spread,
            rejected_outlier_penalty,
        )

    def _raise_sampling_error(
        self,
        grid: GridDetection,
        samples: list[_CellColorSample],
        cell: CellBounds,
        reason: str,
    ) -> Never:
        """Raise one typed fail-closed sampling error with partial diagnostics."""

        full_reason = f"cell ({cell.row}, {cell.column}) {reason}"
        diagnostics = self._diagnostics(
            grid,
            tuple(samples),
            (),
            rejection_reasons=(full_reason,),
        )
        raise ColorDetectionError(
            "Color detection requires reliable corner samples: " + full_reason,
            diagnostics,
        )

    def _representative_lab(
        self,
        bgr_roi: NDArray[np.uint8],
    ) -> tuple[LabColor, float]:
        """Estimate LAB color after trimming pixels farthest from an initial median."""

        lab_pixels = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        floating_pixels = lab_pixels.astype(np.float64)
        initial_median = np.median(floating_pixels, axis=0)
        initial_distances = np.linalg.norm(floating_pixels - initial_median, axis=1)
        retain_count = max(
            1,
            ceil(
                floating_pixels.shape[0] * (1.0 - self._settings.outlier_trim_fraction)
            ),
        )
        retained_indices = np.argsort(initial_distances, kind="stable")[:retain_count]
        retained_pixels = floating_pixels[retained_indices]
        # A second median remains stable when a patch contains a minority local
        # artifact. Its pixels are not interpreted; they simply cannot pull the
        # background estimate as a mean would.
        representative_array = np.median(retained_pixels, axis=0)
        retained_distances = np.linalg.norm(
            retained_pixels - representative_array,
            axis=1,
        )
        representative: LabColor = (
            float(representative_array[0]),
            float(representative_array[1]),
            float(representative_array[2]),
        )
        return representative, float(np.median(retained_distances))

    def _cluster_samples(
        self,
        samples: tuple[_CellColorSample, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Merge the closest complete-link-compatible pair until none remains."""

        clusters: list[tuple[int, ...]] = [(index,) for index in range(len(samples))]
        while True:
            merge_candidate = self._next_merge_candidate(clusters, samples)
            if merge_candidate is None:
                break
            left_index, right_index = merge_candidate
            merged = tuple(sorted((*clusters[left_index], *clusters[right_index])))
            clusters = [
                cluster
                for index, cluster in enumerate(clusters)
                if index not in (left_index, right_index)
            ]
            clusters.append(merged)
            clusters.sort(key=lambda cluster: cluster[0])

        clusters.sort(key=lambda cluster: self._cluster_sort_key(cluster, samples))
        return tuple(clusters)

    def _next_merge_candidate(
        self,
        clusters: list[tuple[int, ...]],
        samples: tuple[_CellColorSample, ...],
    ) -> tuple[int, int] | None:
        """Select the closest valid pair with explicit deterministic tie-breaking."""

        candidates: list[tuple[float, int, int, int, int]] = []
        for left_index, right_index in combinations(range(len(clusters)), 2):
            distance = self._complete_link_distance(
                clusters[left_index],
                clusters[right_index],
                samples,
            )
            if distance <= self._settings.cluster_distance_threshold:
                candidates.append(
                    (
                        distance,
                        clusters[left_index][0],
                        clusters[right_index][0],
                        left_index,
                        right_index,
                    )
                )
        if not candidates:
            return None
        _, _, _, left_index, right_index = min(candidates)
        return left_index, right_index

    @staticmethod
    def _complete_link_distance(
        left: tuple[int, ...],
        right: tuple[int, ...],
        samples: tuple[_CellColorSample, ...],
    ) -> float:
        """Return the largest representative distance across two clusters."""

        return max(
            dist(
                samples[left_index].representative_lab,
                samples[right_index].representative_lab,
            )
            for left_index in left
            for right_index in right
        )

    def _cluster_sort_key(
        self,
        cluster: tuple[int, ...],
        samples: tuple[_CellColorSample, ...],
    ) -> tuple[float, float, float, int]:
        """Order logical IDs by LAB center, then stable source index."""

        center = self._cluster_center(cluster, samples)
        return (*center, cluster[0])

    @staticmethod
    def _cluster_center(
        cluster: tuple[int, ...],
        samples: tuple[_CellColorSample, ...],
    ) -> LabColor:
        """Average robust representatives into one diagnostic class center."""

        member_colors = np.asarray(
            [samples[index].representative_lab for index in cluster],
            dtype=np.float64,
        )
        center = np.mean(member_colors, axis=0)
        return float(center[0]), float(center[1]), float(center[2])

    def _build_observations(
        self,
        samples: tuple[_CellColorSample, ...],
        clusters: tuple[tuple[int, ...], ...],
        centers: tuple[LabColor, ...],
    ) -> dict[int, ColorObservation]:
        """Assign sorted logical IDs and calibrated confidence to every sample."""

        observations: dict[int, ColorObservation] = {}
        for cluster_index, (cluster, center) in enumerate(
            zip(clusters, centers, strict=True)
        ):
            for sample_index in cluster:
                sample = samples[sample_index]
                homogeneity = self._clamp(
                    1.0 - sample.spread / self._settings.maximum_within_cell_spread
                )
                cluster_fit = self._clamp(
                    1.0
                    - dist(sample.representative_lab, center)
                    / self._settings.cluster_distance_threshold
                )
                confidence = self._clamp(
                    self._settings.homogeneity_confidence_weight * homogeneity
                    + self._settings.cluster_fit_confidence_weight * cluster_fit
                )
                observations[sample_index] = ColorObservation(
                    row=sample.row,
                    column=sample.column,
                    color_id=f"C{cluster_index}",
                    confidence=confidence,
                    representative_lab=sample.representative_lab,
                )
        return observations

    def _diagnostics(
        self,
        grid: GridDetection,
        samples: tuple[_CellColorSample, ...],
        centers: tuple[LabColor, ...],
        *,
        minimum_intercluster_distance: float | None = None,
        rejection_reasons: tuple[str, ...] = (),
    ) -> ColorDetectionDiagnostics:
        """Translate adapter state into puzzle-neutral immutable primitives."""

        return ColorDetectionDiagnostics(
            rows=grid.rows,
            columns=grid.columns,
            sample_inner_fraction=self._settings.sample_inner_fraction,
            cluster_distance_threshold=self._settings.cluster_distance_threshold,
            sample_pixel_counts=tuple(sample.pixel_count for sample in samples),
            within_cell_spreads=tuple(sample.spread for sample in samples),
            cluster_centers_lab=centers,
            minimum_intercluster_distance=minimum_intercluster_distance,
            rejection_reasons=rejection_reasons,
        )

    def _empty_diagnostics(
        self,
        grid: GridDetection,
        rejection_reasons: tuple[str, ...],
    ) -> ColorDetectionDiagnostics:
        """Build failure context when validation stops before cell sampling."""

        return self._diagnostics(
            grid,
            (),
            (),
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _minimum_center_distance(centers: tuple[LabColor, ...]) -> float | None:
        """Return class separation evidence when at least two classes exist."""

        if len(centers) < 2:
            return None
        return min(dist(left, right) for left, right in combinations(centers, 2))

    @staticmethod
    def _clamp(value: float) -> float:
        """Keep public confidence calculations in their documented unit range."""

        return max(0.0, min(1.0, value))
