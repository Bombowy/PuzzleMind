"""Classical OpenCV adapter for puzzle-neutral cell-color classification."""

from dataclasses import dataclass
from itertools import combinations
from math import ceil, dist

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


class OpenCvColorDetector(ColorDetector):
    """Classify cell backgrounds by robust LAB sampling and complete-link clusters.

    The adapter samples only each cell's configured center fraction, converts BGR
    pixels to OpenCV LAB, starts from a channel-wise median, removes the configured
    fraction of farthest pixels, and averages the retained sample. This suppresses
    grid borders, highlights, and minority symbol strokes without interpreting any
    symbol semantics.

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
        """Produce one robust representative from each row-major central ROI."""

        samples: list[_CellColorSample] = []
        for cell in grid.cells:
            bounds = self._inner_sample_bounds(cell)
            left, top, right, bottom = bounds
            bgr_roi = screenshot.image[top:bottom, left:right]
            pixel_count = int(bgr_roi.shape[0] * bgr_roi.shape[1])
            if pixel_count < self._settings.minimum_sample_pixels:
                diagnostics = self._diagnostics(
                    grid,
                    tuple(samples),
                    (),
                    rejection_reasons=(
                        (
                            f"cell ({cell.row}, {cell.column}) provides only "
                            f"{pixel_count} central pixels; minimum is "
                            f"{self._settings.minimum_sample_pixels}"
                        ),
                    ),
                )
                raise ColorDetectionError(
                    "Color detection requires larger cell samples: "
                    + diagnostics.rejection_reasons[0],
                    diagnostics,
                )
            representative, spread = self._representative_lab(bgr_roi)
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

    def _inner_sample_bounds(self, cell: CellBounds) -> tuple[int, int, int, int]:
        """Return a non-empty half-open crop centered inside one logical cell."""

        horizontal_inset = int(
            cell.width * (1.0 - self._settings.sample_inner_fraction) / 2.0
        )
        vertical_inset = int(
            cell.height * (1.0 - self._settings.sample_inner_fraction) / 2.0
        )
        return (
            cell.x + horizontal_inset,
            cell.y + vertical_inset,
            cell.x + cell.width - horizontal_inset,
            cell.y + cell.height - vertical_inset,
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
        # A second median remains stable even when a future symbol occupies a
        # sizeable minority of the central ROI. Its pixels are not interpreted;
        # they simply cannot pull the background estimate as a mean would.
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
