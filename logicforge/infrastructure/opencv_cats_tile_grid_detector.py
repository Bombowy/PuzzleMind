"""OpenCV Cats board/grid detection fitted directly from colored tile lattices."""

from dataclasses import dataclass, replace
from itertools import pairwise
from math import ceil, floor
from statistics import fmean, median, pstdev

import cv2
import numpy as np

from logicforge.config.settings import CatsTileGridDetectionSettings
from logicforge.plugins.cats.tile_grid import (
    CatsTileComponentDiagnostic,
    CatsTileGridDetection,
    CatsTileGridDetectionError,
    CatsTileGridDetector,
    CatsTileGridDiagnostics,
)
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot


def _clamp_unit(value: float) -> float:
    """Clamp one public score into its documented unit interval."""

    return max(0.0, min(1.0, value))


def _coefficient_of_variation(values: tuple[float, ...]) -> float:
    """Return deterministic population CV or one for unusable measurements."""

    if not values:
        return 1.0
    mean = fmean(values)
    if mean <= 0.0:
        return 1.0
    return float(pstdev(values) / mean)


def _round_pixel(value: float) -> int:
    """Round non-negative pixel geometry by a deterministic half-up rule."""

    return floor(value + 0.5)


@dataclass(frozen=True, slots=True)
class _TileComponent:
    """Retain one connected component and its public diagnostic index."""

    index: int
    diagnostic: CatsTileComponentDiagnostic


@dataclass(frozen=True, slots=True)
class _SizeFamily:
    """Group geometrically similar tile candidates before spatial fitting."""

    components: tuple[_TileComponent, ...]
    median_width: float
    median_height: float
    width_cv: float
    height_cv: float


@dataclass(frozen=True, slots=True)
class _CenterRun:
    """Describe one ordered regular subsequence of clustered tile centers."""

    centers: tuple[float, ...]
    pitch: float
    pitch_cv: float


@dataclass(frozen=True, slots=True)
class _CenterCluster:
    """Keep one aligned center and the number of components supporting it."""

    center: float
    support: int


@dataclass(frozen=True, slots=True)
class _LatticeCandidate:
    """Keep a complete fitted lattice plus evidence needed for total ordering."""

    family: _SizeFamily
    row_centers: tuple[float, ...]
    column_centers: tuple[float, ...]
    row_pitch: float
    column_pitch: float
    row_pitch_cv: float
    column_pitch_cv: float
    selected_component_indices: tuple[int, ...]
    missing_slot_coordinates: tuple[tuple[int, int], ...]
    row_component_counts: tuple[int, ...]
    column_component_counts: tuple[int, ...]
    row_support_ratios: tuple[float, ...]
    column_support_ratios: tuple[float, ...]
    minimum_row_support_ratio: float
    minimum_column_support_ratio: float
    occupancy_ratio: float
    mean_slot_residual: float
    grid_score: float
    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...]


class OpenCvCatsTileGridDetector(CatsTileGridDetector):
    """Derive Cats board and cells from repeated colored tile centers, not contours."""

    def __init__(
        self,
        settings: CatsTileGridDetectionSettings | None = None,
    ) -> None:
        """Use immutable scale-relative settings or calibrated production defaults."""

        self._settings = settings or CatsTileGridDetectionSettings()

    def detect(self, screenshot: Screenshot) -> CatsTileGridDetection:
        """Fit the highest-scoring complete 2D tile lattice in the full screenshot."""

        components, raw_component_count = self._extract_components(screenshot)
        tile_candidates = tuple(
            component
            for component in components
            if not component.diagnostic.rejection_reasons
        )
        families = self._build_size_families(tile_candidates)
        lattice_candidates = tuple(
            candidate
            for family in families
            for candidate in self._fit_family_lattices(screenshot, family)
        )
        accepted = tuple(
            candidate for candidate in lattice_candidates if candidate.accepted
        )
        if not accepted:
            best = (
                min(lattice_candidates, key=self._lattice_sort_key)
                if lattice_candidates
                else None
            )
            reasons = self._failure_reasons(
                raw_component_count,
                len(tile_candidates),
                families,
                best,
            )
            diagnostics = self._diagnostics(
                raw_component_count=raw_component_count,
                components=components,
                candidate_tile_count=len(tile_candidates),
                candidate=best,
                rejection_reasons=reasons,
            )
            raise CatsTileGridDetectionError(
                "No complete regular Cats tile lattice passed detection: "
                + "; ".join(reasons),
                diagnostics,
            )

        selected = min(accepted, key=self._lattice_sort_key)
        grid = self._grid_detection(selected)
        board = BoardDetection(
            x=selected.vertical_lines[0],
            y=selected.horizontal_lines[0],
            width=selected.vertical_lines[-1] - selected.vertical_lines[0],
            height=selected.horizontal_lines[-1] - selected.horizontal_lines[0],
            confidence=selected.grid_score,
        )
        diagnostics = self._diagnostics(
            raw_component_count=raw_component_count,
            components=components,
            candidate_tile_count=len(tile_candidates),
            candidate=selected,
            rejection_reasons=(),
        )
        return CatsTileGridDetection(board=board, grid=grid, diagnostics=diagnostics)

    def _extract_components(
        self,
        screenshot: Screenshot,
    ) -> tuple[tuple[_TileComponent, ...], int]:
        """Build HSV-or-LAB chroma mask and measure each connected color region."""

        image = screenshot.image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_float = lab.astype(np.float64)
        chroma = np.hypot(lab_float[:, :, 1] - 128.0, lab_float[:, :, 2] - 128.0)
        mask = np.where(
            (hsv[:, :, 1] >= self._settings.tile_minimum_hsv_saturation)
            | (chroma >= self._settings.tile_minimum_lab_chroma),
            np.uint8(255),
            np.uint8(0),
        )
        kernel_size = max(
            1,
            _round_pixel(
                min(screenshot.width, screenshot.height)
                * self._settings.tile_mask_kernel_relative_size
            ),
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        screenshot_area = screenshot.width * screenshot.height
        components: list[_TileComponent] = []
        for label in range(1, label_count):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            fill_ratio = area / (width * height)
            aspect_ratio = width / height
            area_ratio = area / screenshot_area
            reasons: list[str] = []
            if area_ratio < self._settings.tile_minimum_component_area_ratio:
                reasons.append("component area was below tile minimum")
            if area_ratio > self._settings.tile_maximum_component_area_ratio:
                reasons.append("component area exceeded tile maximum")
            if not (
                self._settings.tile_minimum_aspect_ratio
                <= aspect_ratio
                <= self._settings.tile_maximum_aspect_ratio
            ):
                reasons.append("component aspect ratio was outside tile range")
            if fill_ratio < self._settings.tile_minimum_fill_ratio:
                reasons.append("component fill ratio was below tile minimum")
            mean_chroma = float(np.mean(chroma[labels == label]))
            diagnostic = CatsTileComponentDiagnostic(
                x=x,
                y=y,
                width=width,
                height=height,
                center_x=float(centroids[label, 0]),
                center_y=float(centroids[label, 1]),
                area=area,
                fill_ratio=_clamp_unit(fill_ratio),
                aspect_ratio=aspect_ratio,
                mean_lab_chroma=mean_chroma,
                accepted=False,
                rejection_reasons=tuple(reasons),
            )
            components.append(
                _TileComponent(index=len(components), diagnostic=diagnostic)
            )
        return tuple(components), label_count - 1

    def _build_size_families(
        self,
        components: tuple[_TileComponent, ...],
    ) -> tuple[_SizeFamily, ...]:
        """Create deterministic anchored families of similar widths and heights."""

        minimum_count = ceil(
            self._settings.tile_grid_minimum_rows
            * self._settings.tile_grid_minimum_columns
            * self._settings.tile_grid_minimum_occupancy_ratio
        )
        tolerance = self._settings.tile_size_family_tolerance_ratio
        family_indices: set[tuple[int, ...]] = set()
        for anchor in components:
            anchor_width = anchor.diagnostic.width
            anchor_height = anchor.diagnostic.height
            member_indices = tuple(
                component.index
                for component in components
                if abs(component.diagnostic.width / anchor_width - 1.0) <= tolerance
                and abs(component.diagnostic.height / anchor_height - 1.0) <= tolerance
            )
            if len(member_indices) >= minimum_count:
                family_indices.add(member_indices)

        by_index = {component.index: component for component in components}
        families: list[_SizeFamily] = []
        for indices in sorted(family_indices, key=lambda item: (-len(item), item)):
            family_components = tuple(by_index[index] for index in indices)
            widths = tuple(
                float(member.diagnostic.width) for member in family_components
            )
            heights = tuple(
                float(member.diagnostic.height) for member in family_components
            )
            width_cv = _coefficient_of_variation(widths)
            height_cv = _coefficient_of_variation(heights)
            if max(width_cv, height_cv) > self._settings.tile_size_cv_maximum:
                continue
            families.append(
                _SizeFamily(
                    components=family_components,
                    median_width=float(median(widths)),
                    median_height=float(median(heights)),
                    width_cv=width_cv,
                    height_cv=height_cv,
                )
            )
        return tuple(families)

    def _fit_family_lattices(
        self,
        screenshot: Screenshot,
        family: _SizeFamily,
    ) -> tuple[_LatticeCandidate, ...]:
        """Fit regular row/column center runs and assign unique tile slots."""

        row_clusters = self._cluster_centers(
            family.components,
            axis="y",
            tolerance=(
                family.median_height
                * self._settings.tile_center_cluster_tolerance_ratio
            ),
        )
        column_clusters = self._cluster_centers(
            family.components,
            axis="x",
            tolerance=(
                family.median_width * self._settings.tile_center_cluster_tolerance_ratio
            ),
        )
        # A real axis center needs repeated orthogonal image evidence, but it does
        # not need a component at every eventual Cartesian intersection. Exact
        # support ratios are evaluated only after both maximal axis runs are known.
        row_centers = tuple(
            cluster.center for cluster in row_clusters if cluster.support >= 2
        )
        column_centers = tuple(
            cluster.center for cluster in column_clusters if cluster.support >= 2
        )
        row_runs = self._regular_center_runs(
            row_centers,
            minimum_count=self._settings.tile_grid_minimum_rows,
        )
        column_runs = self._regular_center_runs(
            column_centers,
            minimum_count=self._settings.tile_grid_minimum_columns,
        )
        return tuple(
            self._evaluate_lattice(screenshot, family, rows, columns)
            for rows in row_runs
            for columns in column_runs
        )

    @staticmethod
    def _cluster_centers(
        components: tuple[_TileComponent, ...],
        *,
        axis: str,
        tolerance: float,
    ) -> tuple[_CenterCluster, ...]:
        """Cluster aligned component centroids using tile-relative tolerance."""

        values = sorted(
            (
                component.diagnostic.center_x
                if axis == "x"
                else component.diagnostic.center_y
            )
            for component in components
        )
        groups: list[list[float]] = []
        for value in values:
            if not groups or abs(value - float(median(groups[-1]))) > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return tuple(
            _CenterCluster(center=float(median(group)), support=len(group))
            for group in groups
        )

    def _regular_center_runs(
        self,
        centers: tuple[float, ...],
        *,
        minimum_count: int,
    ) -> tuple[_CenterRun, ...]:
        """Enumerate bounded contiguous center sequences with regular pitch."""

        runs: list[_CenterRun] = []
        for start in range(len(centers)):
            for end in range(start + minimum_count, len(centers) + 1):
                selected = centers[start:end]
                pitches = tuple(
                    current - previous for previous, current in pairwise(selected)
                )
                if not pitches or any(pitch <= 0.0 for pitch in pitches):
                    continue
                pitch_cv = _coefficient_of_variation(pitches)
                if pitch_cv > self._settings.tile_pitch_cv_maximum:
                    continue
                runs.append(
                    _CenterRun(
                        centers=selected,
                        pitch=float(median(pitches)),
                        pitch_cv=pitch_cv,
                    )
                )
        runs.sort(key=lambda run: (-len(run.centers), run.pitch_cv, run.centers))
        if not runs:
            return ()
        maximal_count = len(runs[0].centers)
        return tuple(run for run in runs if len(run.centers) == maximal_count)[:64]

    def _evaluate_lattice(
        self,
        screenshot: Screenshot,
        family: _SizeFamily,
        rows: _CenterRun,
        columns: _CenterRun,
    ) -> _LatticeCandidate:
        """Assign components to unique slots and score independent grid evidence."""

        assigned: dict[tuple[int, int], _TileComponent] = {}
        residuals: list[float] = []
        duplicate_slot = False
        for component in family.components:
            row_index = min(
                range(len(rows.centers)),
                key=lambda index: (
                    abs(component.diagnostic.center_y - rows.centers[index]),
                    index,
                ),
            )
            column_index = min(
                range(len(columns.centers)),
                key=lambda index: (
                    abs(component.diagnostic.center_x - columns.centers[index]),
                    index,
                ),
            )
            x_residual = (
                abs(component.diagnostic.center_x - columns.centers[column_index])
                / columns.pitch
            )
            y_residual = (
                abs(component.diagnostic.center_y - rows.centers[row_index])
                / rows.pitch
            )
            if (
                x_residual > self._settings.tile_slot_residual_ratio
                or y_residual > self._settings.tile_slot_residual_ratio
            ):
                continue
            slot = (row_index, column_index)
            if slot in assigned:
                duplicate_slot = True
                continue
            assigned[slot] = component
            residuals.append((x_residual + y_residual) / 2.0)

        slot_count = len(rows.centers) * len(columns.centers)
        occupancy = len(assigned) / slot_count
        row_component_counts = tuple(
            sum(assigned_row == row for assigned_row, _ in assigned)
            for row in range(len(rows.centers))
        )
        column_component_counts = tuple(
            sum(assigned_column == column for _, assigned_column in assigned)
            for column in range(len(columns.centers))
        )
        row_support_ratios = tuple(
            count / len(columns.centers) for count in row_component_counts
        )
        column_support_ratios = tuple(
            count / len(rows.centers) for count in column_component_counts
        )
        minimum_row_support = min(row_support_ratios, default=0.0)
        minimum_column_support = min(column_support_ratios, default=0.0)
        missing_slots = tuple(
            (row, column)
            for row in range(len(rows.centers))
            for column in range(len(columns.centers))
            if (row, column) not in assigned
        )
        mean_residual = fmean(residuals) if residuals else 1.0
        size_consistency = _clamp_unit(
            1.0
            - max(family.width_cv, family.height_cv)
            / self._settings.tile_size_cv_maximum
        )
        row_regularity = _clamp_unit(
            1.0 - rows.pitch_cv / self._settings.tile_pitch_cv_maximum
        )
        column_regularity = _clamp_unit(
            1.0 - columns.pitch_cv / self._settings.tile_pitch_cv_maximum
        )
        residual_quality = _clamp_unit(
            1.0 - mean_residual / self._settings.tile_slot_residual_ratio
        )
        score = _clamp_unit(
            0.20 * size_consistency
            + 0.25 * row_regularity
            + 0.25 * column_regularity
            + 0.20 * occupancy
            + 0.10 * residual_quality
        )
        reasons: list[str] = []
        if duplicate_slot:
            reasons.append("multiple tile components were assigned to one slot")
        if occupancy < self._settings.tile_grid_minimum_occupancy_ratio:
            reasons.append("tile lattice occupancy was below threshold")
        if minimum_row_support < self._settings.tile_grid_minimum_row_support_ratio:
            reasons.append("a fitted row had insufficient real component support")
        if (
            minimum_column_support
            < self._settings.tile_grid_minimum_column_support_ratio
        ):
            reasons.append("a fitted column had insufficient real component support")
        horizontal_lines = self._boundaries(
            rows.centers,
            rows.pitch,
            screenshot.height,
        )
        vertical_lines = self._boundaries(
            columns.centers,
            columns.pitch,
            screenshot.width,
        )
        if horizontal_lines is None or vertical_lines is None:
            reasons.append(
                "tile lattice board extrapolation exceeded screenshot bounds"
            )
        if score < self._settings.tile_grid_minimum_score:
            reasons.append("tile lattice score was below threshold")
        return _LatticeCandidate(
            family=family,
            row_centers=rows.centers,
            column_centers=columns.centers,
            row_pitch=rows.pitch,
            column_pitch=columns.pitch,
            row_pitch_cv=rows.pitch_cv,
            column_pitch_cv=columns.pitch_cv,
            selected_component_indices=tuple(
                sorted(component.index for component in assigned.values())
            ),
            missing_slot_coordinates=missing_slots,
            row_component_counts=row_component_counts,
            column_component_counts=column_component_counts,
            row_support_ratios=row_support_ratios,
            column_support_ratios=column_support_ratios,
            minimum_row_support_ratio=_clamp_unit(minimum_row_support),
            minimum_column_support_ratio=_clamp_unit(minimum_column_support),
            occupancy_ratio=_clamp_unit(occupancy),
            mean_slot_residual=_clamp_unit(mean_residual),
            grid_score=score,
            horizontal_lines=horizontal_lines or (),
            vertical_lines=vertical_lines or (),
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _boundaries(
        centers: tuple[float, ...],
        pitch: float,
        screenshot_extent: int,
    ) -> tuple[int, ...] | None:
        """Extrapolate half-pitch outer bounds and midpoint internal boundaries."""

        floating = (
            centers[0] - pitch / 2.0,
            *(left + right for left, right in pairwise(centers)),
            centers[-1] + pitch / 2.0,
        )
        floating = (
            floating[0],
            *(value / 2.0 for value in floating[1:-1]),
            floating[-1],
        )
        if floating[0] < 0.0 or floating[-1] > screenshot_extent:
            return None
        boundaries = tuple(_round_pixel(value) for value in floating)
        if any(current <= previous for previous, current in pairwise(boundaries)):
            return None
        return boundaries

    def _grid_detection(self, candidate: _LatticeCandidate) -> GridDetection:
        """Convert a fitted lattice directly into exact row-major public cells."""

        cells = tuple(
            CellBounds(
                row=row,
                column=column,
                x=candidate.vertical_lines[column],
                y=candidate.horizontal_lines[row],
                width=(
                    candidate.vertical_lines[column + 1]
                    - candidate.vertical_lines[column]
                ),
                height=(
                    candidate.horizontal_lines[row + 1]
                    - candidate.horizontal_lines[row]
                ),
                center_x=min(
                    candidate.vertical_lines[column + 1] - 1,
                    max(
                        candidate.vertical_lines[column],
                        _round_pixel(candidate.column_centers[column]),
                    ),
                ),
                center_y=min(
                    candidate.horizontal_lines[row + 1] - 1,
                    max(
                        candidate.horizontal_lines[row],
                        _round_pixel(candidate.row_centers[row]),
                    ),
                ),
            )
            for row in range(len(candidate.row_centers))
            for column in range(len(candidate.column_centers))
        )
        return GridDetection(
            horizontal_lines=candidate.horizontal_lines,
            vertical_lines=candidate.vertical_lines,
            rows=len(candidate.row_centers),
            columns=len(candidate.column_centers),
            cells=cells,
            confidence=candidate.grid_score,
        )

    def _diagnostics(
        self,
        *,
        raw_component_count: int,
        components: tuple[_TileComponent, ...],
        candidate_tile_count: int,
        candidate: _LatticeCandidate | None,
        rejection_reasons: tuple[str, ...],
    ) -> CatsTileGridDiagnostics:
        """Translate internal fitting state into immutable primitive diagnostics."""

        selected = (
            set(candidate.selected_component_indices)
            if candidate is not None
            else set()
        )
        public_components = tuple(
            replace(
                component.diagnostic,
                accepted=component.index in selected and not rejection_reasons,
                rejection_reasons=(
                    component.diagnostic.rejection_reasons
                    if component.diagnostic.rejection_reasons
                    else (
                        ()
                        if component.index in selected and not rejection_reasons
                        else ("component was not selected by an accepted lattice",)
                    )
                ),
            )
            for component in components
        )
        family = candidate.family if candidate is not None else None
        return CatsTileGridDiagnostics(
            component_count=raw_component_count,
            candidate_tile_count=candidate_tile_count,
            selected_tile_count=(
                len(candidate.selected_component_indices)
                if candidate is not None and not rejection_reasons
                else 0
            ),
            row_count=len(candidate.row_centers) if candidate is not None else 0,
            column_count=(
                len(candidate.column_centers) if candidate is not None else 0
            ),
            row_centers=candidate.row_centers if candidate is not None else (),
            column_centers=candidate.column_centers if candidate is not None else (),
            horizontal_pitch=(candidate.column_pitch if candidate is not None else 0.0),
            vertical_pitch=(candidate.row_pitch if candidate is not None else 0.0),
            horizontal_pitch_cv=(
                _clamp_unit(candidate.column_pitch_cv) if candidate is not None else 0.0
            ),
            vertical_pitch_cv=(
                _clamp_unit(candidate.row_pitch_cv) if candidate is not None else 0.0
            ),
            median_tile_width=family.median_width if family is not None else 0.0,
            median_tile_height=family.median_height if family is not None else 0.0,
            tile_width_cv=(_clamp_unit(family.width_cv) if family is not None else 0.0),
            tile_height_cv=(
                _clamp_unit(family.height_cv) if family is not None else 0.0
            ),
            occupancy_ratio=(
                candidate.occupancy_ratio if candidate is not None else 0.0
            ),
            missing_slot_coordinates=(
                candidate.missing_slot_coordinates if candidate is not None else ()
            ),
            row_component_counts=(
                candidate.row_component_counts if candidate is not None else ()
            ),
            column_component_counts=(
                candidate.column_component_counts if candidate is not None else ()
            ),
            row_support_ratios=(
                candidate.row_support_ratios if candidate is not None else ()
            ),
            column_support_ratios=(
                candidate.column_support_ratios if candidate is not None else ()
            ),
            minimum_row_support_ratio=(
                candidate.minimum_row_support_ratio if candidate is not None else 0.0
            ),
            minimum_column_support_ratio=(
                candidate.minimum_column_support_ratio if candidate is not None else 0.0
            ),
            mean_slot_residual=(
                candidate.mean_slot_residual if candidate is not None else 0.0
            ),
            grid_score=candidate.grid_score if candidate is not None else 0.0,
            components=public_components,
            rejection_reasons=rejection_reasons,
        )

    @staticmethod
    def _failure_reasons(
        raw_component_count: int,
        candidate_tile_count: int,
        families: tuple[_SizeFamily, ...],
        best: _LatticeCandidate | None,
    ) -> tuple[str, ...]:
        """Build concise failure context without hiding the best lattice evidence."""

        if raw_component_count == 0:
            return ("tile chroma mask contained no connected components",)
        if candidate_tile_count == 0:
            return ("no connected component passed tile geometry",)
        if not families:
            return ("no stable tile-size family contained enough components",)
        if best is None:
            return ("no regular row and column center runs were found",)
        return best.rejection_reasons or ("no tile lattice passed mandatory evidence",)

    @staticmethod
    def _lattice_sort_key(
        candidate: _LatticeCandidate,
    ) -> tuple[int, float, int, float, float, float, float, float, float]:
        """Order candidates by maximal supported lattice and explicit evidence."""

        return (
            -(len(candidate.row_centers) * len(candidate.column_centers)),
            -min(
                candidate.minimum_row_support_ratio,
                candidate.minimum_column_support_ratio,
            ),
            -len(candidate.selected_component_indices),
            -candidate.occupancy_ratio,
            -candidate.grid_score,
            candidate.row_pitch_cv + candidate.column_pitch_cv,
            candidate.mean_slot_residual,
            candidate.row_centers[0],
            candidate.column_centers[0],
        )
