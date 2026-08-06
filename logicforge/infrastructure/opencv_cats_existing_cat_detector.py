"""Classical OpenCV detection of Cats already present inside fitted cells."""

from collections import Counter
from itertools import combinations
from math import floor, hypot

import cv2
import numpy as np

from logicforge.config.settings import CatsExistingCatDetectionSettings
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatCellDiagnostic,
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
    CatsExistingCatDetector,
    CatsExistingCatDiagnostics,
    CatsExistingCatObservation,
)
from logicforge.vision.color_detector import ColorDetectionResult, ColorObservation
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot


def _round_pixel(value: float) -> int:
    """Round non-negative scale-relative geometry deterministically."""

    return floor(value + 0.5)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


class OpenCvCatsExistingCatDetector(CatsExistingCatDetector):
    """Find large central LAB foreground components per public CellBounds only."""

    def __init__(
        self,
        settings: CatsExistingCatDetectionSettings | None = None,
    ) -> None:
        self._settings = settings or CatsExistingCatDetectionSettings()

    def detect(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        colors: ColorDetectionResult,
    ) -> CatsExistingCatDetection:
        """Measure every cell once and fail closed on contract or Cats conflicts."""

        self._validate_inputs(screenshot, grid, colors)
        diagnostics = tuple(
            self._measure_cell(screenshot, cell, colors.observations[index])
            for index, cell in enumerate(grid.cells)
        )
        cats = tuple(
            CatsExistingCatObservation(
                row=cell.row,
                column=cell.column,
                confidence=cell.score,
            )
            for cell in diagnostics
            if cell.accepted
        )
        self._validate_cats(cats, colors, diagnostics)
        return CatsExistingCatDetection(
            cats=cats,
            diagnostics=CatsExistingCatDiagnostics(cells=diagnostics),
        )

    def _validate_inputs(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        colors: ColorDetectionResult,
    ) -> None:
        reasons: list[str] = []
        if (
            colors.diagnostics.rows != grid.rows
            or colors.diagnostics.columns != grid.columns
        ):
            reasons.append("color result dimensions do not match the fitted grid")
        if len(colors.observations) != grid.rows * grid.columns:
            reasons.append("color observations do not cover every fitted cell")
        for index, cell in enumerate(grid.cells):
            expected = divmod(index, grid.columns)
            if (cell.row, cell.column) != expected:
                reasons.append("grid cells are not in row-major order")
                break
            if (
                cell.x + cell.width > screenshot.width
                or cell.y + cell.height > screenshot.height
            ):
                reasons.append("a fitted CellBounds exceeds screenshot bounds")
                break
            if index < len(colors.observations):
                observation = colors.observations[index]
                if (observation.row, observation.column) != (cell.row, cell.column):
                    reasons.append(
                        "color observation coordinates do not match CellBounds"
                    )
                    break
        if reasons:
            diagnostics = CatsExistingCatDiagnostics(
                cells=(),
                rejection_reasons=tuple(dict.fromkeys(reasons)),
            )
            raise CatsExistingCatDetectionError(
                "Malformed existing-cat detection inputs: "
                + "; ".join(diagnostics.rejection_reasons),
                diagnostics,
            )

    def _measure_cell(
        self,
        screenshot: Screenshot,
        cell: CellBounds,
        observation: ColorObservation,
    ) -> CatsExistingCatCellDiagnostic:
        # The caller validated the row-major ColorObservation contract. Keeping
        # this access local makes the pixel backend independent of color clustering.
        representative_lab = observation.representative_lab
        horizontal_inset = _round_pixel(
            cell.width * self._settings.cat_roi_horizontal_inset_ratio
        )
        vertical_inset = _round_pixel(
            cell.height * self._settings.cat_roi_vertical_inset_ratio
        )
        left = cell.x + horizontal_inset
        right = cell.x + cell.width - horizontal_inset
        top = cell.y + vertical_inset
        bottom = cell.y + cell.height - vertical_inset
        roi = screenshot.image[top:bottom, left:right]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        background = np.asarray(representative_lab, dtype=np.float32)
        distances = np.linalg.norm(lab - background, axis=2)
        mask = np.where(
            distances >= self._settings.cat_foreground_lab_distance_threshold,
            np.uint8(255),
            np.uint8(0),
        )
        kernel_size = max(
            1,
            _round_pixel(
                min(cell.width, cell.height)
                * self._settings.cat_mask_kernel_relative_size
            ),
        )
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        roi_height, roi_width = mask.shape
        roi_area = roi_width * roi_height
        foreground_ratio = float(np.count_nonzero(mask) / roi_area)
        label_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if label_count <= 1:
            largest_area = width = height = 0
            component_center_x = roi_width / 2.0
            component_center_y = roi_height / 2.0
        else:
            largest_label = max(
                range(1, label_count),
                key=lambda label: (int(stats[label, cv2.CC_STAT_AREA]), -label),
            )
            largest_area = int(stats[largest_label, cv2.CC_STAT_AREA])
            width = int(stats[largest_label, cv2.CC_STAT_WIDTH])
            height = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
            component_center_x = float(centroids[largest_label, 0])
            component_center_y = float(centroids[largest_label, 1])
        largest_ratio = largest_area / roi_area
        width_ratio = width / roi_width
        height_ratio = height / roi_height
        half_diagonal = hypot(roi_width, roi_height) / 2.0
        center_offset = _clamp_unit(
            hypot(
                component_center_x - (roi_width - 1) / 2.0,
                component_center_y - (roi_height - 1) / 2.0,
            )
            / half_diagonal
        )
        score = _clamp_unit(
            0.25 * foreground_ratio
            + 0.25 * largest_ratio
            + 0.20 * width_ratio
            + 0.20 * height_ratio
            + 0.10 * (1.0 - center_offset)
        )
        reasons: list[str] = []
        if foreground_ratio < self._settings.cat_minimum_foreground_ratio:
            reasons.append("foreground ratio below cat minimum")
        if largest_ratio < self._settings.cat_minimum_largest_component_ratio:
            reasons.append("largest component ratio below cat minimum")
        if width_ratio < self._settings.cat_minimum_component_width_ratio:
            reasons.append("component width ratio below cat minimum")
        if height_ratio < self._settings.cat_minimum_component_height_ratio:
            reasons.append("component height ratio below cat minimum")
        if center_offset > self._settings.cat_maximum_center_offset_ratio:
            reasons.append("component center offset exceeded cat maximum")
        if score < self._settings.cat_minimum_score:
            reasons.append("cat score below minimum")
        return CatsExistingCatCellDiagnostic(
            row=cell.row,
            column=cell.column,
            roi_x=left,
            roi_y=top,
            roi_width=roi_width,
            roi_height=roi_height,
            foreground_ratio=_clamp_unit(foreground_ratio),
            largest_component_ratio=_clamp_unit(largest_ratio),
            component_width_ratio=_clamp_unit(width_ratio),
            component_height_ratio=_clamp_unit(height_ratio),
            center_offset_ratio=center_offset,
            score=score,
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _validate_cats(
        cats: tuple[CatsExistingCatObservation, ...],
        colors: ColorDetectionResult,
        cells: tuple[CatsExistingCatCellDiagnostic, ...],
    ) -> None:
        reasons: list[str] = []
        row_counts = Counter(cat.row for cat in cats)
        column_counts = Counter(cat.column for cat in cats)
        if any(count > 1 for count in row_counts.values()):
            reasons.append("multiple existing cats were detected in one row")
        if any(count > 1 for count in column_counts.values()):
            reasons.append("multiple existing cats were detected in one column")
        color_counts = Counter(colors.color_matrix[cat.row][cat.column] for cat in cats)
        if any(count > 1 for count in color_counts.values()):
            reasons.append("multiple existing cats were detected on one original color")
        if any(
            max(abs(first.row - second.row), abs(first.column - second.column)) <= 1
            for first, second in combinations(cats, 2)
        ):
            reasons.append("detected existing cats touch orthogonally or diagonally")
        if reasons:
            diagnostics = CatsExistingCatDiagnostics(
                cells=cells,
                rejection_reasons=tuple(reasons),
            )
            raise CatsExistingCatDetectionError(
                "Contradictory existing-cat evidence: " + "; ".join(reasons),
                diagnostics,
            )
