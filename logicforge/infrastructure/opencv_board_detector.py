"""Classical OpenCV adapter for deterministic rectangular board localization."""

from collections import Counter
from dataclasses import dataclass, replace
from math import hypot
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.opencv_grid_envelope_refinement import (
    OpenCvGridEnvelopeRefiner,
)
from logicforge.infrastructure.opencv_internal_grid_evidence import (
    InternalGridEvidence,
    OpenCvInternalGridEvidenceAnalyzer,
)
from logicforge.vision.board_detector import (
    BoardCandidateDiagnostic,
    BoardDetection,
    BoardDetectionAnalysis,
    BoardDetectionDiagnostics,
    BoardDetectionError,
    BoardDetector,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.screenshot import Screenshot


def _clamp_unit(value: float) -> float:
    """Clamp a floating-point scoring component into the inclusive unit interval."""

    return max(0.0, min(1.0, value))


def _triangular_score(
    value: float,
    minimum: float,
    preferred: float,
    maximum: float,
) -> float:
    """Score a value linearly toward one preferred point within accepted bounds."""

    if value <= minimum or value >= maximum:
        return 0.0
    if value <= preferred:
        return _clamp_unit((value - minimum) / (preferred - minimum))
    return _clamp_unit((maximum - value) / (maximum - preferred))


def _aspect_ratio_score(aspect_ratio: float, minimum: float, maximum: float) -> float:
    """Score rectangular aspect ratio with a peak at a square ratio of one."""

    if aspect_ratio <= 1.0:
        return _clamp_unit((aspect_ratio - minimum) / (1.0 - minimum))
    return _clamp_unit((maximum - aspect_ratio) / (maximum - 1.0))


def _intersection_over_union(
    first: BoardCandidateDiagnostic,
    second: BoardCandidateDiagnostic,
) -> float:
    """Measure bounding-box overlap for deterministic duplicate suppression."""

    intersection_left = max(first.x, second.x)
    intersection_top = max(first.y, second.y)
    intersection_right = min(first.x + first.width, second.x + second.width)
    intersection_bottom = min(first.y + first.height, second.y + second.height)
    intersection_width = max(0, intersection_right - intersection_left)
    intersection_height = max(0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height
    first_area = first.width * first.height
    second_area = second.width * second.height
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area else 0.0


@dataclass(frozen=True, slots=True)
class _BoardSeedCandidate:
    """Keep one contour diagnostic paired with its exact internal grid evidence."""

    diagnostic: BoardCandidateDiagnostic
    grid_evidence: InternalGridEvidence


@dataclass(frozen=True, slots=True)
class _BoardFamilyCandidate:
    """Represent one seed's verified maximal envelope for final selection."""

    seed: _BoardSeedCandidate
    x: int
    y: int
    width: int
    height: int
    grid_evidence: InternalGridEvidence
    confidence: float
    refinement: BoardEnvelopeRefinementDiagnostic | None


class OpenCvBoardDetector(BoardDetector):
    """Locate a puzzle board through edges, thresholding, contours, and scoring.

    The geometry subscore is ``0.25 * area + 0.25 * rectangularity +
    0.20 * aspect ratio + 0.15 * edge density + 0.15 * location``. Final confidence
    is ``0.40 * geometry + 0.60 * grid evidence`` by default. Mandatory grid checks
    are evaluated before confidence, so geometry can never authorize a board alone.
    """

    def __init__(self, settings: BoardDetectionSettings | None = None) -> None:
        """Receive all thresholds through one immutable typed settings record."""

        self._settings = settings or BoardDetectionSettings()
        self._grid_analyzer = OpenCvInternalGridEvidenceAnalyzer(self._settings)
        self._envelope_refiner = OpenCvGridEnvelopeRefiner(
            self._settings,
            self._grid_analyzer,
        )

    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Return the highest-confidence reliable rectangle."""

        return self.analyze(screenshot).detection

    def analyze(self, screenshot: Screenshot) -> BoardDetectionAnalysis:
        """Detect a board and retain deterministic candidate diagnostics."""

        grayscale = cast(
            NDArray[np.uint8],
            cv2.cvtColor(screenshot.image, cv2.COLOR_BGR2GRAY),
        )
        kernel_size = self._settings.gaussian_blur_kernel_size
        blurred = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
        edges = cast(
            NDArray[np.uint8],
            cv2.Canny(
                blurred,
                self._settings.canny_lower_threshold,
                self._settings.canny_upper_threshold,
            ),
        )
        _, thresholded = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )

        morphology_kernel = self._create_morphology_kernel(
            screenshot.width,
            screenshot.height,
        )
        edge_envelope_kernel = self._create_relative_kernel(
            screenshot.width,
            screenshot.height,
            self._settings.edge_envelope_kernel_relative_size,
        )
        edge_envelope = cv2.morphologyEx(
            cv2.dilate(
                edges,
                edge_envelope_kernel,
                iterations=self._settings.edge_envelope_iterations,
            ),
            cv2.MORPH_CLOSE,
            edge_envelope_kernel,
            iterations=self._settings.edge_envelope_iterations,
        )
        masks = (
            cv2.morphologyEx(
                edges,
                cv2.MORPH_CLOSE,
                morphology_kernel,
                iterations=self._settings.morphology_iterations,
            ),
            cv2.morphologyEx(
                thresholded,
                cv2.MORPH_CLOSE,
                morphology_kernel,
                iterations=self._settings.morphology_iterations,
            ),
            edge_envelope,
        )

        contours: list[cv2.typing.MatLike] = []
        for mask in masks:
            mask_contours, _ = cv2.findContours(
                mask,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            contours.extend(mask_contours)

        seed_candidates = [
            seed
            for contour in contours
            if (
                seed := self._evaluate_contour(
                    contour,
                    grayscale,
                    edges,
                    screenshot.width,
                    screenshot.height,
                )
            )
            is not None
        ]
        deduplicated_seeds = self._suppress_duplicates(seed_candidates)
        families, refinements = self._refine_seed_families(
            grayscale,
            deduplicated_seeds,
        )
        return self._select_detection(
            contour_count=len(contours),
            seeds=deduplicated_seeds,
            families=families,
            refinements=refinements,
        )

    def _create_morphology_kernel(self, width: int, height: int) -> NDArray[np.uint8]:
        """Create an odd scale-relative closing kernel for the current resolution."""

        return self._create_relative_kernel(
            width,
            height,
            self._settings.morphology_kernel_relative_size,
        )

    @staticmethod
    def _create_relative_kernel(
        width: int,
        height: int,
        relative_kernel_size: float,
    ) -> NDArray[np.uint8]:
        """Build one odd rectangular kernel from a typed scale-relative setting."""

        relative_size = min(width, height) * relative_kernel_size
        kernel_size = max(3, round(relative_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cast(
            NDArray[np.uint8],
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (kernel_size, kernel_size),
            ),
        )

    def _evaluate_contour(
        self,
        contour: cv2.typing.MatLike,
        grayscale: NDArray[np.uint8],
        edges: NDArray[np.uint8],
        screenshot_width: int,
        screenshot_height: int,
    ) -> _BoardSeedCandidate | None:
        """Convert one sufficiently large contour into measurements and decisions."""

        x, y, width, height = cv2.boundingRect(contour)
        screenshot_area = screenshot_width * screenshot_height
        bounding_area = width * height
        diagnostic_area_floor = (
            screenshot_area * self._settings.minimum_relative_area * 0.25
        )
        if bounding_area < diagnostic_area_floor:
            return None

        relative_area = bounding_area / screenshot_area
        aspect_ratio = width / height
        contour_area = abs(cv2.contourArea(contour))
        rectangularity = _clamp_unit(contour_area / bounding_area)
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour,
            self._settings.polygon_epsilon_ratio * perimeter,
            True,
        )
        edge_region = edges[y : y + height, x : x + width]
        edge_density = cv2.countNonZero(edge_region) / bounding_area
        location_score, inside_content = self._location_score(
            x,
            y,
            width,
            height,
            screenshot_width,
            screenshot_height,
        )

        rejection_reasons: list[str] = []
        if not (
            self._settings.minimum_relative_area
            <= relative_area
            <= self._settings.maximum_relative_area
        ):
            rejection_reasons.append("relative area outside configured range")
        if not (
            self._settings.minimum_aspect_ratio
            <= aspect_ratio
            <= self._settings.maximum_aspect_ratio
        ):
            rejection_reasons.append("implausible aspect ratio")
        if rectangularity < self._settings.minimum_rectangularity:
            rejection_reasons.append("insufficient rectangularity")
        if edge_density < self._settings.minimum_edge_density:
            rejection_reasons.append("insufficient edge strength")
        if not inside_content:
            rejection_reasons.append("outside BlueStacks content area")
        if len(approximation) < 4:
            rejection_reasons.append("contour is not a rectangular polygon")

        geometry_score = self._geometry_score(
            relative_area=relative_area,
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            edge_density=edge_density,
            location_score=location_score,
        )
        grid_evidence = self._empty_grid_evidence()
        if not rejection_reasons:
            grayscale_roi = grayscale[y : y + height, x : x + width]
            grid_evidence = self._grid_analyzer.analyze(grayscale_roi)
            rejection_reasons.extend(
                self._grid_analyzer.rejection_reasons(grid_evidence)
            )
        confidence = self._final_confidence(geometry_score, grid_evidence.score)
        diagnostic = BoardCandidateDiagnostic(
            x=x,
            y=y,
            width=width,
            height=height,
            relative_area=relative_area,
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            edge_density=edge_density,
            location_score=location_score,
            geometry_score=geometry_score,
            horizontal_grid_line_positions=(grid_evidence.horizontal_line_positions),
            vertical_grid_line_positions=grid_evidence.vertical_line_positions,
            horizontal_grid_line_count=grid_evidence.horizontal_line_count,
            vertical_grid_line_count=grid_evidence.vertical_line_count,
            estimated_rows=grid_evidence.estimated_rows,
            estimated_columns=grid_evidence.estimated_columns,
            horizontal_spacing_coefficient_of_variation=(
                grid_evidence.horizontal_spacing_coefficient_of_variation
            ),
            vertical_spacing_coefficient_of_variation=(
                grid_evidence.vertical_spacing_coefficient_of_variation
            ),
            horizontal_spacing_regularity=(grid_evidence.horizontal_spacing_regularity),
            vertical_spacing_regularity=grid_evidence.vertical_spacing_regularity,
            horizontal_line_coverage=grid_evidence.horizontal_line_coverage,
            vertical_line_coverage=grid_evidence.vertical_line_coverage,
            grid_evidence_score=grid_evidence.score,
            confidence=confidence,
            accepted=not rejection_reasons,
            rejection_reasons=tuple(rejection_reasons),
        )
        return _BoardSeedCandidate(
            diagnostic=diagnostic,
            grid_evidence=grid_evidence,
        )

    def _location_score(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        screenshot_width: int,
        screenshot_height: int,
    ) -> tuple[float, bool]:
        """Score center proximity and reject title, toolbar, border, or clipped UI."""

        left = screenshot_width * self._settings.border_exclusion
        top = screenshot_height * max(
            self._settings.border_exclusion,
            self._settings.top_content_exclusion,
        )
        right = screenshot_width * (
            1.0
            - max(
                self._settings.border_exclusion,
                self._settings.right_toolbar_exclusion,
            )
        )
        bottom = screenshot_height * (
            1.0
            - max(
                self._settings.border_exclusion,
                self._settings.bottom_content_exclusion,
            )
        )
        inside_content = (
            x >= left and y >= top and x + width <= right and y + height <= bottom
        )

        candidate_center_x = (x + width / 2.0) / screenshot_width
        candidate_center_y = (y + height / 2.0) / screenshot_height
        distance = hypot(
            candidate_center_x - self._settings.expected_center_x,
            candidate_center_y - self._settings.expected_center_y,
        )
        maximum_distance = hypot(1.0, 1.0)
        return _clamp_unit(1.0 - distance / maximum_distance), inside_content

    def _geometry_score(
        self,
        *,
        relative_area: float,
        aspect_ratio: float,
        rectangularity: float,
        edge_density: float,
        location_score: float,
    ) -> float:
        """Combine normalized geometric evidence into the documented subscore."""

        area_score = _triangular_score(
            relative_area,
            self._settings.minimum_relative_area,
            self._settings.preferred_relative_area,
            self._settings.maximum_relative_area,
        )
        rectangularity_score = _clamp_unit(
            (rectangularity - self._settings.minimum_rectangularity)
            / (1.0 - self._settings.minimum_rectangularity)
        )
        aspect_score = _aspect_ratio_score(
            aspect_ratio,
            self._settings.minimum_aspect_ratio,
            self._settings.maximum_aspect_ratio,
        )
        edge_score = _triangular_score(
            edge_density,
            self._settings.minimum_edge_density,
            self._settings.preferred_edge_density,
            self._settings.maximum_edge_density,
        )
        geometry_score = (
            0.25 * area_score
            + 0.25 * rectangularity_score
            + 0.20 * aspect_score
            + 0.15 * edge_score
            + 0.15 * location_score
        )
        return _clamp_unit(geometry_score)

    def _final_confidence(self, geometry_score: float, grid_score: float) -> float:
        """Weight mandatory grid evidence at least as strongly as geometry evidence."""

        return _clamp_unit(
            self._settings.geometry_confidence_weight * geometry_score
            + self._settings.grid_confidence_weight * grid_score
        )

    @staticmethod
    def _empty_grid_evidence() -> InternalGridEvidence:
        """Represent intentionally skipped analysis for geometry-invalid candidates."""

        return InternalGridEvidence(
            horizontal_line_positions=(),
            vertical_line_positions=(),
            horizontal_line_count=0,
            vertical_line_count=0,
            estimated_rows=0,
            estimated_columns=0,
            horizontal_spacing_coefficient_of_variation=1.0,
            vertical_spacing_coefficient_of_variation=1.0,
            horizontal_spacing_regularity=0.0,
            vertical_spacing_regularity=0.0,
            horizontal_line_coverage=0.0,
            vertical_line_coverage=0.0,
            score=0.0,
        )

    def _suppress_duplicates(
        self,
        candidates: list[_BoardSeedCandidate],
    ) -> tuple[_BoardSeedCandidate, ...]:
        """Mark overlapping contour duplicates while preserving diagnostic evidence."""

        ordered = sorted(
            candidates, key=lambda seed: self._candidate_sort_key(seed.diagnostic)
        )
        accepted_unique: list[BoardCandidateDiagnostic] = []
        results: list[_BoardSeedCandidate] = []
        for seed in ordered:
            candidate = seed.diagnostic
            is_duplicate = candidate.accepted and any(
                _intersection_over_union(candidate, accepted)
                >= self._settings.duplicate_iou_threshold
                for accepted in accepted_unique
            )
            if is_duplicate:
                results.append(
                    replace(
                        seed,
                        diagnostic=replace(
                            candidate,
                            accepted=False,
                            rejection_reasons=(
                                *candidate.rejection_reasons,
                                "duplicate candidate geometry",
                            ),
                        ),
                    )
                )
                continue

            results.append(seed)
            if candidate.accepted:
                accepted_unique.append(candidate)
        return tuple(results)

    def _refine_seed_families(
        self,
        grayscale: NDArray[np.uint8],
        seeds: tuple[_BoardSeedCandidate, ...],
    ) -> tuple[
        tuple[_BoardFamilyCandidate, ...],
        tuple[BoardEnvelopeRefinementDiagnostic, ...],
    ]:
        """Replace each accepted seed with its verified maximal family envelope."""

        families: list[_BoardFamilyCandidate] = []
        refinements: list[BoardEnvelopeRefinementDiagnostic] = []
        for seed in seeds:
            diagnostic = seed.diagnostic
            if not diagnostic.accepted:
                continue
            result = self._envelope_refiner.refine(
                grayscale,
                diagnostic,
                seed.grid_evidence,
            )
            refinements.extend(result.diagnostics)
            selected_refinement = result.selected_diagnostic
            selected_evidence = result.selected_grid_evidence
            if selected_refinement is None or selected_evidence is None:
                x, y, width, height = (
                    diagnostic.x,
                    diagnostic.y,
                    diagnostic.width,
                    diagnostic.height,
                )
                grid_evidence = seed.grid_evidence
            else:
                x, y, width, height = (
                    selected_refinement.refined_x,
                    selected_refinement.refined_y,
                    selected_refinement.refined_width,
                    selected_refinement.refined_height,
                )
                grid_evidence = selected_evidence
            families.append(
                _BoardFamilyCandidate(
                    seed=seed,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    grid_evidence=grid_evidence,
                    confidence=self._final_confidence(
                        diagnostic.geometry_score,
                        grid_evidence.score,
                    ),
                    refinement=selected_refinement,
                )
            )
        return self._suppress_family_duplicates(families), tuple(refinements)

    def _suppress_family_duplicates(
        self,
        families: list[_BoardFamilyCandidate],
    ) -> tuple[_BoardFamilyCandidate, ...]:
        """Collapse refined and contour-derived copies of the same final envelope."""

        ordered = sorted(families, key=self._family_sort_key)
        unique: list[_BoardFamilyCandidate] = []
        for family in ordered:
            if any(
                self._family_intersection_over_union(family, accepted)
                >= self._settings.duplicate_iou_threshold
                for accepted in unique
            ):
                continue
            unique.append(family)
        return tuple(unique)

    def _select_detection(
        self,
        *,
        contour_count: int,
        seeds: tuple[_BoardSeedCandidate, ...],
        families: tuple[_BoardFamilyCandidate, ...],
        refinements: tuple[BoardEnvelopeRefinementDiagnostic, ...],
    ) -> BoardDetectionAnalysis:
        """Select the deterministic winner or raise an actionable typed error."""

        candidates = tuple(seed.diagnostic for seed in seeds)
        top_family = families[0] if families else None
        competitive_count = (
            sum(
                top_family.confidence - family.confidence
                <= self._settings.ambiguity_score_delta
                for family in families
            )
            if top_family is not None
            else 0
        )
        diagnostics = BoardDetectionDiagnostics(
            contour_count=contour_count,
            candidates=candidates,
            selected_candidate=(
                top_family.seed.diagnostic if top_family is not None else None
            ),
            competitive_candidate_count=competitive_count,
            envelope_refinements=refinements,
            selected_refinement=(
                top_family.refinement if top_family is not None else None
            ),
        )

        if top_family is None:
            reason_counts = Counter(
                reason
                for candidate in candidates
                for reason in candidate.rejection_reasons
            )
            reason_summary = ", ".join(
                f"{reason}: {count}" for reason, count in sorted(reason_counts.items())
            )
            raise BoardDetectionError(
                "No board candidate passed geometry and mandatory grid validation. "
                f"Contours: {contour_count}; retained candidates: {len(candidates)}; "
                f"rejections: {reason_summary or 'none above diagnostic area floor'}.",
                diagnostics,
            )
        if top_family.confidence < self._settings.minimum_confidence:
            raise BoardDetectionError(
                "Best board candidate was below the confidence threshold: "
                f"{top_family.confidence:.3f} < "
                f"{self._settings.minimum_confidence:.3f}. "
                f"Competitive candidates: {competitive_count}.",
                diagnostics,
            )

        detection = BoardDetection(
            x=top_family.x,
            y=top_family.y,
            width=top_family.width,
            height=top_family.height,
            confidence=top_family.confidence,
        )
        return BoardDetectionAnalysis(detection=detection, diagnostics=diagnostics)

    @staticmethod
    def _family_intersection_over_union(
        first: _BoardFamilyCandidate,
        second: _BoardFamilyCandidate,
    ) -> float:
        """Measure final family envelopes without exposing a synthetic contour."""

        intersection_left = max(first.x, second.x)
        intersection_top = max(first.y, second.y)
        intersection_right = min(first.x + first.width, second.x + second.width)
        intersection_bottom = min(first.y + first.height, second.y + second.height)
        intersection_width = max(0, intersection_right - intersection_left)
        intersection_height = max(0, intersection_bottom - intersection_top)
        intersection_area = intersection_width * intersection_height
        union_area = (
            first.width * first.height
            + second.width * second.height
            - intersection_area
        )
        return intersection_area / union_area if union_area else 0.0

    @staticmethod
    def _family_sort_key(
        family: _BoardFamilyCandidate,
    ) -> tuple[float, int, int, int, int, int]:
        """Preserve confidence selection while preferring maximal verified ties."""

        return (
            -family.confidence,
            -(
                family.grid_evidence.estimated_rows
                * family.grid_evidence.estimated_columns
            ),
            -(family.width * family.height),
            family.y,
            family.x,
            -family.width,
        )

    @staticmethod
    def _candidate_sort_key(
        candidate: BoardCandidateDiagnostic,
    ) -> tuple[float, float, int, int, int, int]:
        """Provide total deterministic ordering for tied or near-tied candidates."""

        return (
            -candidate.confidence,
            -candidate.relative_area,
            candidate.y,
            candidate.x,
            -candidate.width,
            -candidate.height,
        )
