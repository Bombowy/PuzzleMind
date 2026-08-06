"""Conservative maximal regular-grid envelope refinement for board seeds."""

from dataclasses import dataclass, replace
from itertools import pairwise
from statistics import fmean, median
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.opencv_internal_grid_evidence import (
    InternalGridEvidence,
    InternalGridEvidenceAnalyzer,
)
from logicforge.vision.board_detector import (
    BoardCandidateDiagnostic,
    BoardEnvelopeRefinementDiagnostic,
)


def _clamp_unit(value: float) -> float:
    """Clamp one refinement metric into its public unit interval."""

    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class _GridEnvelopeRefinementCandidate:
    """Retain backend evidence needed to select one extension deterministically."""

    diagnostic: BoardEnvelopeRefinementDiagnostic
    refined_grid_evidence: InternalGridEvidence | None
    added_size_error: float
    combined_spacing_cv: float


@dataclass(frozen=True, slots=True)
class GridEnvelopeRefinementResult:
    """Return primitive diagnostics plus an optional accepted internal evidence."""

    diagnostics: tuple[BoardEnvelopeRefinementDiagnostic, ...]
    selected_diagnostic: BoardEnvelopeRefinementDiagnostic | None
    selected_grid_evidence: InternalGridEvidence | None


class OpenCvGridEnvelopeRefiner:
    """Extend a credible contour seed by one image-supported regular cell band."""

    _DIRECTIONS: ClassVar[tuple[str, ...]] = ("left", "right", "top", "bottom")
    _DIRECTION_ORDER: ClassVar[dict[str, int]] = {
        direction: index for index, direction in enumerate(_DIRECTIONS)
    }

    def __init__(
        self,
        settings: BoardDetectionSettings,
        analyzer: InternalGridEvidenceAnalyzer,
    ) -> None:
        """Reuse exactly the analyzer instance that validated contour seed ROIs."""

        self._settings = settings
        self._analyzer = analyzer

    def refine(
        self,
        grayscale: NDArray[np.uint8],
        seed: BoardCandidateDiagnostic,
        seed_evidence: InternalGridEvidence,
    ) -> GridEnvelopeRefinementResult:
        """Evaluate four bounded extensions and fail closed on close alternatives."""

        typical_height = self._typical_cell_size(
            seed_evidence.horizontal_line_positions,
            seed.height,
        )
        typical_width = self._typical_cell_size(
            seed_evidence.vertical_line_positions,
            seed.width,
        )
        skip_reasons: list[str] = []
        if (
            not self._settings.grid_envelope_refinement_enabled
            or self._settings.grid_envelope_maximum_added_cells_per_side == 0
        ):
            skip_reasons.append("refinement disabled")
        if (
            seed_evidence.estimated_rows
            < self._settings.grid_envelope_minimum_seed_rows
        ):
            skip_reasons.append("seed row count below minimum")
        if (
            seed_evidence.estimated_columns
            < self._settings.grid_envelope_minimum_seed_columns
        ):
            skip_reasons.append("seed column count below minimum")
        if typical_width < 2.0:
            skip_reasons.append("typical cell width was invalid")
        if typical_height < 2.0:
            skip_reasons.append("typical cell height was invalid")
        if seed_evidence.horizontal_spacing_coefficient_of_variation > (
            self._settings.maximum_horizontal_spacing_coefficient_of_variation
        ):
            skip_reasons.append("seed horizontal spacing CV exceeded maximum")
        if seed_evidence.vertical_spacing_coefficient_of_variation > (
            self._settings.maximum_vertical_spacing_coefficient_of_variation
        ):
            skip_reasons.append("seed vertical spacing CV exceeded maximum")
        if skip_reasons:
            return GridEnvelopeRefinementResult(
                self._skipped_diagnostics(
                    seed,
                    typical_cell_width=typical_width,
                    typical_cell_height=typical_height,
                    rejection_reasons=tuple(skip_reasons),
                ),
                None,
                None,
            )

        candidates = tuple(
            self._evaluate_direction(
                grayscale,
                seed,
                seed_evidence,
                direction=direction,
                typical_cell_width=typical_width,
                typical_cell_height=typical_height,
            )
            for direction in self._DIRECTIONS
        )
        accepted = tuple(
            candidate for candidate in candidates if candidate.diagnostic.accepted
        )
        if not accepted:
            return GridEnvelopeRefinementResult(
                tuple(candidate.diagnostic for candidate in candidates),
                None,
                None,
            )

        ordered = sorted(accepted, key=self._selection_key)
        best = ordered[0]
        ambiguous = tuple(
            candidate
            for candidate in ordered[1:]
            if (
                candidate.diagnostic.direction != best.diagnostic.direction
                and abs(
                    candidate.diagnostic.refinement_score
                    - best.diagnostic.refinement_score
                )
                <= self._settings.grid_envelope_ambiguity_delta
            )
        )
        if ambiguous:
            ambiguous_directions = tuple(
                candidate.diagnostic.direction for candidate in (best, *ambiguous)
            )
            diagnostics = tuple(
                (
                    replace(
                        candidate.diagnostic,
                        accepted=False,
                        rejection_reasons=(
                            *candidate.diagnostic.rejection_reasons,
                            "ambiguous grid-envelope refinements: "
                            + ", ".join(ambiguous_directions),
                        ),
                    )
                    if candidate in (best, *ambiguous)
                    else candidate.diagnostic
                )
                for candidate in candidates
            )
            return GridEnvelopeRefinementResult(diagnostics, None, None)

        return GridEnvelopeRefinementResult(
            tuple(candidate.diagnostic for candidate in candidates),
            best.diagnostic,
            best.refined_grid_evidence,
        )

    def _skipped_diagnostics(
        self,
        seed: BoardCandidateDiagnostic,
        *,
        typical_cell_width: float,
        typical_cell_height: float,
        rejection_reasons: tuple[str, ...],
    ) -> tuple[BoardEnvelopeRefinementDiagnostic, ...]:
        """Explain every direction skipped before image-candidate evaluation."""

        diagnostics: list[BoardEnvelopeRefinementDiagnostic] = []
        for direction in self._DIRECTIONS:
            typical_size = (
                typical_cell_width
                if direction in {"left", "right"}
                else typical_cell_height
            )
            added_pixels = round(typical_size) if typical_size >= 2.0 else 0
            x, y, width, height = self._extension_geometry(
                seed,
                direction=direction,
                added_pixels=added_pixels,
            )
            candidate = self._rejected_candidate(
                seed,
                direction=direction,
                x=x,
                y=y,
                width=width,
                height=height,
                added_pixels=added_pixels,
                added_size_error=(
                    abs(added_pixels / typical_size - 1.0)
                    if typical_size >= 2.0
                    else 1.0
                ),
                rejection_reasons=rejection_reasons,
            )
            diagnostics.append(candidate.diagnostic)
        return tuple(diagnostics)

    @staticmethod
    def _typical_cell_size(
        positions: tuple[float, ...],
        extent: int,
    ) -> float:
        """Convert median normalized separator spacing to seed-relative pixels."""

        if len(positions) < 2:
            return 0.0
        spacings = tuple(
            current - previous for previous, current in pairwise(positions)
        )
        if not spacings or any(spacing <= 0.0 for spacing in spacings):
            return 0.0
        return median(spacings) * extent

    def _evaluate_direction(
        self,
        grayscale: NDArray[np.uint8],
        seed: BoardCandidateDiagnostic,
        seed_evidence: InternalGridEvidence,
        *,
        direction: str,
        typical_cell_width: float,
        typical_cell_height: float,
    ) -> _GridEnvelopeRefinementCandidate:
        """Analyze one proposed one-cell extension through all hard conditions."""

        horizontal = direction in {"left", "right"}
        typical_size = typical_cell_width if horizontal else typical_cell_height
        added_pixels = max(1, round(typical_size))
        x, y, width, height = self._extension_geometry(
            seed,
            direction=direction,
            added_pixels=added_pixels,
        )
        added_size_ratio = added_pixels / typical_size
        added_size_error = abs(added_size_ratio - 1.0)
        added_size_score = self._added_size_score(added_size_ratio)
        reasons: list[str] = []
        screenshot_height, screenshot_width = grayscale.shape
        if not self._inside_screenshot(
            x,
            y,
            width,
            height,
            screenshot_width,
            screenshot_height,
        ):
            reasons.append("refined envelope lies outside screenshot")
        elif not self._inside_content_area(
            x,
            y,
            width,
            height,
            screenshot_width,
            screenshot_height,
        ):
            reasons.append("refined envelope lies outside content area")
        if not (
            self._settings.grid_envelope_minimum_added_size_ratio
            <= added_size_ratio
            <= self._settings.grid_envelope_maximum_added_size_ratio
        ):
            reasons.append("added band size is not one typical cell")

        if reasons:
            return self._rejected_candidate(
                seed,
                direction=direction,
                x=x,
                y=y,
                width=width,
                height=height,
                added_pixels=added_pixels,
                added_size_error=added_size_error,
                rejection_reasons=tuple(reasons),
            )

        roi = grayscale[y : y + height, x : x + width]
        refined = self._analyzer.analyze(roi)
        reasons.extend(self._analyzer.rejection_reasons(refined))
        expected_rows = seed_evidence.estimated_rows + (0 if horizontal else 1)
        expected_columns = seed_evidence.estimated_columns + (1 if horizontal else 0)
        if refined.estimated_rows != expected_rows:
            reasons.append("refined row count is not seed rows plus the proposed band")
        if refined.estimated_columns != expected_columns:
            reasons.append(
                "refined column count is not seed columns plus the proposed band"
            )

        old_border_score = self._old_border_match_score(
            seed,
            refined,
            direction=direction,
            added_pixels=added_pixels,
            typical_cell_size=typical_size,
        )
        if old_border_score <= 0.0:
            reasons.append("old seed border was not found as an internal separator")

        continuation_score, supported_fraction = self._continuation_evidence(
            grayscale,
            seed,
            seed_evidence,
            direction=direction,
            x=x,
            y=y,
            width=width,
            height=height,
            typical_cell_width=typical_cell_width,
            typical_cell_height=typical_cell_height,
        )
        if (
            supported_fraction
            < self._settings.grid_envelope_minimum_supported_separator_fraction
        ):
            reasons.append("too few orthogonal separators continue into added band")

        spacing_increase = max(
            refined.horizontal_spacing_coefficient_of_variation
            - seed_evidence.horizontal_spacing_coefficient_of_variation,
            refined.vertical_spacing_coefficient_of_variation
            - seed_evidence.vertical_spacing_coefficient_of_variation,
            0.0,
        )
        maximum_increase = self._settings.grid_envelope_maximum_spacing_cv_increase
        spacing_score = (
            _clamp_unit(1.0 - spacing_increase / maximum_increase)
            if maximum_increase > 0.0
            else float(spacing_increase == 0.0)
        )
        if spacing_increase > maximum_increase:
            reasons.append("refined spacing CV worsened beyond configured tolerance")
        if refined.score < seed_evidence.score - (
            self._settings.grid_envelope_maximum_grid_score_drop
        ):
            reasons.append("refined grid score dropped beyond configured tolerance")
        if not self._all_spacings_positive(refined):
            reasons.append("refined grid contains non-positive separator spacing")

        refinement_score = _clamp_unit(
            0.30 * old_border_score
            + 0.30 * continuation_score
            + 0.15 * supported_fraction
            + 0.15 * spacing_score
            + 0.10 * added_size_score
        )
        if refinement_score < self._settings.grid_envelope_minimum_refinement_score:
            reasons.append("grid-envelope refinement score is below threshold")
        diagnostic = BoardEnvelopeRefinementDiagnostic(
            seed_x=seed.x,
            seed_y=seed.y,
            seed_width=seed.width,
            seed_height=seed.height,
            refined_x=x,
            refined_y=y,
            refined_width=width,
            refined_height=height,
            direction=direction,
            added_pixels=added_pixels,
            seed_rows=seed_evidence.estimated_rows,
            seed_columns=seed_evidence.estimated_columns,
            refined_rows=refined.estimated_rows,
            refined_columns=refined.estimated_columns,
            old_border_match_score=old_border_score,
            separator_continuation_score=continuation_score,
            supported_separator_fraction=supported_fraction,
            spacing_score=spacing_score,
            refined_grid_score=refined.score,
            refinement_score=refinement_score,
            accepted=not reasons,
            rejection_reasons=tuple(dict.fromkeys(reasons)),
        )
        return _GridEnvelopeRefinementCandidate(
            diagnostic=diagnostic,
            refined_grid_evidence=refined,
            added_size_error=added_size_error,
            combined_spacing_cv=(
                refined.horizontal_spacing_coefficient_of_variation
                + refined.vertical_spacing_coefficient_of_variation
            )
            / 2.0,
        )

    @staticmethod
    def _extension_geometry(
        seed: BoardCandidateDiagnostic,
        *,
        direction: str,
        added_pixels: int,
    ) -> tuple[int, int, int, int]:
        """Build one non-recursive extension that always contains the whole seed."""

        if direction == "left":
            return seed.x - added_pixels, seed.y, seed.width + added_pixels, seed.height
        if direction == "right":
            return seed.x, seed.y, seed.width + added_pixels, seed.height
        if direction == "top":
            return seed.x, seed.y - added_pixels, seed.width, seed.height + added_pixels
        return seed.x, seed.y, seed.width, seed.height + added_pixels

    @staticmethod
    def _inside_screenshot(
        x: int,
        y: int,
        width: int,
        height: int,
        screenshot_width: int,
        screenshot_height: int,
    ) -> bool:
        """Reject invalid or clipped rectangles before NumPy ROI slicing."""

        return (
            x >= 0
            and y >= 0
            and width > 0
            and height > 0
            and x + width <= screenshot_width
            and y + height <= screenshot_height
        )

    def _inside_content_area(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        screenshot_width: int,
        screenshot_height: int,
    ) -> bool:
        """Apply the BoardDetector's existing title, toolbar, and border exclusions."""

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
        return x >= left and y >= top and x + width <= right and y + height <= bottom

    def _old_border_match_score(
        self,
        seed: BoardCandidateDiagnostic,
        refined: InternalGridEvidence,
        *,
        direction: str,
        added_pixels: int,
        typical_cell_size: float,
    ) -> float:
        """Require the former outer seed boundary as a refined internal separator."""

        horizontal = direction in {"left", "right"}
        refined_extent = (
            seed.width + added_pixels if horizontal else seed.height + added_pixels
        )
        expected = (
            added_pixels / refined_extent
            if direction in {"left", "top"}
            else (seed.width if horizontal else seed.height) / refined_extent
        )
        positions = (
            refined.vertical_line_positions
            if horizontal
            else refined.horizontal_line_positions
        )
        internal_positions = positions[1:-1]
        if not internal_positions:
            return 0.0
        tolerance = (
            self._settings.grid_envelope_separator_position_tolerance_ratio
            * typical_cell_size
            / refined_extent
        )
        if tolerance <= 0.0:
            return 0.0
        distance = min(abs(position - expected) for position in internal_positions)
        return _clamp_unit(1.0 - distance / tolerance)

    def _continuation_evidence(
        self,
        grayscale: NDArray[np.uint8],
        seed: BoardCandidateDiagnostic,
        seed_evidence: InternalGridEvidence,
        *,
        direction: str,
        x: int,
        y: int,
        width: int,
        height: int,
        typical_cell_width: float,
        typical_cell_height: float,
    ) -> tuple[float, float]:
        """Probe aligned Sobel responses for separators crossing the added band."""

        horizontal_extension = direction in {"left", "right"}
        derivative_x, derivative_y = (0, 1) if horizontal_extension else (1, 0)
        gradient = np.abs(
            cv2.Sobel(
                grayscale,
                cv2.CV_64F,
                derivative_x,
                derivative_y,
                ksize=3,
            )
        )
        normalized_gradient = np.clip(gradient / (4.0 * 255.0), 0.0, 1.0)
        if direction == "left":
            band = normalized_gradient[seed.y : seed.y + seed.height, x : seed.x]
        elif direction == "right":
            band = normalized_gradient[
                seed.y : seed.y + seed.height,
                seed.x + seed.width : x + width,
            ]
        elif direction == "top":
            band = normalized_gradient[y : seed.y, seed.x : seed.x + seed.width]
        else:
            band = normalized_gradient[
                seed.y + seed.height : y + height,
                seed.x : seed.x + seed.width,
            ]
        positions = (
            seed_evidence.horizontal_line_positions[1:-1]
            if horizontal_extension
            else seed_evidence.vertical_line_positions[1:-1]
        )
        if band.size == 0 or not positions:
            return 0.0, 0.0

        typical_probe_size = (
            typical_cell_height if horizontal_extension else typical_cell_width
        )
        probe_radius = max(
            1,
            round(
                typical_probe_size
                * self._settings.grid_envelope_continuation_probe_thickness_ratio
                / 2.0
            ),
        )
        responses: list[float] = []
        for position in positions:
            if horizontal_extension:
                center = round(position * (seed.height - 1))
                start = max(0, center - probe_radius)
                end = min(band.shape[0], center + probe_radius + 1)
                probe = band[start:end, :]
                response = float(np.mean(np.max(probe, axis=0)))
            else:
                center = round(position * (seed.width - 1))
                start = max(0, center - probe_radius)
                end = min(band.shape[1], center + probe_radius + 1)
                probe = band[:, start:end]
                response = float(np.mean(np.max(probe, axis=1)))
            responses.append(_clamp_unit(response))

        minimum = self._settings.grid_envelope_minimum_line_continuation_response
        supported_fraction = sum(response >= minimum for response in responses) / len(
            responses
        )
        return _clamp_unit(fmean(responses)), _clamp_unit(supported_fraction)

    def _added_size_score(self, ratio: float) -> float:
        """Score one cell-sized band without authorizing half/two-cell extensions."""

        if not (
            self._settings.grid_envelope_minimum_added_size_ratio
            <= ratio
            <= self._settings.grid_envelope_maximum_added_size_ratio
        ):
            return 0.0
        scale = max(
            1.0 - self._settings.grid_envelope_minimum_added_size_ratio,
            self._settings.grid_envelope_maximum_added_size_ratio - 1.0,
        )
        return _clamp_unit(1.0 - abs(ratio - 1.0) / scale)

    @staticmethod
    def _all_spacings_positive(evidence: InternalGridEvidence) -> bool:
        """Reject collapsed normalized cells before public pixel conversion."""

        return all(
            current > previous
            for positions in (
                evidence.horizontal_line_positions,
                evidence.vertical_line_positions,
            )
            for previous, current in pairwise(positions)
        )

    def _rejected_candidate(
        self,
        seed: BoardCandidateDiagnostic,
        *,
        direction: str,
        x: int,
        y: int,
        width: int,
        height: int,
        added_pixels: int,
        added_size_error: float,
        rejection_reasons: tuple[str, ...],
    ) -> _GridEnvelopeRefinementCandidate:
        """Build bounded zero-evidence diagnostics for a geometry rejection."""

        diagnostic = BoardEnvelopeRefinementDiagnostic(
            seed_x=seed.x,
            seed_y=seed.y,
            seed_width=seed.width,
            seed_height=seed.height,
            refined_x=x,
            refined_y=y,
            refined_width=width,
            refined_height=height,
            direction=direction,
            added_pixels=added_pixels,
            seed_rows=seed.estimated_rows,
            seed_columns=seed.estimated_columns,
            refined_rows=0,
            refined_columns=0,
            old_border_match_score=0.0,
            separator_continuation_score=0.0,
            supported_separator_fraction=0.0,
            spacing_score=0.0,
            refined_grid_score=0.0,
            refinement_score=0.0,
            accepted=False,
            rejection_reasons=rejection_reasons,
        )
        return _GridEnvelopeRefinementCandidate(
            diagnostic=diagnostic,
            refined_grid_evidence=None,
            added_size_error=added_size_error,
            combined_spacing_cv=1.0,
        )

    def _selection_key(
        self,
        candidate: _GridEnvelopeRefinementCandidate,
    ) -> tuple[float, float, float, float, float, float, int]:
        """Implement the total maximal-envelope ordering required for ties."""

        diagnostic = candidate.diagnostic
        return (
            -(diagnostic.refined_rows * diagnostic.refined_columns),
            -diagnostic.refinement_score,
            -diagnostic.refined_grid_score,
            -diagnostic.separator_continuation_score,
            candidate.combined_spacing_cv,
            candidate.added_size_error,
            self._DIRECTION_ORDER[diagnostic.direction],
        )
