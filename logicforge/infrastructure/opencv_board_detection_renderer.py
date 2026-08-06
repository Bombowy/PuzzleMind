"""OpenCV debug visualization for puzzle-board detection diagnostics."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
from logicforge.vision.board_detector import (
    BoardCandidateDiagnostic,
    BoardDetectionAnalysis,
    BoardDetectionDiagnostics,
    BoardEnvelopeRefinementDiagnostic,
)
from logicforge.vision.screenshot import Screenshot


class BoardDebugRenderError(RuntimeError):
    """Report an explicit board-overlay rendering or persistence failure."""


class OpenCvBoardDetectionDebugRenderer:
    """Render backend diagnostics without coupling the detector to filesystem I/O.

    The renderer owns all OpenCV drawing and encoding concerns. The application can
    therefore run detection without side effects and opt into an overlay only for
    a debugging workflow.
    """

    _SELECTED_COLOR = (40, 220, 40)
    _ACCEPTED_COLOR = (0, 190, 255)
    _REJECTED_COLOR = (80, 80, 220)
    _SEED_COLOR = (255, 180, 40)
    _ADDED_BAND_COLOR = (40, 180, 255)
    _REJECTED_REFINEMENT_COLOR = (140, 80, 220)
    _HORIZONTAL_GRID_COLOR = (255, 210, 0)
    _VERTICAL_GRID_COLOR = (220, 80, 255)
    _TEXT_COLOR = (255, 255, 255)

    def __init__(self, settings: BoardDetectionSettings | None = None) -> None:
        """Use the same immutable settings that control candidate visualization."""

        self._settings = settings or BoardDetectionSettings()

    def render(
        self,
        screenshot: Screenshot,
        analysis: BoardDetectionAnalysis,
        *,
        draw_rejected_candidates: bool | None = None,
        draw_grid_lines: bool = True,
    ) -> NDArray[np.uint8]:
        """Return a writable BGR copy annotated with the selected board and metrics."""

        overlay = screenshot.image.copy()
        should_draw_rejected = (
            self._settings.debug_rejected_candidates
            if draw_rejected_candidates is None
            else draw_rejected_candidates
        )

        if should_draw_rejected:
            for candidate in analysis.diagnostics.candidates:
                if candidate == analysis.diagnostics.selected_candidate:
                    continue
                color = (
                    self._ACCEPTED_COLOR if candidate.accepted else self._REJECTED_COLOR
                )
                cv2.rectangle(
                    overlay,
                    (candidate.x, candidate.y),
                    (candidate.x + candidate.width, candidate.y + candidate.height),
                    color,
                    1,
                )
                self._draw_candidate_grid_label(overlay, candidate, color)
            for refinement in analysis.diagnostics.envelope_refinements:
                if refinement.accepted:
                    continue
                self._draw_refinement_rectangle(
                    overlay,
                    refinement,
                    self._REJECTED_REFINEMENT_COLOR,
                    thickness=1,
                )

        detection = analysis.detection
        selected_candidate = analysis.diagnostics.selected_candidate
        selected_refinement = analysis.diagnostics.selected_refinement
        if draw_grid_lines and selected_candidate is not None:
            self._draw_grid_lines(overlay, selected_candidate)
        if selected_refinement is not None:
            self._draw_selected_refinement(overlay, selected_refinement)
        cv2.rectangle(
            overlay,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            self._SELECTED_COLOR,
            3,
        )
        self._draw_metrics(overlay, analysis)
        return overlay

    def render_rejected_candidates(
        self,
        screenshot: Screenshot,
        diagnostics: BoardDetectionDiagnostics,
    ) -> NDArray[np.uint8]:
        """Visualize fail-closed candidates directly from ``BoardDetectionError``.

        This in-memory operation lets diagnostic callers inspect advertisement-like
        failures even when no successful ``BoardDetectionAnalysis`` can exist.
        Persistence remains a separate explicit caller decision.
        """

        overlay = screenshot.image.copy()
        for candidate in diagnostics.candidates:
            if candidate.accepted:
                continue
            cv2.rectangle(
                overlay,
                (candidate.x, candidate.y),
                (candidate.x + candidate.width, candidate.y + candidate.height),
                self._REJECTED_COLOR,
                2,
            )
            self._draw_candidate_grid_label(
                overlay,
                candidate,
                self._REJECTED_COLOR,
            )
        for refinement in diagnostics.envelope_refinements:
            if refinement.accepted:
                continue
            self._draw_refinement_rectangle(
                overlay,
                refinement,
                self._REJECTED_REFINEMENT_COLOR,
                thickness=1,
            )
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        analysis: BoardDetectionAnalysis,
        destination: Path,
        *,
        debug: bool,
        draw_rejected_candidates: bool | None = None,
        draw_grid_lines: bool = True,
    ) -> Path | None:
        """Persist an overlay only when the caller explicitly enables debug output."""

        if not debug:
            return None

        output_path = destination.resolve()
        overlay = self.render(
            screenshot,
            analysis,
            draw_rejected_candidates=draw_rejected_candidates,
            draw_grid_lines=draw_grid_lines,
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(str(output_path), overlay)
        except (OSError, cv2.error) as error:
            raise BoardDebugRenderError(
                f'Could not save the board debug overlay to "{output_path}".'
            ) from error

        if not written:
            raise BoardDebugRenderError(
                f'OpenCV could not encode the board overlay at "{output_path}".'
            )
        return output_path

    def _draw_metrics(
        self,
        overlay: NDArray[np.uint8],
        analysis: BoardDetectionAnalysis,
    ) -> None:
        """Place readable geometry and confidence labels without changing the model."""

        detection = analysis.detection
        selected = analysis.diagnostics.selected_candidate
        if selected is None:
            raise BoardDebugRenderError(
                "A successful board analysis must identify its selected candidate."
            )
        lines: tuple[str, ...] = (
            f"x={detection.x}, y={detection.y}",
            f"width={detection.width}, height={detection.height}",
            f"confidence={detection.confidence:.3f}",
            (
                f"rows={selected.estimated_rows}, "
                f"columns={selected.estimated_columns}"
            ),
            (
                f"horizontal lines={selected.horizontal_grid_line_count}, "
                f"vertical lines={selected.vertical_grid_line_count}"
            ),
            f"grid evidence={selected.grid_evidence_score:.3f}",
        )
        refinement = analysis.diagnostics.selected_refinement
        if refinement is not None:
            lines = (
                *lines[:3],
                (
                    f"seed={refinement.seed_rows}x{refinement.seed_columns}, "
                    f"refined={refinement.refined_rows}x{refinement.refined_columns}"
                ),
                f"direction={refinement.direction}, added={refinement.added_pixels}px",
                (
                    f"continuation={refinement.separator_continuation_score:.3f}, "
                    f"supported={refinement.supported_separator_fraction:.3f}"
                ),
                f"refinement={refinement.refinement_score:.3f}",
                f"grid evidence={refinement.refined_grid_score:.3f}",
            )
        origin_x = max(8, detection.x)
        line_step = 22
        required_height = len(lines) * line_step + 10
        available_above = detection.y >= required_height
        origin_y = (
            detection.y - (len(lines) - 1) * line_step - 10
            if available_above
            else detection.y + 24
        )
        text_sizes = tuple(
            cv2.getTextSize(
                line,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )[0]
            for line in lines
        )
        panel_right = min(
            overlay.shape[1] - 1,
            origin_x + max(width for width, _ in text_sizes) + 12,
        )
        panel_top = max(0, origin_y - 18)
        panel_bottom = min(
            overlay.shape[0] - 1,
            origin_y + (len(lines) - 1) * line_step + 8,
        )
        cv2.rectangle(
            overlay,
            (origin_x - 5, panel_top),
            (panel_right, panel_bottom),
            (24, 24, 24),
            cv2.FILLED,
        )

        for index, line in enumerate(lines):
            position = (origin_x, origin_y + index * line_step)
            cv2.putText(
                overlay,
                line,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self._TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _draw_grid_lines(
        self,
        overlay: NDArray[np.uint8],
        candidate: BoardCandidateDiagnostic,
    ) -> None:
        """Draw de-duplicated internal boundaries from normalized diagnostics."""

        right = candidate.x + candidate.width
        bottom = candidate.y + candidate.height
        for normalized_y in candidate.horizontal_grid_line_positions[1:-1]:
            y = candidate.y + round(normalized_y * (candidate.height - 1))
            cv2.line(
                overlay,
                (candidate.x, y),
                (right, y),
                self._HORIZONTAL_GRID_COLOR,
                1,
                cv2.LINE_AA,
            )
        for normalized_x in candidate.vertical_grid_line_positions[1:-1]:
            x = candidate.x + round(normalized_x * (candidate.width - 1))
            cv2.line(
                overlay,
                (x, candidate.y),
                (x, bottom),
                self._VERTICAL_GRID_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _draw_selected_refinement(
        self,
        overlay: NDArray[np.uint8],
        refinement: BoardEnvelopeRefinementDiagnostic,
    ) -> None:
        """Show the contour seed and the verified newly added cell band."""

        band_x, band_y, band_width, band_height = self._added_band_rectangle(refinement)
        tinted = overlay.copy()
        cv2.rectangle(
            tinted,
            (band_x, band_y),
            (band_x + band_width, band_y + band_height),
            self._ADDED_BAND_COLOR,
            cv2.FILLED,
        )
        cv2.addWeighted(tinted, 0.20, overlay, 0.80, 0.0, overlay)
        cv2.rectangle(
            overlay,
            (band_x, band_y),
            (band_x + band_width, band_y + band_height),
            self._ADDED_BAND_COLOR,
            2,
        )
        cv2.rectangle(
            overlay,
            (refinement.seed_x, refinement.seed_y),
            (
                refinement.seed_x + refinement.seed_width,
                refinement.seed_y + refinement.seed_height,
            ),
            self._SEED_COLOR,
            1,
        )

    @staticmethod
    def _added_band_rectangle(
        refinement: BoardEnvelopeRefinementDiagnostic,
    ) -> tuple[int, int, int, int]:
        """Derive the one-cell difference between the seed and refined envelope."""

        if refinement.direction == "left":
            return (
                refinement.refined_x,
                refinement.refined_y,
                refinement.added_pixels,
                refinement.refined_height,
            )
        if refinement.direction == "right":
            return (
                refinement.seed_x + refinement.seed_width,
                refinement.refined_y,
                refinement.added_pixels,
                refinement.refined_height,
            )
        if refinement.direction == "top":
            return (
                refinement.refined_x,
                refinement.refined_y,
                refinement.refined_width,
                refinement.added_pixels,
            )
        return (
            refinement.refined_x,
            refinement.seed_y + refinement.seed_height,
            refinement.refined_width,
            refinement.added_pixels,
        )

    @staticmethod
    def _draw_refinement_rectangle(
        overlay: NDArray[np.uint8],
        refinement: BoardEnvelopeRefinementDiagnostic,
        color: tuple[int, int, int],
        *,
        thickness: int,
    ) -> None:
        """Draw one rejected or accepted refined rectangle from public primitives."""

        cv2.rectangle(
            overlay,
            (refinement.refined_x, refinement.refined_y),
            (
                refinement.refined_x + refinement.refined_width,
                refinement.refined_y + refinement.refined_height,
            ),
            color,
            thickness,
        )

    @staticmethod
    def _draw_candidate_grid_label(
        overlay: NDArray[np.uint8],
        candidate: BoardCandidateDiagnostic,
        color: tuple[int, int, int],
    ) -> None:
        """Annotate rejected rectangles with compact primitive grid diagnostics."""

        label = (
            f"grid={candidate.grid_evidence_score:.2f} "
            f"H={candidate.horizontal_grid_line_count} "
            f"V={candidate.vertical_grid_line_count}"
        )
        position = (max(2, candidate.x), max(14, candidate.y - 4))
        cv2.putText(
            overlay,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )
