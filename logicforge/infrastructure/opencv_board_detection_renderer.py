"""OpenCV debug visualization for puzzle-board detection diagnostics."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
from logicforge.vision.board_detector import BoardDetectionAnalysis
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

        detection = analysis.detection
        cv2.rectangle(
            overlay,
            (detection.x, detection.y),
            (detection.x + detection.width, detection.y + detection.height),
            self._SELECTED_COLOR,
            3,
        )
        self._draw_metrics(overlay, analysis)
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        analysis: BoardDetectionAnalysis,
        destination: Path,
        *,
        debug: bool,
        draw_rejected_candidates: bool | None = None,
    ) -> Path | None:
        """Persist an overlay only when the caller explicitly enables debug output."""

        if not debug:
            return None

        output_path = destination.resolve()
        overlay = self.render(
            screenshot,
            analysis,
            draw_rejected_candidates=draw_rejected_candidates,
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
        lines = (
            f"x={detection.x}, y={detection.y}",
            f"width={detection.width}, height={detection.height}",
            f"confidence={detection.confidence:.3f}",
        )
        origin_x = max(8, detection.x)
        available_above = detection.y >= 76
        origin_y = detection.y - 56 if available_above else detection.y + 24
        line_step = 22
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
