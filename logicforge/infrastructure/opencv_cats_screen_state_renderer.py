"""Explicit OpenCV debug overlay for Cats screen-state classification."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.plugins.cats.screen_state import (
    CatsScreenRect,
    CatsScreenStateDetection,
)
from logicforge.vision.screenshot import Screenshot


class CatsScreenStateDebugRenderError(RuntimeError):
    """Report an explicit Cats state-overlay rendering or persistence failure."""


class OpenCvCatsScreenStateDebugRenderer:
    """Draw primitive state evidence onto a writable copy of the screenshot."""

    _VIEWPORT_COLOR = (255, 80, 180)
    _LEVEL_COLOR = (0, 255, 255)
    _RANKING_COLOR = (255, 220, 0)
    _BOARD_COLOR = (40, 220, 40)
    _ACTION_COLOR = (220, 40, 220)
    _TEXT_COLOR = (255, 255, 255)
    _BACKGROUND_COLOR = (24, 24, 24)

    def render(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
    ) -> NDArray[np.uint8]:
        """Return a writable annotated BGR copy without mutating source pixels."""

        overlay = screenshot.image.copy()
        diagnostics = detection.diagnostics
        if diagnostics.game_viewport_candidate is not None:
            self._draw_rect(
                overlay,
                diagnostics.game_viewport_candidate,
                self._VIEWPORT_COLOR,
                thickness=3,
            )
        if diagnostics.level_button_candidate is not None:
            self._draw_rect(
                overlay,
                diagnostics.level_button_candidate,
                self._LEVEL_COLOR,
                thickness=3,
            )
        for card in diagnostics.ranking_card_candidates:
            self._draw_rect(overlay, card, self._RANKING_COLOR, thickness=2)
        if diagnostics.board_candidate is not None:
            self._draw_rect(
                overlay,
                diagnostics.board_candidate,
                self._BOARD_COLOR,
                thickness=2,
            )
        if detection.action_point is not None:
            cv2.drawMarker(
                overlay,
                (detection.action_point.x, detection.action_point.y),
                self._ACTION_COLOR,
                cv2.MARKER_CROSS,
                18,
                3,
                cv2.LINE_AA,
            )
        self._draw_metrics(overlay, detection)
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsScreenStateDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Persist a PNG only after an explicit debug request."""

        if not debug:
            return None
        overlay = self.render(screenshot, detection)
        output_path = destination.resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(str(output_path), overlay)
        except (OSError, cv2.error) as error:
            raise CatsScreenStateDebugRenderError(
                f'Could not save Cats screen-state overlay to "{output_path}".'
            ) from error
        if not written:
            raise CatsScreenStateDebugRenderError(
                f'OpenCV could not encode Cats screen-state overlay at "{output_path}".'
            )
        return output_path

    @staticmethod
    def _draw_rect(
        overlay: NDArray[np.uint8],
        rect: CatsScreenRect,
        color: tuple[int, int, int],
        *,
        thickness: int,
    ) -> None:
        """Draw one half-open diagnostic rectangle with safe image clipping."""

        right = min(overlay.shape[1] - 1, rect.x + rect.width - 1)
        bottom = min(overlay.shape[0] - 1, rect.y + rect.height - 1)
        cv2.rectangle(
            overlay,
            (min(rect.x, right), min(rect.y, bottom)),
            (right, bottom),
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_metrics(
        self,
        overlay: NDArray[np.uint8],
        detection: CatsScreenStateDetection,
    ) -> None:
        """Display state confidence and both transition-screen subscores."""

        lines = (
            f"state={detection.state.name}, confidence={detection.confidence:.3f}",
            f"game viewport score={detection.diagnostics.game_viewport_score:.3f}",
            f"level button score={detection.diagnostics.level_button_score:.3f}",
            f"ranking score={detection.diagnostics.ranking_score:.3f}",
        )
        line_step = 22
        widths = tuple(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0][0]
            for line in lines
        )
        cv2.rectangle(
            overlay,
            (6, 6),
            (
                min(overlay.shape[1] - 1, max(widths) + 20),
                min(overlay.shape[0] - 1, 12 + len(lines) * line_step),
            ),
            self._BACKGROUND_COLOR,
            cv2.FILLED,
        )
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (12, 25 + index * line_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                self._TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
