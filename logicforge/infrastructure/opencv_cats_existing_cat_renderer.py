"""Explicit OpenCV debug rendering for per-cell existing-cat evidence."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.plugins.cats.existing_cat import CatsExistingCatDetection
from logicforge.vision.grid_detector import GridDetection
from logicforge.vision.screenshot import Screenshot


class CatsExistingCatDebugRenderError(RuntimeError):
    """Report explicit existing-cat overlay persistence failures."""


class OpenCvCatsExistingCatDebugRenderer:
    """Draw fitted cells, ROIs, accepted cats, and suspicious rejected cells."""

    def render(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        detection: CatsExistingCatDetection,
        *,
        show_rois: bool = True,
    ) -> NDArray[np.uint8]:
        """Return an annotated BGR copy without changing source pixels."""

        overlay = screenshot.image.copy()
        accepted = {(cat.row, cat.column) for cat in detection.cats}
        rejected_scores = sorted(
            (
                diagnostic.score
                for diagnostic in detection.diagnostics.cells
                if not diagnostic.accepted
            ),
            reverse=True,
        )
        suspicious_threshold = (
            rejected_scores[min(4, len(rejected_scores) - 1)]
            if rejected_scores
            else 1.0
        )
        for cell in grid.cells:
            cv2.rectangle(
                overlay,
                (cell.x, cell.y),
                (cell.x + cell.width - 1, cell.y + cell.height - 1),
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
        for diagnostic in detection.diagnostics.cells:
            coordinate = (diagnostic.row, diagnostic.column)
            is_accepted = coordinate in accepted
            is_suspicious = not is_accepted and diagnostic.score >= suspicious_threshold
            if show_rois:
                cv2.rectangle(
                    overlay,
                    (diagnostic.roi_x, diagnostic.roi_y),
                    (
                        diagnostic.roi_x + diagnostic.roi_width - 1,
                        diagnostic.roi_y + diagnostic.roi_height - 1,
                    ),
                    (190, 130, 40),
                    1,
                    cv2.LINE_AA,
                )
            if not is_accepted and not is_suspicious:
                continue
            color = (40, 225, 40) if is_accepted else (20, 150, 245)
            thickness = 3 if is_accepted else 2
            cv2.rectangle(
                overlay,
                (diagnostic.roi_x, diagnostic.roi_y),
                (
                    diagnostic.roi_x + diagnostic.roi_width - 1,
                    diagnostic.roi_y + diagnostic.roi_height - 1,
                ),
                color,
                thickness,
                cv2.LINE_AA,
            )
            text = (
                f"{diagnostic.row},{diagnostic.column} "
                f"fg={diagnostic.foreground_ratio:.2f} "
                f"cc={diagnostic.largest_component_ratio:.2f} "
                f"s={diagnostic.score:.2f}"
            )
            cv2.putText(
                overlay,
                text,
                (diagnostic.roi_x + 2, max(12, diagnostic.roi_y + 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                color,
                1,
                cv2.LINE_AA,
            )
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        grid: GridDetection,
        detection: CatsExistingCatDetection,
        destination: Path,
        *,
        debug: bool,
        show_rois: bool = True,
    ) -> Path | None:
        """Persist only under explicit debug opt-in."""

        if not debug:
            return None
        output_path = destination.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = cv2.imwrite(
                str(output_path),
                self.render(screenshot, grid, detection, show_rois=show_rois),
            )
        except (OSError, cv2.error) as error:
            raise CatsExistingCatDebugRenderError(
                f'Could not save Cats existing-cat overlay to "{output_path}".'
            ) from error
        if not written:
            raise CatsExistingCatDebugRenderError(
                f'OpenCV could not encode Cats existing-cat overlay at "{output_path}".'
            )
        return output_path
