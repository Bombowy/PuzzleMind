"""OpenCV diagnostics for Cats tile components and fitted lattice geometry."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.plugins.cats.tile_grid import CatsTileGridDetection
from logicforge.vision.screenshot import Screenshot


class CatsTileGridDebugRenderError(RuntimeError):
    """Report explicit Cats tile-grid overlay persistence failures."""


class OpenCvCatsTileGridDebugRenderer:
    """Render component, center, boundary, and score evidence on a copied image."""

    _CANDIDATE_COLOR = (110, 110, 110)
    _SELECTED_COLOR = (0, 210, 255)
    _BOARD_COLOR = (40, 225, 40)
    _ROW_COLOR = (255, 180, 20)
    _COLUMN_COLOR = (220, 70, 240)
    _CENTER_COLOR = (20, 20, 20)
    _MISSING_SLOT_COLOR = (30, 30, 235)

    def render(
        self,
        screenshot: Screenshot,
        detection: CatsTileGridDetection,
    ) -> NDArray[np.uint8]:
        """Return an annotated BGR copy without mutating immutable source pixels."""

        overlay = screenshot.image.copy()
        for component in detection.diagnostics.components:
            color = (
                self._SELECTED_COLOR if component.accepted else self._CANDIDATE_COLOR
            )
            thickness = 2 if component.accepted else 1
            cv2.rectangle(
                overlay,
                (component.x, component.y),
                (component.x + component.width - 1, component.y + component.height - 1),
                color,
                thickness,
                cv2.LINE_AA,
            )

        board = detection.board
        for center_y in detection.diagnostics.row_centers:
            y = round(center_y)
            cv2.line(
                overlay,
                (board.x, y),
                (board.x + board.width - 1, y),
                self._ROW_COLOR,
                1,
                cv2.LINE_AA,
            )
        for center_x in detection.diagnostics.column_centers:
            x = round(center_x)
            cv2.line(
                overlay,
                (x, board.y),
                (x, board.y + board.height - 1),
                self._COLUMN_COLOR,
                1,
                cv2.LINE_AA,
            )
        for y in detection.grid.horizontal_lines:
            cv2.line(
                overlay,
                (board.x, y),
                (board.x + board.width - 1, y),
                self._ROW_COLOR,
                1,
                cv2.LINE_AA,
            )
        for x in detection.grid.vertical_lines:
            cv2.line(
                overlay,
                (x, board.y),
                (x, board.y + board.height - 1),
                self._COLUMN_COLOR,
                1,
                cv2.LINE_AA,
            )
        for cell in detection.grid.cells:
            cv2.circle(
                overlay,
                (cell.center_x, cell.center_y),
                2,
                self._CENTER_COLOR,
                cv2.FILLED,
                cv2.LINE_AA,
            )
        for row, column in detection.diagnostics.missing_slot_coordinates:
            cell = detection.grid.cells[row * detection.grid.columns + column]
            cv2.rectangle(
                overlay,
                (cell.x + 2, cell.y + 2),
                (cell.x + cell.width - 3, cell.y + cell.height - 3),
                self._MISSING_SLOT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                overlay,
                (cell.x + 4, cell.y + 4),
                (cell.x + cell.width - 5, cell.y + cell.height - 5),
                self._MISSING_SLOT_COLOR,
                2,
                cv2.LINE_AA,
            )
        cv2.rectangle(
            overlay,
            (board.x, board.y),
            (board.x + board.width - 1, board.y + board.height - 1),
            self._BOARD_COLOR,
            3,
            cv2.LINE_AA,
        )
        self._draw_metrics(overlay, detection)
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        detection: CatsTileGridDetection,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Persist one rendered overlay only when diagnostics are enabled."""

        if not debug:
            return None
        output_path = destination.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = cv2.imwrite(str(output_path), self.render(screenshot, detection))
        except (OSError, cv2.error) as error:
            raise CatsTileGridDebugRenderError(
                f'Could not save Cats tile-grid overlay to "{output_path}".'
            ) from error
        if not written:
            raise CatsTileGridDebugRenderError(
                f'OpenCV could not encode Cats tile-grid overlay at "{output_path}".'
            )
        return output_path

    @staticmethod
    def _draw_metrics(
        overlay: NDArray[np.uint8],
        detection: CatsTileGridDetection,
    ) -> None:
        """Draw compact lattice counts, pitch regularity, occupancy, and score."""

        diagnostics = detection.diagnostics
        lines = (
            f"grid={detection.grid.rows}x{detection.grid.columns} "
            f"tiles={diagnostics.selected_tile_count}",
            f"occupancy={diagnostics.occupancy_ratio:.3f} "
            f"score={diagnostics.grid_score:.3f}",
            f"pitch x/y={diagnostics.horizontal_pitch:.2f}/"
            f"{diagnostics.vertical_pitch:.2f}",
            f"pitch CV x/y={diagnostics.horizontal_pitch_cv:.3f}/"
            f"{diagnostics.vertical_pitch_cv:.3f}",
            f"support row/col min={diagnostics.minimum_row_support_ratio:.3f}/"
            f"{diagnostics.minimum_column_support_ratio:.3f} "
            f"missing={len(diagnostics.missing_slot_coordinates)}",
        )
        x = max(8, detection.board.x)
        y = detection.board.y - 79 if detection.board.y >= 96 else 22
        widths = tuple(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0][0]
            for line in lines
        )
        cv2.rectangle(
            overlay,
            (x - 5, max(0, y - 17)),
            (min(overlay.shape[1] - 1, x + max(widths) + 8), y + 86),
            (25, 25, 25),
            cv2.FILLED,
        )
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (x, y + 20 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
