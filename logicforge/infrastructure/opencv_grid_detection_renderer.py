"""Explicit OpenCV debug visualization for public grid and cell geometry."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import GridDetection, GridDetectionDiagnostics
from logicforge.vision.screenshot import Screenshot


class GridDebugRenderError(RuntimeError):
    """Report an explicit grid-overlay rendering or persistence failure."""


class OpenCvGridDetectionDebugRenderer:
    """Draw full-screenshot boundaries and cells without mutating source pixels."""

    _BOARD_COLOR = (40, 220, 40)
    _HORIZONTAL_COLOR = (255, 210, 0)
    _VERTICAL_COLOR = (220, 80, 255)
    _CENTER_COLOR = (0, 220, 255)
    _REJECTED_COLOR = (80, 80, 220)
    _TEXT_COLOR = (255, 255, 255)

    def render(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection,
        *,
        draw_cell_centers: bool = True,
        detailed: bool = False,
    ) -> NDArray[np.uint8]:
        """Return an annotated BGR copy while preserving immutable input pixels."""

        overlay = screenshot.image.copy()
        self._draw_board(overlay, board, self._BOARD_COLOR)
        self._draw_boundaries(overlay, board, grid)
        if draw_cell_centers:
            self._draw_cells(overlay, grid, detailed=detailed)
        self._draw_metrics(overlay, board, grid)
        return overlay

    def render_failure(
        self,
        screenshot: Screenshot,
        diagnostics: GridDetectionDiagnostics,
    ) -> NDArray[np.uint8]:
        """Render available rejected geometry directly from typed failure evidence."""

        overlay = screenshot.image.copy()
        board = BoardDetection(
            x=diagnostics.board_x,
            y=diagnostics.board_y,
            width=diagnostics.board_width,
            height=diagnostics.board_height,
            confidence=0.0,
        )
        if board.width > 0 and board.height > 0 and board.x >= 0 and board.y >= 0:
            self._draw_board(overlay, board, self._REJECTED_COLOR)
            for y in diagnostics.horizontal_lines:
                self._draw_horizontal_line(overlay, board, y, self._REJECTED_COLOR)
            for x in diagnostics.vertical_lines:
                self._draw_vertical_line(overlay, board, x, self._REJECTED_COLOR)
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection,
        destination: Path,
        *,
        debug: bool,
        draw_cell_centers: bool = True,
        detailed: bool = False,
    ) -> Path | None:
        """Persist a grid overlay only when explicit debug behavior is enabled."""

        if not debug:
            return None
        overlay = self.render(
            screenshot,
            board,
            grid,
            draw_cell_centers=draw_cell_centers,
            detailed=detailed,
        )
        return self._save_overlay(overlay, destination)

    def save_failure_debug_overlay(
        self,
        screenshot: Screenshot,
        diagnostics: GridDetectionDiagnostics,
        destination: Path,
        *,
        debug: bool,
    ) -> Path | None:
        """Persist rejected primitive diagnostics only under explicit debug mode."""

        if not debug:
            return None
        return self._save_overlay(
            self.render_failure(screenshot, diagnostics),
            destination,
        )

    @staticmethod
    def _save_overlay(overlay: NDArray[np.uint8], destination: Path) -> Path:
        """Encode one already-rendered BGR overlay with typed failure reporting."""

        output_path = destination.resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(str(output_path), overlay)
        except (OSError, cv2.error) as error:
            raise GridDebugRenderError(
                f'Could not save the grid debug overlay to "{output_path}".'
            ) from error
        if not written:
            raise GridDebugRenderError(
                f'OpenCV could not encode the grid overlay at "{output_path}".'
            )
        return output_path

    @staticmethod
    def _draw_board(
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        color: tuple[int, int, int],
    ) -> None:
        """Draw the half-open board boundary clipped to drawable image pixels."""

        right = min(overlay.shape[1] - 1, board.x + board.width)
        bottom = min(overlay.shape[0] - 1, board.y + board.height)
        cv2.rectangle(overlay, (board.x, board.y), (right, bottom), color, 2)

    def _draw_boundaries(
        self,
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        grid: GridDetection,
    ) -> None:
        """Draw each public boundary once in its orientation-specific color."""

        for y in grid.horizontal_lines:
            self._draw_horizontal_line(overlay, board, y, self._HORIZONTAL_COLOR)
        for x in grid.vertical_lines:
            self._draw_vertical_line(overlay, board, x, self._VERTICAL_COLOR)

    @staticmethod
    def _draw_horizontal_line(
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one screenshot-space horizontal boundary with safe clipping."""

        drawable_y = min(overlay.shape[0] - 1, max(0, y))
        right = min(overlay.shape[1] - 1, board.x + board.width)
        cv2.line(
            overlay,
            (board.x, drawable_y),
            (right, drawable_y),
            color,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_vertical_line(
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        x: int,
        color: tuple[int, int, int],
    ) -> None:
        """Draw one screenshot-space vertical boundary with safe clipping."""

        drawable_x = min(overlay.shape[1] - 1, max(0, x))
        bottom = min(overlay.shape[0] - 1, board.y + board.height)
        cv2.line(
            overlay,
            (drawable_x, board.y),
            (drawable_x, bottom),
            color,
            1,
            cv2.LINE_AA,
        )

    def _draw_cells(
        self,
        overlay: NDArray[np.uint8],
        grid: GridDetection,
        *,
        detailed: bool,
    ) -> None:
        """Draw centers and optional stable zero-based row/column labels."""

        for cell in grid.cells:
            cv2.circle(
                overlay,
                (cell.center_x, cell.center_y),
                2,
                self._CENTER_COLOR,
                cv2.FILLED,
                cv2.LINE_AA,
            )
            if detailed:
                cv2.putText(
                    overlay,
                    f"{cell.row},{cell.column}",
                    (cell.x + 3, cell.y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    self._TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )

    def _draw_metrics(
        self,
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        grid: GridDetection,
    ) -> None:
        """Display public dimensions, counts, and grid-only confidence."""

        lines = (
            f"rows={grid.rows}, columns={grid.columns}",
            (
                f"horizontal lines={len(grid.horizontal_lines)}, "
                f"vertical lines={len(grid.vertical_lines)}"
            ),
            f"cells={len(grid.cells)}, grid confidence={grid.confidence:.3f}",
        )
        line_step = 21
        origin_x = max(8, board.x)
        origin_y = board.y - 52 if board.y >= 68 else board.y + 22
        widths = tuple(
            cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
            for line in lines
        )
        cv2.rectangle(
            overlay,
            (origin_x - 5, max(0, origin_y - 17)),
            (
                min(overlay.shape[1] - 1, origin_x + max(widths) + 10),
                min(overlay.shape[0] - 1, origin_y + 2 * line_step + 7),
            ),
            (24, 24, 24),
            cv2.FILLED,
        )
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (origin_x, origin_y + index * line_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self._TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
