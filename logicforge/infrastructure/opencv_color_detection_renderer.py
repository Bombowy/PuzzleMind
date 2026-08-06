"""Explicit OpenCV debug visualization for classified cell colors."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import ColorDetectionSettings
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.color_detector import ColorDetectionResult, LabColor
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.screenshot import Screenshot


class ColorDebugRenderError(RuntimeError):
    """Report an explicit color-overlay rendering or persistence failure."""


class OpenCvColorDetectionDebugRenderer:
    """Label cell classes on a copied screenshot without mutating source pixels."""

    _BOARD_COLOR = (40, 220, 40)
    _CELL_COLOR = (235, 235, 235)
    _TEXT_COLOR = (255, 255, 255)
    _TEXT_OUTLINE_COLOR = (20, 20, 20)
    _SAMPLE_REGION_COLOR = (255, 120, 0)

    def __init__(self, settings: ColorDetectionSettings | None = None) -> None:
        """Share exact corner-patch geometry with the configured color detector."""

        self._sampling_geometry = OpenCvColorDetector(settings)

    def render(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection,
        result: ColorDetectionResult,
        *,
        draw_representative_swatches: bool = True,
        draw_sample_regions: bool = False,
    ) -> NDArray[np.uint8]:
        """Return a BGR overlay with board, cells, logical IDs, and summary."""

        if len(grid.cells) != len(result.observations):
            raise ColorDebugRenderError(
                "Grid cells and color observations must have matching counts."
            )
        overlay = screenshot.image.copy()
        self._draw_board(overlay, board)
        for cell, observation in zip(
            grid.cells,
            result.observations,
            strict=True,
        ):
            self._draw_cell(overlay, cell)
            if draw_sample_regions:
                self._draw_sample_regions(overlay, cell)
            self._draw_label(overlay, cell, observation.color_id)
            if draw_representative_swatches:
                self._draw_swatch(
                    overlay,
                    cell,
                    self._lab_to_bgr(observation.representative_lab),
                )
        self._draw_summary(overlay, board, grid, result)
        return overlay

    def save_debug_overlay(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
        grid: GridDetection,
        result: ColorDetectionResult,
        destination: Path,
        *,
        debug: bool,
        draw_representative_swatches: bool = True,
        draw_sample_regions: bool = False,
    ) -> Path | None:
        """Persist the annotated copy only under explicit debug behavior."""

        if not debug:
            return None
        overlay = self.render(
            screenshot,
            board,
            grid,
            result,
            draw_representative_swatches=draw_representative_swatches,
            draw_sample_regions=draw_sample_regions,
        )
        output_path = destination.resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            written = cv2.imwrite(str(output_path), overlay)
        except (OSError, cv2.error) as error:
            raise ColorDebugRenderError(
                f'Could not save the color debug overlay to "{output_path}".'
            ) from error
        if not written:
            raise ColorDebugRenderError(
                f'OpenCV could not encode the color overlay at "{output_path}".'
            )
        return output_path

    def _draw_board(
        self,
        overlay: NDArray[np.uint8],
        board: BoardDetection,
    ) -> None:
        """Draw the selected half-open board rectangle in green."""

        right = min(overlay.shape[1] - 1, board.x + board.width)
        bottom = min(overlay.shape[0] - 1, board.y + board.height)
        cv2.rectangle(
            overlay,
            (board.x, board.y),
            (right, bottom),
            self._BOARD_COLOR,
            2,
        )

    def _draw_cell(
        self,
        overlay: NDArray[np.uint8],
        cell: CellBounds,
    ) -> None:
        """Outline one half-open cell while clipping its drawable bottom-right."""

        right = min(overlay.shape[1] - 1, cell.x + cell.width)
        bottom = min(overlay.shape[0] - 1, cell.y + cell.height)
        cv2.rectangle(
            overlay,
            (cell.x, cell.y),
            (right, bottom),
            self._CELL_COLOR,
            1,
            cv2.LINE_AA,
        )

    def _draw_sample_regions(
        self,
        overlay: NDArray[np.uint8],
        cell: CellBounds,
    ) -> None:
        """Outline the exact TL, TR, BL, BR half-open color evidence regions."""

        for left, top, right, bottom in self._sampling_geometry._corner_sample_bounds(
            cell
        ):
            cv2.rectangle(
                overlay,
                (left, top),
                (right - 1, bottom - 1),
                self._SAMPLE_REGION_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _draw_label(
        self,
        overlay: NDArray[np.uint8],
        cell: CellBounds,
        color_id: str,
    ) -> None:
        """Center a readable logical identifier using an outline plus foreground."""

        font_scale = max(0.32, min(0.62, min(cell.width, cell.height) / 90.0))
        text_size, baseline = cv2.getTextSize(
            color_id,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )
        origin = (
            cell.center_x - text_size[0] // 2,
            cell.center_y + (text_size[1] - baseline) // 2,
        )
        cv2.putText(
            overlay,
            color_id,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self._TEXT_OUTLINE_COLOR,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            color_id,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self._TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_swatch(
        overlay: NDArray[np.uint8],
        cell: CellBounds,
        bgr: tuple[int, int, int],
    ) -> None:
        """Draw a small diagnostic patch of the robust representative color."""

        swatch_size = max(3, min(10, min(cell.width, cell.height) // 6))
        left = cell.x + 3
        top = cell.y + cell.height - swatch_size - 3
        right = min(cell.x + cell.width - 1, left + swatch_size)
        bottom = min(cell.y + cell.height - 1, top + swatch_size)
        cv2.rectangle(overlay, (left, top), (right, bottom), bgr, cv2.FILLED)
        cv2.rectangle(overlay, (left, top), (right, bottom), (20, 20, 20), 1)

    def _draw_summary(
        self,
        overlay: NDArray[np.uint8],
        board: BoardDetection,
        grid: GridDetection,
        result: ColorDetectionResult,
    ) -> None:
        """Render one compact global class-count and confidence summary."""

        summary = (
            f"colors={result.color_count}, cells={len(result.observations)}, "
            f"grid={grid.rows}x{grid.columns}, confidence={result.mean_confidence:.3f}"
        )
        origin_x = max(8, board.x)
        origin_y = board.y - 10 if board.y >= 32 else board.y + 20
        text_width = cv2.getTextSize(
            summary,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )[
            0
        ][0]
        cv2.rectangle(
            overlay,
            (origin_x - 4, max(0, origin_y - 17)),
            (
                min(overlay.shape[1] - 1, origin_x + text_width + 5),
                min(overlay.shape[0] - 1, origin_y + 5),
            ),
            (24, 24, 24),
            cv2.FILLED,
        )
        cv2.putText(
            overlay,
            summary,
            (origin_x, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self._TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _lab_to_bgr(lab: LabColor) -> tuple[int, int, int]:
        """Convert one OpenCV-scale LAB diagnostic tuple into a BGR swatch."""

        lab_pixel = np.asarray([[tuple(round(value) for value in lab)]], dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)[0, 0]
        return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])
