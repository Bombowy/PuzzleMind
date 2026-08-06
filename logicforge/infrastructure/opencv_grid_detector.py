"""OpenCV adapter exposing validated grid boundaries and deterministic cells."""

from itertools import pairwise
from math import floor, isfinite
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import (
    BoardDetectionSettings,
    GridExtractionSettings,
)
from logicforge.infrastructure.opencv_internal_grid_evidence import (
    InternalGridEvidence,
    InternalGridEvidenceAnalyzer,
    OpenCvInternalGridEvidenceAnalyzer,
)
from logicforge.vision.board_detector import BoardDetection
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionDiagnostics,
    GridDetectionError,
    GridDetector,
)
from logicforge.vision.screenshot import Screenshot


class OpenCvGridDetector(GridDetector):
    """Convert shared normalized grid evidence into screenshot-space cell geometry.

    The adapter does not detect lines independently. It delegates ROI analysis and
    mandatory evidence rules to ``OpenCvInternalGridEvidenceAnalyzer``, which is the
    same implementation used by ``OpenCvBoardDetector``. This class owns only board
    validation, half-up pixel conversion, strict monotonicity, and cell generation.
    """

    def __init__(
        self,
        board_settings: BoardDetectionSettings | None = None,
        extraction_settings: GridExtractionSettings | None = None,
        analyzer: InternalGridEvidenceAnalyzer | None = None,
    ) -> None:
        """Compose shared evidence settings with extraction-only pixel constraints."""

        self._board_settings = board_settings or BoardDetectionSettings()
        self._extraction_settings = extraction_settings or GridExtractionSettings()
        self._analyzer = analyzer or OpenCvInternalGridEvidenceAnalyzer(
            self._board_settings
        )

    def detect(self, screenshot: Screenshot, board: BoardDetection) -> GridDetection:
        """Return complete geometry or fail without partial cells or guessed lines."""

        board_rejections = self._validate_board(screenshot, board)
        if board_rejections:
            self._raise_detection_error(
                board,
                evidence=None,
                horizontal_lines=(),
                vertical_lines=(),
                rejection_reasons=board_rejections,
            )

        roi = screenshot.image[
            board.y : board.y + board.height,
            board.x : board.x + board.width,
        ]
        grayscale_roi = cast(
            NDArray[np.uint8],
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
        )
        evidence = self._analyzer.analyze(grayscale_roi)
        horizontal_lines, horizontal_rejections = self._convert_boundaries(
            evidence.horizontal_line_positions,
            offset=board.y,
            extent=board.height,
            axis_name="horizontal",
        )
        vertical_lines, vertical_rejections = self._convert_boundaries(
            evidence.vertical_line_positions,
            offset=board.x,
            extent=board.width,
            axis_name="vertical",
        )
        rejection_reasons = (
            *self._analyzer.rejection_reasons(evidence),
            *horizontal_rejections,
            *vertical_rejections,
        )
        if rejection_reasons:
            self._raise_detection_error(
                board,
                evidence=evidence,
                horizontal_lines=horizontal_lines,
                vertical_lines=vertical_lines,
                rejection_reasons=rejection_reasons,
            )

        cells, cell_rejections = self._create_cells(
            horizontal_lines,
            vertical_lines,
            board,
        )
        if cell_rejections:
            self._raise_detection_error(
                board,
                evidence=evidence,
                horizontal_lines=horizontal_lines,
                vertical_lines=vertical_lines,
                rejection_reasons=cell_rejections,
            )

        rows = len(horizontal_lines) - 1
        columns = len(vertical_lines) - 1
        expected_cell_count = rows * columns
        if len(cells) != expected_cell_count:
            self._raise_detection_error(
                board,
                evidence=evidence,
                horizontal_lines=horizontal_lines,
                vertical_lines=vertical_lines,
                rejection_reasons=(
                    "cell count does not equal estimated rows times columns",
                ),
            )
        return GridDetection(
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            rows=rows,
            columns=columns,
            cells=cells,
            confidence=evidence.score,
        )

    @staticmethod
    def _validate_board(
        screenshot: Screenshot,
        board: BoardDetection,
    ) -> tuple[str, ...]:
        """Validate the complete board rectangle before any NumPy ROI slicing."""

        reasons: list[str] = []
        if board.x < 0 or board.y < 0:
            reasons.append("board x and y must be non-negative")
        if board.width <= 0 or board.height <= 0:
            reasons.append("board width and height must be positive")
        if board.x + board.width > screenshot.width:
            reasons.append("board right edge exceeds screenshot width")
        if board.y + board.height > screenshot.height:
            reasons.append("board bottom edge exceeds screenshot height")
        if not isfinite(board.confidence) or not 0.0 <= board.confidence <= 1.0:
            reasons.append("board confidence must be finite and within 0.0 and 1.0")
        return tuple(reasons)

    @staticmethod
    def _convert_boundaries(
        normalized_positions: tuple[float, ...],
        *,
        offset: int,
        extent: int,
        axis_name: str,
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Convert normalized boundaries with deterministic round-half-up semantics.

        Outer boundaries are fixed exactly to ``offset`` and ``offset + extent``.
        Interior values use ``floor(value * extent + 0.5)``. No adjustment is made
        to repair duplicates because doing so would fabricate separator geometry.
        """

        if len(normalized_positions) < 2:
            return (), (f"{axis_name} normalized boundaries are incomplete",)
        reasons: list[str] = []
        if normalized_positions[0] != 0.0 or normalized_positions[-1] != 1.0:
            reasons.append(
                f"{axis_name} normalized boundaries must start at 0.0 and end at 1.0"
            )
        if any(
            not isfinite(position) or not 0.0 <= position <= 1.0
            for position in normalized_positions
        ):
            reasons.append(
                f"{axis_name} normalized boundaries must be finite within 0.0 and 1.0"
            )
        if any(
            current <= previous for previous, current in pairwise(normalized_positions)
        ):
            reasons.append(
                f"{axis_name} normalized boundaries are not strictly increasing"
            )
        if reasons:
            return (), tuple(reasons)

        converted = (
            offset,
            *(
                offset + floor(position * extent + 0.5)
                for position in normalized_positions[1:-1]
            ),
            offset + extent,
        )
        if any(current <= previous for previous, current in pairwise(converted)):
            reasons.append(
                f"{axis_name} boundaries collapse to duplicate or reversed pixels"
            )
        return converted, tuple(reasons)

    def _create_cells(
        self,
        horizontal_lines: tuple[int, ...],
        vertical_lines: tuple[int, ...],
        board: BoardDetection,
    ) -> tuple[tuple[CellBounds, ...], tuple[str, ...]]:
        """Generate validated row-major half-open rectangles from adjacent lines."""

        reasons: list[str] = []
        cells: list[CellBounds] = []
        for row, (top, bottom) in enumerate(pairwise(horizontal_lines)):
            height = bottom - top
            if height < self._extraction_settings.minimum_cell_height_pixels:
                reasons.append(
                    f"cell row {row} height is below minimum pixel requirement"
                )
                continue
            for column, (left, right) in enumerate(pairwise(vertical_lines)):
                width = right - left
                if width < self._extraction_settings.minimum_cell_width_pixels:
                    reasons.append(
                        f"cell column {column} width is below minimum pixel requirement"
                    )
                    continue
                if (
                    left < board.x
                    or top < board.y
                    or right > board.x + board.width
                    or bottom > board.y + board.height
                ):
                    reasons.append(f"cell ({row}, {column}) lies outside board bounds")
                    continue
                cells.append(
                    CellBounds(
                        row=row,
                        column=column,
                        x=left,
                        y=top,
                        width=width,
                        height=height,
                        center_x=left + width // 2,
                        center_y=top + height // 2,
                    )
                )
        return tuple(cells), tuple(dict.fromkeys(reasons))

    @staticmethod
    def _raise_detection_error(
        board: BoardDetection,
        *,
        evidence: InternalGridEvidence | None,
        horizontal_lines: tuple[int, ...],
        vertical_lines: tuple[int, ...],
        rejection_reasons: tuple[str, ...],
    ) -> None:
        """Raise one typed error containing every available primitive diagnostic."""

        diagnostics = GridDetectionDiagnostics(
            board_x=board.x,
            board_y=board.y,
            board_width=board.width,
            board_height=board.height,
            normalized_horizontal_positions=(
                evidence.horizontal_line_positions if evidence is not None else ()
            ),
            normalized_vertical_positions=(
                evidence.vertical_line_positions if evidence is not None else ()
            ),
            horizontal_lines=horizontal_lines,
            vertical_lines=vertical_lines,
            estimated_rows=evidence.estimated_rows if evidence is not None else 0,
            estimated_columns=(
                evidence.estimated_columns if evidence is not None else 0
            ),
            horizontal_spacing_coefficient_of_variation=(
                evidence.horizontal_spacing_coefficient_of_variation
                if evidence is not None
                else 1.0
            ),
            vertical_spacing_coefficient_of_variation=(
                evidence.vertical_spacing_coefficient_of_variation
                if evidence is not None
                else 1.0
            ),
            horizontal_coverage=(
                evidence.horizontal_line_coverage if evidence is not None else 0.0
            ),
            vertical_coverage=(
                evidence.vertical_line_coverage if evidence is not None else 0.0
            ),
            grid_evidence_score=evidence.score if evidence is not None else 0.0,
            rejection_reasons=rejection_reasons,
        )
        reason_summary = "; ".join(rejection_reasons)
        raise GridDetectionError(
            f"Grid detection failed: {reason_summary}.",
            diagnostics,
        )
