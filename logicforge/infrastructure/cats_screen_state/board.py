"""Board-state evidence for Cats screen-state classification."""

from dataclasses import dataclass

from logicforge.plugins.cats.screen_state import CatsScreenRect, CatsScreenState
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetectionError,
    CatsTileGridDetector,
)
from logicforge.vision.board_detector import BoardDetectionError, BoardDetector
from logicforge.vision.grid_detector import GridDetectionError, GridDetector
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class _BoardStateResult:
    """Retain primitive board evidence for public result assembly."""

    state: CatsScreenState
    confidence: float
    board_candidate: CatsScreenRect | None
    board_confidence: float | None
    grid_confidence: float | None
    rows: int | None
    columns: int | None
    rejection_reasons: tuple[str, ...]


class _BoardStateAnalyzer:
    """Run Cats tile-grid primary, then the generic geometry fallback."""

    def __init__(
        self,
        board_detector: BoardDetector,
        grid_detector: GridDetector,
        tile_grid_detector: CatsTileGridDetector | None,
    ) -> None:
        self._board_detector = board_detector
        self._grid_detector = grid_detector
        self._tile_grid_detector = tile_grid_detector

    def detect(
        self,
        screenshot: Screenshot,
        rejection_reasons: list[str],
    ) -> _BoardStateResult:
        """Classify full-frame board evidence without another capture pass."""

        board_rect: CatsScreenRect | None = None
        board_confidence: float | None = None
        grid_confidence: float | None = None
        rows: int | None = None
        columns: int | None = None
        if self._tile_grid_detector is not None:
            try:
                tile_grid = self._tile_grid_detector.detect(screenshot)
            except CatsTileGridDetectionError:
                rejection_reasons.append(
                    "Cats tile-grid detector rejected the screenshot"
                )
            else:
                board = tile_grid.board
                grid = tile_grid.grid
                board_rect = CatsScreenRect(
                    x=board.x,
                    y=board.y,
                    width=board.width,
                    height=board.height,
                )
                board_confidence = board.confidence
                grid_confidence = grid.confidence
                rows = grid.rows
                columns = grid.columns
                return _BoardStateResult(
                    state=CatsScreenState.BOARD,
                    confidence=min(board_confidence, grid_confidence),
                    board_candidate=board_rect,
                    board_confidence=board_confidence,
                    grid_confidence=grid_confidence,
                    rows=rows,
                    columns=columns,
                    rejection_reasons=tuple(rejection_reasons),
                )

        try:
            board = self._board_detector.detect(screenshot)
            board_rect = CatsScreenRect(
                x=board.x, y=board.y, width=board.width, height=board.height
            )
            board_confidence = board.confidence
        except BoardDetectionError:
            rejection_reasons.append("board detector rejected the screenshot")
        else:
            try:
                grid = self._grid_detector.detect(screenshot, board)
                grid_confidence = grid.confidence
                rows = grid.rows
                columns = grid.columns
            except GridDetectionError:
                rejection_reasons.append("grid detector rejected the board")
            else:
                return _BoardStateResult(
                    state=CatsScreenState.BOARD,
                    confidence=min(board_confidence, grid_confidence),
                    board_candidate=board_rect,
                    board_confidence=board_confidence,
                    grid_confidence=grid_confidence,
                    rows=rows,
                    columns=columns,
                    rejection_reasons=tuple(rejection_reasons),
                )

        return _BoardStateResult(
            state=CatsScreenState.UNKNOWN,
            confidence=0.0,
            board_candidate=board_rect,
            board_confidence=board_confidence,
            grid_confidence=grid_confidence,
            rows=rows,
            columns=columns,
            rejection_reasons=tuple(rejection_reasons),
        )
