"""Public OpenCV facade for viewport-aware Cats screen classification."""

import cv2

from logicforge.config.settings import BoardDetectionSettings
from logicforge.infrastructure.cats_screen_state.board import (
    _BoardStateAnalyzer,
    _BoardStateResult,
)
from logicforge.infrastructure.cats_screen_state.level_complete import (
    _LevelButtonCandidate,
    _LevelCompleteAnalyzer,
)
from logicforge.infrastructure.cats_screen_state.ranking import (
    _RankingAnalyzer,
    _RankingCard,
    _RankingStack,
)
from logicforge.infrastructure.cats_screen_state.scoring import (
    clamp_unit as _clamp_unit,
)
from logicforge.infrastructure.cats_screen_state.settings import (
    CatsScreenStateDetectionSettings,
)
from logicforge.infrastructure.cats_screen_state.viewport import (
    _ViewportAnalyzer,
    _ViewportCandidate,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_grid_detector import OpenCvGridDetector
from logicforge.plugins.cats.screen_state import (
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDetector,
    CatsScreenStateDiagnostics,
)
from logicforge.plugins.cats.tile_grid import CatsTileGridDetector
from logicforge.vision.board_detector import BoardDetector
from logicforge.vision.grid_detector import GridDetector
from logicforge.vision.screenshot import Screenshot


class CatsScreenStateDetectionError(RuntimeError):
    """Report an actual processing failure rather than an unrecognized screen."""


class OpenCvCatsScreenStateDetector(CatsScreenStateDetector):
    """Coordinate viewport, transition, and board classifiers in priority order."""

    def __init__(
        self,
        settings: CatsScreenStateDetectionSettings | None = None,
        board_detector: BoardDetector | None = None,
        grid_detector: GridDetector | None = None,
        tile_grid_detector: CatsTileGridDetector | None = None,
    ) -> None:
        """Use tile-grid-first Cats geometry with an injectable generic fallback."""

        self._settings = settings or CatsScreenStateDetectionSettings()
        board_settings = BoardDetectionSettings()
        resolved_board_detector = board_detector or OpenCvBoardDetector(board_settings)
        resolved_grid_detector = grid_detector or OpenCvGridDetector(board_settings)
        legacy_geometry_was_injected = (
            board_detector is not None or grid_detector is not None
        )
        resolved_tile_grid_detector = (
            tile_grid_detector
            if tile_grid_detector is not None
            else (
                None if legacy_geometry_was_injected else OpenCvCatsTileGridDetector()
            )
        )
        self._viewport = _ViewportAnalyzer(self._settings)
        self._level_complete = _LevelCompleteAnalyzer(self._settings)
        self._ranking = _RankingAnalyzer(self._settings)
        self._board = _BoardStateAnalyzer(
            resolved_board_detector,
            resolved_grid_detector,
            resolved_tile_grid_detector,
        )

    def detect(self, screenshot: Screenshot) -> CatsScreenStateDetection:
        """Classify in LEVEL_COMPLETE, RANKING, BOARD, UNKNOWN priority order."""

        rejection_reasons: list[str] = []
        level_candidate: _LevelButtonCandidate | None = None
        ranking_cards: tuple[_RankingCard, ...] = ()
        ranking_score = 0.0
        try:
            viewport_search = self._viewport.find_game_viewport(screenshot)
            viewport_candidate = viewport_search.candidate
            if viewport_candidate is None:
                rejection_reasons.extend(viewport_search.rejection_reasons)
                rejection_reasons.append("no reliable Cats game viewport was found")
            else:
                context = self._viewport.viewport_context(
                    screenshot, viewport_candidate.rect
                )
                level_candidate = self._level_complete.find_level_button(context)
                if level_candidate is not None and level_candidate.accepted:
                    return self._level_complete_detection(
                        viewport_candidate,
                        level_candidate,
                    )
                if level_candidate is None:
                    rejection_reasons.append(
                        "no viewport-relative level button candidate was found"
                    )
                else:
                    rejection_reasons.extend(level_candidate.rejection_reasons)

                ranking_cards, ranking_stack = self._ranking.find_ranking_stack(context)
                if ranking_stack is not None:
                    ranking_score = ranking_stack.score
                if (
                    ranking_stack is not None
                    and ranking_stack.score >= self._settings.ranking_acceptance_score
                ):
                    return self._ranking_detection(
                        screenshot,
                        viewport_candidate,
                        level_candidate,
                        ranking_stack,
                        tuple(rejection_reasons),
                    )
                if len(ranking_cards) == 1:
                    rejection_reasons.append(
                        "only one viewport-relative ranking card was accepted"
                    )
                elif ranking_stack is not None:
                    rejection_reasons.append(
                        "ranking card stack score was below threshold"
                    )
                elif len(ranking_cards) >= 2:
                    rejection_reasons.append(
                        "viewport-relative ranking cards were not sufficiently aligned"
                    )
                else:
                    rejection_reasons.append(
                        "no viewport-relative ranking cards passed geometry"
                    )
        except cv2.error as error:
            raise CatsScreenStateDetectionError(
                "OpenCV could not analyze Cats viewport or transition geometry."
            ) from error

        try:
            board_result = self._board.detect(screenshot, rejection_reasons)
        except cv2.error as error:
            raise CatsScreenStateDetectionError(
                "OpenCV could not analyze Cats board or grid geometry."
            ) from error
        return self._board_detection(
            board_result,
            viewport_search.candidate,
            viewport_search.best_score,
            level_candidate,
            ranking_cards,
            ranking_score,
        )

    def _level_complete_detection(
        self,
        viewport: _ViewportCandidate,
        candidate: _LevelButtonCandidate,
    ) -> CatsScreenStateDetection:
        """Create the highest-priority result at the global button center."""

        diagnostics = self._diagnostics(
            viewport_candidate=viewport,
            viewport_score=viewport.score,
            level_candidate=candidate,
            ranking_cards=(),
            ranking_score=0.0,
            rejection_reasons=(),
        )
        return CatsScreenStateDetection(
            state=CatsScreenState.LEVEL_COMPLETE,
            confidence=candidate.score,
            action_point=CatsScreenPoint(
                x=candidate.rect.center_x, y=candidate.rect.center_y
            ),
            diagnostics=diagnostics,
        )

    def _ranking_detection(
        self,
        screenshot: Screenshot,
        viewport: _ViewportCandidate,
        level_candidate: _LevelButtonCandidate | None,
        stack: _RankingStack,
        rejection_reasons: tuple[str, ...],
    ) -> CatsScreenStateDetection:
        """Create a ranking result using the feature-owned action geometry."""

        diagnostics = self._diagnostics(
            viewport_candidate=viewport,
            viewport_score=viewport.score,
            level_candidate=level_candidate,
            ranking_cards=stack.cards,
            ranking_score=stack.score,
            rejection_reasons=rejection_reasons,
        )
        return CatsScreenStateDetection(
            state=CatsScreenState.RANKING,
            confidence=stack.score,
            action_point=self._ranking.action_point(screenshot, viewport, stack),
            diagnostics=diagnostics,
        )

    def _board_detection(
        self,
        result: _BoardStateResult,
        viewport_candidate: _ViewportCandidate | None,
        viewport_score: float,
        level_candidate: _LevelButtonCandidate | None,
        ranking_cards: tuple[_RankingCard, ...],
        ranking_score: float,
    ) -> CatsScreenStateDetection:
        """Assemble BOARD or UNKNOWN from primitive analyzer evidence."""

        diagnostics = self._diagnostics(
            viewport_candidate=viewport_candidate,
            viewport_score=viewport_score,
            level_candidate=level_candidate,
            ranking_cards=ranking_cards,
            ranking_score=ranking_score,
            board_candidate=result.board_candidate,
            board_confidence=result.board_confidence,
            grid_confidence=result.grid_confidence,
            rows=result.rows,
            columns=result.columns,
            rejection_reasons=result.rejection_reasons,
        )
        return CatsScreenStateDetection(
            state=result.state,
            confidence=result.confidence,
            action_point=None,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostics(
        *,
        viewport_candidate: _ViewportCandidate | None,
        viewport_score: float,
        level_candidate: _LevelButtonCandidate | None,
        ranking_cards: tuple[_RankingCard, ...],
        ranking_score: float,
        board_candidate: CatsScreenRect | None = None,
        board_confidence: float | None = None,
        grid_confidence: float | None = None,
        rows: int | None = None,
        columns: int | None = None,
        rejection_reasons: tuple[str, ...] = (),
    ) -> CatsScreenStateDiagnostics:
        """Build public diagnostics from immutable primitive screenshot geometry."""

        return CatsScreenStateDiagnostics(
            game_viewport_candidate=(
                viewport_candidate.rect if viewport_candidate is not None else None
            ),
            game_viewport_score=_clamp_unit(viewport_score),
            level_button_candidate=(
                level_candidate.rect if level_candidate is not None else None
            ),
            level_button_score=(
                level_candidate.score if level_candidate is not None else 0.0
            ),
            ranking_card_candidates=tuple(card.rect for card in ranking_cards),
            ranking_score=_clamp_unit(ranking_score),
            board_candidate=board_candidate,
            board_confidence=board_confidence,
            grid_confidence=grid_confidence,
            detected_rows=rows,
            detected_columns=columns,
            rejection_reasons=rejection_reasons,
        )


__all__ = [
    "CatsScreenStateDetectionError",
    "CatsScreenStateDetectionSettings",
    "OpenCvCatsScreenStateDetector",
]
