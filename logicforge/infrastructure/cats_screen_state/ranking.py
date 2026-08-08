"""Ranking-card and stack detection for Cats screen-state classification."""

from dataclasses import dataclass
from itertools import combinations, pairwise
from statistics import fmean, pstdev
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.infrastructure.cats_screen_state.geometry import (
    relative_kernel,
    union_rect,
)
from logicforge.infrastructure.cats_screen_state.scoring import (
    clamp_unit as _clamp_unit,
)
from logicforge.infrastructure.cats_screen_state.settings import (
    CatsScreenStateDetectionSettings,
)
from logicforge.infrastructure.cats_screen_state.viewport import (
    _ViewportCandidate,
    _ViewportContext,
)
from logicforge.plugins.cats.screen_state import CatsScreenPoint, CatsScreenRect
from logicforge.vision.screenshot import Screenshot


@dataclass(frozen=True, slots=True)
class _RankingCard:
    """Retain one filtered viewport-relative bright-card rectangle."""

    rect: CatsScreenRect
    area: int
    fill_ratio: float
    contrast_score: float


@dataclass(frozen=True, slots=True)
class _RankingStack:
    """Retain one aligned top-to-bottom card combination and its score."""

    cards: tuple[_RankingCard, ...]
    score: float
    total_area: int
    centrality: float


class _RankingAnalyzer:
    """Detect deterministic aligned ranking-card stacks within one viewport."""

    def __init__(self, settings: CatsScreenStateDetectionSettings) -> None:
        self._settings = settings

    def find_ranking_stack(
        self,
        viewport: _ViewportContext,
    ) -> tuple[tuple[_RankingCard, ...], _RankingStack | None]:
        """Find the highest-scoring aligned card stack inside the viewport."""

        hsv = cast(NDArray[np.uint8], cv2.cvtColor(viewport.image, cv2.COLOR_BGR2HSV))
        neutral = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                (0, 0, self._settings.ranking_brightness_minimum),
                (179, self._settings.ranking_saturation_maximum, 255),
            ),
        )
        warm = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                (
                    0,
                    self._settings.ranking_warm_saturation_minimum,
                    self._settings.ranking_warm_brightness_minimum,
                ),
                (
                    self._settings.ranking_warm_hue_maximum,
                    self._settings.ranking_warm_saturation_maximum,
                    255,
                ),
            ),
        )
        card_mask = cast(NDArray[np.uint8], cv2.bitwise_or(neutral, warm))
        width = viewport.rect.width
        height = viewport.rect.height
        left = round(width * self._settings.ranking_region_left_ratio)
        right = round(width * self._settings.ranking_region_right_ratio)
        top = round(height * self._settings.ranking_region_top_ratio)
        bottom = round(height * self._settings.ranking_region_bottom_ratio)
        region_mask = np.zeros_like(card_mask)
        region_mask[top:bottom, left:right] = card_mask[top:bottom, left:right]
        close_kernel = self._ranking_close_kernel(width, height)
        processed = cast(
            NDArray[np.uint8],
            cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, close_kernel),
        )
        open_kernel = relative_kernel(width, height, 0.003)
        processed = cast(
            NDArray[np.uint8],
            cv2.morphologyEx(processed, cv2.MORPH_OPEN, open_kernel),
        )
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        grayscale = cast(
            NDArray[np.uint8], cv2.cvtColor(viewport.image, cv2.COLOR_BGR2GRAY)
        )
        cards = tuple(
            sorted(
                (
                    card
                    for contour in contours
                    if (
                        card := self._measure_ranking_card(
                            contour, processed, grayscale, viewport
                        )
                    )
                    is not None
                ),
                key=lambda card: (card.rect.y, card.rect.x, -card.area),
            )
        )
        if len(cards) > self._settings.ranking_maximum_card_count:
            return cards, None
        stacks: list[_RankingStack] = []
        for stack_size in range(
            self._settings.ranking_minimum_card_count, len(cards) + 1
        ):
            for card_combination in combinations(cards, stack_size):
                stack = self._measure_ranking_stack(card_combination, viewport.rect)
                if stack is not None:
                    stacks.append(stack)
        best = min(stacks, key=self._ranking_stack_sort_key) if stacks else None
        return cards, best

    def _measure_ranking_card(
        self,
        contour: cv2.typing.MatLike,
        processed_mask: NDArray[np.uint8],
        grayscale: NDArray[np.uint8],
        viewport: _ViewportContext,
    ) -> _RankingCard | None:
        """Filter one connected light component using viewport-relative evidence."""

        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        width_ratio = width / viewport.rect.width
        height_ratio = height / viewport.rect.height
        area_ratio = area / (viewport.rect.width * viewport.rect.height)
        aspect_ratio = width / height
        fill_ratio = (
            cv2.countNonZero(processed_mask[y : y + height, x : x + width]) / area
        )
        inside = float(np.mean(grayscale[y : y + height, x : x + width]))
        band = max(1, round(viewport.rect.height * 0.012))
        surroundings: list[NDArray[np.uint8]] = []
        if y > 0:
            surroundings.append(grayscale[max(0, y - band) : y, x : x + width])
        if y + height < viewport.rect.height:
            surroundings.append(
                grayscale[
                    y + height : min(viewport.rect.height, y + height + band),
                    x : x + width,
                ]
            )
        background = (
            fmean(float(np.mean(part)) for part in surroundings)
            if surroundings
            else float(np.median(grayscale))
        )
        contrast = _clamp_unit((inside - background) / 90.0)
        if not (
            self._settings.ranking_card_minimum_width_ratio
            <= width_ratio
            <= self._settings.ranking_card_maximum_width_ratio
            and self._settings.ranking_card_minimum_height_ratio
            <= height_ratio
            <= self._settings.ranking_card_maximum_height_ratio
            and self._settings.ranking_card_minimum_area_ratio
            <= area_ratio
            <= self._settings.ranking_card_maximum_area_ratio
            and self._settings.ranking_card_minimum_aspect_ratio
            <= aspect_ratio
            <= self._settings.ranking_card_maximum_aspect_ratio
            and fill_ratio >= self._settings.ranking_card_minimum_fill_ratio
            and contrast >= self._settings.ranking_card_minimum_contrast_score
        ):
            return None
        return _RankingCard(
            rect=CatsScreenRect(
                x=viewport.rect.x + x,
                y=viewport.rect.y + y,
                width=width,
                height=height,
            ),
            area=area,
            fill_ratio=fill_ratio,
            contrast_score=contrast,
        )

    def _measure_ranking_stack(
        self,
        cards: tuple[_RankingCard, ...],
        viewport_rect: CatsScreenRect,
    ) -> _RankingStack | None:
        """Validate alignment and spacing in viewport-relative coordinates."""

        ordered = tuple(sorted(cards, key=lambda card: (card.rect.y, card.rect.x)))
        widths = tuple(card.rect.width for card in ordered)
        mean_width = fmean(widths)
        width_variation = (max(widths) - min(widths)) / mean_width
        left_spread = (
            max(card.rect.x for card in ordered) - min(card.rect.x for card in ordered)
        ) / viewport_rect.width
        right_edges = tuple(card.rect.x + card.rect.width for card in ordered)
        right_spread = (max(right_edges) - min(right_edges)) / viewport_rect.width
        gaps = tuple(
            current.rect.y - (previous.rect.y + previous.rect.height)
            for previous, current in pairwise(ordered)
        )
        if any(
            gap < viewport_rect.height * self._settings.ranking_minimum_gap_ratio
            or gap > viewport_rect.height * self._settings.ranking_maximum_gap_ratio
            for gap in gaps
        ):
            return None
        gap_coefficient = (
            pstdev(gaps) / fmean(gaps) if len(gaps) >= 2 and fmean(gaps) > 0 else 0.0
        )
        if (
            width_variation > self._settings.ranking_maximum_width_variation
            or left_spread > self._settings.ranking_maximum_edge_alignment_ratio
            or right_spread > self._settings.ranking_maximum_edge_alignment_ratio
            or gap_coefficient
            > self._settings.ranking_maximum_gap_coefficient_of_variation
        ):
            return None

        union = union_rect(tuple(card.rect for card in ordered))
        width_similarity = _clamp_unit(1.0 - width_variation)
        alignment = _clamp_unit(
            1.0
            - max(left_spread, right_spread)
            / self._settings.ranking_maximum_edge_alignment_ratio
        )
        gap_regularity = _clamp_unit(
            1.0
            - gap_coefficient
            / self._settings.ranking_maximum_gap_coefficient_of_variation
        )
        contrast = fmean(card.contrast_score for card in ordered)
        local_union_center = union.center_x - viewport_rect.x
        centrality = _clamp_unit(
            1.0 - abs(local_union_center / viewport_rect.width - 0.5) / 0.5
        )
        count_score = _clamp_unit(len(ordered) / 3.0)
        score = _clamp_unit(
            0.20 * count_score
            + 0.20 * width_similarity
            + 0.20 * alignment
            + 0.15 * gap_regularity
            + 0.15 * contrast
            + 0.10 * centrality
        )
        return _RankingStack(
            cards=ordered,
            score=score,
            total_area=sum(card.area for card in ordered),
            centrality=centrality,
        )

    @staticmethod
    def _ranking_stack_sort_key(
        stack: _RankingStack,
    ) -> tuple[float, int, int, float, int, int]:
        """Order stacks independently from contour and combination enumeration."""

        return (
            -stack.score,
            -len(stack.cards),
            -stack.total_area,
            -stack.centrality,
            stack.cards[0].rect.y,
            stack.cards[0].rect.x,
        )

    def action_point(
        self,
        screenshot: Screenshot,
        viewport: _ViewportCandidate,
        stack: _RankingStack,
    ) -> CatsScreenPoint:
        """Map one accepted stack to its unchanged global action geometry."""

        rectangles = tuple(card.rect for card in stack.cards)
        union = union_rect(rectangles)
        local_union_bottom = union.y + union.height - viewport.rect.y
        local_action_y = max(
            local_union_bottom
            + round(viewport.rect.height * self._settings.ranking_action_margin_ratio),
            round(viewport.rect.height * self._settings.ranking_action_minimum_y_ratio),
        )
        local_action_y = min(
            local_action_y,
            viewport.rect.height - 1,
            round(viewport.rect.height * self._settings.ranking_action_maximum_y_ratio),
        )
        action_y = min(screenshot.height - 1, viewport.rect.y + local_action_y)
        action_x = min(screenshot.width - 1, max(0, union.center_x))
        return CatsScreenPoint(x=action_x, y=action_y)

    def _ranking_close_kernel(self, width: int, height: int) -> NDArray[np.uint8]:
        """Close horizontal card interruptions without joining separate card rows."""

        kernel_width = max(
            3,
            round(width * self._settings.ranking_morphology_horizontal_kernel_ratio),
        )
        kernel_height = max(
            1,
            round(height * self._settings.ranking_morphology_vertical_kernel_ratio),
        )
        if kernel_width % 2 == 0:
            kernel_width += 1
        if kernel_height % 2 == 0:
            kernel_height += 1
        return cast(
            NDArray[np.uint8],
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height)),
        )
