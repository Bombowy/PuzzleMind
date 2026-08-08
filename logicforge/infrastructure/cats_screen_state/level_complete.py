"""Warm-CTA detection for the LEVEL_COMPLETE Cats screen state."""

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.infrastructure.cats_screen_state.geometry import relative_kernel
from logicforge.infrastructure.cats_screen_state.scoring import (
    clamp_unit as _clamp_unit,
)
from logicforge.infrastructure.cats_screen_state.scoring import (
    threshold_score as _threshold_score,
)
from logicforge.infrastructure.cats_screen_state.scoring import (
    triangular_score as _triangular_score,
)
from logicforge.infrastructure.cats_screen_state.settings import (
    CatsScreenStateDetectionSettings,
)
from logicforge.infrastructure.cats_screen_state.viewport import _ViewportContext
from logicforge.plugins.cats.screen_state import CatsScreenRect


@dataclass(frozen=True, slots=True)
class _LevelButtonCandidate:
    """Retain primitive level-button evidence for deterministic selection."""

    rect: CatsScreenRect
    score: float
    warm_fill_ratio: float
    rectangularity: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


class _LevelCompleteAnalyzer:
    """Detect and rank viewport-relative warm CTA candidates."""

    def __init__(self, settings: CatsScreenStateDetectionSettings) -> None:
        self._settings = settings

    def find_level_button(
        self,
        viewport: _ViewportContext,
    ) -> _LevelButtonCandidate | None:
        """Find the best lower red/orange CTA relative to the game viewport."""

        hsv = cast(NDArray[np.uint8], cv2.cvtColor(viewport.image, cv2.COLOR_BGR2HSV))
        low_hue_mask = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                (
                    self._settings.level_warm_cta_hue_minimum,
                    self._settings.level_warm_cta_saturation_minimum,
                    self._settings.level_warm_cta_value_minimum,
                ),
                (self._settings.level_warm_cta_hue_maximum, 255, 255),
            ),
        )
        red_wrap_mask = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                (
                    self._settings.level_warm_cta_red_wrap_hue_minimum,
                    self._settings.level_warm_cta_saturation_minimum,
                    self._settings.level_warm_cta_value_minimum,
                ),
                (179, 255, 255),
            ),
        )
        mask = cast(NDArray[np.uint8], cv2.bitwise_or(low_hue_mask, red_wrap_mask))
        region_start = round(
            viewport.rect.height * self._settings.level_region_start_y_ratio
        )
        mask[:region_start, :] = 0
        kernel = relative_kernel(
            viewport.rect.width,
            viewport.rect.height,
            self._settings.level_morphology_kernel_ratio,
        )
        processed = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=self._settings.level_morphology_iterations,
        )
        processed = cv2.morphologyEx(
            processed,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self._settings.level_morphology_iterations,
        )
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates = tuple(
            self._measure_level_candidate(contour, mask, viewport)
            for contour in contours
            if cv2.contourArea(contour) > 0
        )
        accepted = tuple(candidate for candidate in candidates if candidate.accepted)
        pool = accepted or candidates
        return min(pool, key=self._level_sort_key) if pool else None

    def _measure_level_candidate(
        self,
        contour: cv2.typing.MatLike,
        warm_mask: NDArray[np.uint8],
        viewport: _ViewportContext,
    ) -> _LevelButtonCandidate:
        """Measure locally, then translate the retained rect to screenshot space."""

        x, y, width, height = cv2.boundingRect(contour)
        local_rect = CatsScreenRect(x=x, y=y, width=width, height=height)
        rectangle_area = width * height
        viewport_area = viewport.rect.width * viewport.rect.height
        width_ratio = width / viewport.rect.width
        height_ratio = height / viewport.rect.height
        area_ratio = rectangle_area / viewport_area
        aspect_ratio = width / height
        center_y_ratio = local_rect.center_y / viewport.rect.height
        rectangularity = _clamp_unit(abs(cv2.contourArea(contour)) / rectangle_area)
        warm_fill = (
            cv2.countNonZero(warm_mask[y : y + height, x : x + width]) / rectangle_area
        )
        width_score = _triangular_score(
            width_ratio,
            self._settings.level_button_minimum_width_ratio,
            0.70,
            self._settings.level_button_maximum_width_ratio,
        )
        height_score = _triangular_score(
            height_ratio,
            self._settings.level_button_minimum_height_ratio,
            0.09,
            self._settings.level_button_maximum_height_ratio,
        )
        position_score = _triangular_score(
            center_y_ratio,
            self._settings.level_button_minimum_center_y_ratio,
            0.90,
            self._settings.level_button_maximum_center_y_ratio,
        )
        aspect_score = _triangular_score(
            aspect_ratio,
            self._settings.level_button_minimum_aspect_ratio,
            7.0,
            self._settings.level_button_maximum_aspect_ratio,
        )
        fill_score = _threshold_score(
            warm_fill, self._settings.level_button_minimum_warm_fill_ratio
        )
        rectangularity_score = _threshold_score(
            rectangularity, self._settings.level_button_minimum_rectangularity
        )
        score = _clamp_unit(
            0.20 * position_score
            + 0.15 * width_score
            + 0.10 * height_score
            + 0.15 * aspect_score
            + 0.25 * fill_score
            + 0.15 * rectangularity_score
        )
        reasons: list[str] = []
        if width_ratio < self._settings.level_button_minimum_width_ratio:
            reasons.append(
                "level button candidate width was below viewport-relative minimum"
            )
        elif width_ratio > self._settings.level_button_maximum_width_ratio:
            reasons.append(
                "level button candidate width exceeded viewport-relative maximum"
            )
        if not (
            self._settings.level_button_minimum_height_ratio
            <= height_ratio
            <= self._settings.level_button_maximum_height_ratio
        ):
            reasons.append("level button height was outside viewport-relative range")
        if not (
            self._settings.level_button_minimum_center_y_ratio
            <= center_y_ratio
            <= self._settings.level_button_maximum_center_y_ratio
        ):
            reasons.append("level button center was outside the lower viewport region")
        if not (
            self._settings.level_button_minimum_aspect_ratio
            <= aspect_ratio
            <= self._settings.level_button_maximum_aspect_ratio
        ):
            reasons.append("level button aspect ratio was outside range")
        if not (
            self._settings.level_button_minimum_area_ratio
            <= area_ratio
            <= self._settings.level_button_maximum_area_ratio
        ):
            reasons.append("level button area was outside viewport-relative range")
        if warm_fill < self._settings.level_button_minimum_warm_fill_ratio:
            reasons.append("level button warm CTA fill was below threshold")
        if rectangularity < self._settings.level_button_minimum_rectangularity:
            reasons.append("level button rectangularity was below threshold")
        if score < self._settings.level_button_acceptance_score:
            reasons.append("level button score was below threshold")
        global_rect = CatsScreenRect(
            x=viewport.rect.x + x,
            y=viewport.rect.y + y,
            width=width,
            height=height,
        )
        return _LevelButtonCandidate(
            rect=global_rect,
            score=score,
            warm_fill_ratio=warm_fill,
            rectangularity=rectangularity,
            accepted=not reasons,
            rejection_reasons=tuple(reasons),
        )

    @staticmethod
    def _level_sort_key(
        candidate: _LevelButtonCandidate,
    ) -> tuple[float, float, int, int, int]:
        """Order level candidates independent of contour enumeration order."""

        return (
            -candidate.score,
            -candidate.warm_fill_ratio,
            -candidate.rect.width,
            -candidate.rect.center_y,
            candidate.rect.x,
        )
