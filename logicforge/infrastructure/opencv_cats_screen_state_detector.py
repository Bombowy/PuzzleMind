"""Classical OpenCV adapter for viewport-aware Cats screen classification."""

from dataclasses import dataclass, fields
from itertools import combinations, pairwise
from math import isfinite
from statistics import fmean, pstdev
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.config.settings import BoardDetectionSettings
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
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetectionError,
    CatsTileGridDetector,
)
from logicforge.vision.board_detector import BoardDetectionError, BoardDetector
from logicforge.vision.grid_detector import GridDetectionError, GridDetector
from logicforge.vision.screenshot import Screenshot


class CatsScreenStateDetectionError(RuntimeError):
    """Report an actual processing failure rather than an unrecognized screen."""


@dataclass(frozen=True, slots=True)
class CatsScreenStateDetectionSettings:
    """Configure scale-relative viewport and transition-overlay heuristics."""

    viewport_minimum_height_ratio: float = 0.88
    viewport_minimum_aspect_ratio: float = 0.48
    viewport_preferred_aspect_ratio: float = 9.0 / 16.0
    viewport_maximum_aspect_ratio: float = 0.66
    viewport_content_top_search_ratio: float = 0.10
    viewport_content_top_minimum_ratio: float = 0.01
    viewport_content_top_maximum_ratio: float = 0.08
    viewport_content_top_fallback_ratio: float = 0.032
    viewport_minimum_content_top_boundary_score: float = 0.12
    viewport_boundary_probe_ratio: float = 0.006
    viewport_boundary_smoothing_ratio: float = 0.006
    viewport_minimum_boundary_score: float = 0.08
    viewport_minimum_internal_activity: float = 0.12
    viewport_minimum_score: float = 0.54

    level_orange_hue_minimum: int = 5
    level_orange_hue_maximum: int = 28
    level_orange_saturation_minimum: int = 145
    level_orange_value_minimum: int = 120
    level_region_start_y_ratio: float = 0.60
    level_button_minimum_width_ratio: float = 0.55
    level_button_maximum_width_ratio: float = 0.92
    level_button_minimum_height_ratio: float = 0.055
    level_button_maximum_height_ratio: float = 0.15
    level_button_minimum_center_y_ratio: float = 0.78
    level_button_maximum_center_y_ratio: float = 0.98
    level_button_minimum_aspect_ratio: float = 3.8
    level_button_maximum_aspect_ratio: float = 10.5
    level_button_minimum_area_ratio: float = 0.03
    level_button_maximum_area_ratio: float = 0.14
    level_button_minimum_orange_fill_ratio: float = 0.48
    level_button_minimum_rectangularity: float = 0.50
    level_button_acceptance_score: float = 0.60
    level_morphology_kernel_ratio: float = 0.009
    level_morphology_iterations: int = 1

    ranking_brightness_minimum: int = 170
    ranking_saturation_maximum: int = 105
    ranking_warm_hue_maximum: int = 35
    ranking_warm_saturation_minimum: int = 20
    ranking_warm_saturation_maximum: int = 190
    ranking_warm_brightness_minimum: int = 160
    ranking_region_left_ratio: float = 0.03
    ranking_region_right_ratio: float = 0.97
    ranking_region_top_ratio: float = 0.20
    ranking_region_bottom_ratio: float = 0.80
    ranking_card_minimum_width_ratio: float = 0.55
    ranking_card_maximum_width_ratio: float = 0.95
    ranking_card_minimum_height_ratio: float = 0.05
    ranking_card_maximum_height_ratio: float = 0.17
    ranking_card_minimum_area_ratio: float = 0.025
    ranking_card_maximum_area_ratio: float = 0.15
    ranking_card_minimum_aspect_ratio: float = 3.0
    ranking_card_maximum_aspect_ratio: float = 13.0
    ranking_card_minimum_fill_ratio: float = 0.42
    ranking_card_minimum_contrast_score: float = 0.18
    ranking_maximum_width_variation: float = 0.18
    ranking_maximum_edge_alignment_ratio: float = 0.055
    ranking_minimum_gap_ratio: float = 0.006
    ranking_maximum_gap_ratio: float = 0.14
    ranking_maximum_gap_coefficient_of_variation: float = 0.55
    ranking_minimum_card_count: int = 2
    ranking_maximum_card_count: int = 3
    ranking_acceptance_score: float = 0.64
    ranking_morphology_horizontal_kernel_ratio: float = 0.055
    ranking_morphology_vertical_kernel_ratio: float = 0.003
    ranking_action_margin_ratio: float = 0.035
    ranking_action_minimum_y_ratio: float = 0.80
    ranking_action_maximum_y_ratio: float = 0.93

    def __post_init__(self) -> None:
        """Reject invalid colors, ratios, scores, ranges, and divisor settings."""

        if not (
            0 <= self.level_orange_hue_minimum < self.level_orange_hue_maximum <= 179
        ):
            raise ValueError(
                "Orange hue thresholds must satisfy 0 <= min < max <= 179."
            )
        if not 0 <= self.ranking_warm_hue_maximum <= 179:
            raise ValueError("ranking_warm_hue_maximum must be within 0 and 179.")
        byte_fields = {
            "level_orange_saturation_minimum": self.level_orange_saturation_minimum,
            "level_orange_value_minimum": self.level_orange_value_minimum,
            "ranking_brightness_minimum": self.ranking_brightness_minimum,
            "ranking_saturation_maximum": self.ranking_saturation_maximum,
            "ranking_warm_saturation_minimum": (self.ranking_warm_saturation_minimum),
            "ranking_warm_saturation_maximum": (self.ranking_warm_saturation_maximum),
            "ranking_warm_brightness_minimum": (self.ranking_warm_brightness_minimum),
        }
        for name, byte_value in byte_fields.items():
            if not 0 <= byte_value <= 255:
                raise ValueError(f"{name} must be within 0 and 255.")
        if self.ranking_warm_saturation_minimum >= self.ranking_warm_saturation_maximum:
            raise ValueError("Warm saturation thresholds must be ordered.")

        unit_fields = {
            field.name: cast(float, getattr(self, field.name))
            for field in fields(self)
            if (field.name.endswith("_ratio") and "aspect_ratio" not in field.name)
            or field.name.endswith("_score")
        }
        for name, unit_value in unit_fields.items():
            if not isfinite(unit_value) or not 0.0 <= unit_value <= 1.0:
                raise ValueError(f"{name} must be finite within 0.0 and 1.0.")

        additional_unit_fields = {
            "ranking_maximum_width_variation": self.ranking_maximum_width_variation,
            "ranking_maximum_gap_coefficient_of_variation": (
                self.ranking_maximum_gap_coefficient_of_variation
            ),
        }
        for name, tolerance_value in additional_unit_fields.items():
            if not isfinite(tolerance_value) or not 0.0 <= tolerance_value <= 1.0:
                raise ValueError(f"{name} must be finite within 0.0 and 1.0.")

        ordered_unit_ranges = (
            (
                self.viewport_content_top_minimum_ratio,
                self.viewport_content_top_maximum_ratio,
                "viewport content-top search bounds",
            ),
            (
                self.level_button_minimum_width_ratio,
                self.level_button_maximum_width_ratio,
                "level button width",
            ),
            (
                self.level_button_minimum_height_ratio,
                self.level_button_maximum_height_ratio,
                "level button height",
            ),
            (
                self.level_button_minimum_center_y_ratio,
                self.level_button_maximum_center_y_ratio,
                "level button center y",
            ),
            (
                self.level_button_minimum_area_ratio,
                self.level_button_maximum_area_ratio,
                "level button area",
            ),
            (
                self.ranking_region_left_ratio,
                self.ranking_region_right_ratio,
                "ranking horizontal region",
            ),
            (
                self.ranking_region_top_ratio,
                self.ranking_region_bottom_ratio,
                "ranking vertical region",
            ),
            (
                self.ranking_card_minimum_width_ratio,
                self.ranking_card_maximum_width_ratio,
                "ranking card width",
            ),
            (
                self.ranking_card_minimum_height_ratio,
                self.ranking_card_maximum_height_ratio,
                "ranking card height",
            ),
            (
                self.ranking_card_minimum_area_ratio,
                self.ranking_card_maximum_area_ratio,
                "ranking card area",
            ),
            (
                self.ranking_minimum_gap_ratio,
                self.ranking_maximum_gap_ratio,
                "ranking gap",
            ),
            (
                self.ranking_action_minimum_y_ratio,
                self.ranking_action_maximum_y_ratio,
                "ranking action y",
            ),
        )
        for minimum, maximum, name in ordered_unit_ranges:
            if minimum >= maximum:
                raise ValueError(f"{name} thresholds must satisfy minimum < maximum.")

        if not (
            isfinite(self.viewport_minimum_aspect_ratio)
            and isfinite(self.viewport_preferred_aspect_ratio)
            and isfinite(self.viewport_maximum_aspect_ratio)
            and 0.0
            < self.viewport_minimum_aspect_ratio
            < self.viewport_preferred_aspect_ratio
            < self.viewport_maximum_aspect_ratio
        ):
            raise ValueError(
                "Viewport aspect ratios must satisfy 0 < minimum < preferred < maximum."
            )
        aspect_ranges = (
            (
                self.level_button_minimum_aspect_ratio,
                self.level_button_maximum_aspect_ratio,
                "Level button",
            ),
            (
                self.ranking_card_minimum_aspect_ratio,
                self.ranking_card_maximum_aspect_ratio,
                "Ranking card",
            ),
        )
        for minimum, maximum, name in aspect_ranges:
            if not (
                isfinite(minimum) and isfinite(maximum) and 0.0 < minimum < maximum
            ):
                raise ValueError(
                    f"{name} aspect thresholds must be positive and ordered."
                )

        positive_ratios = {
            "viewport_minimum_height_ratio": self.viewport_minimum_height_ratio,
            "viewport_content_top_search_ratio": (
                self.viewport_content_top_search_ratio
            ),
            "viewport_boundary_probe_ratio": self.viewport_boundary_probe_ratio,
            "viewport_boundary_smoothing_ratio": (
                self.viewport_boundary_smoothing_ratio
            ),
            "level_morphology_kernel_ratio": self.level_morphology_kernel_ratio,
            "ranking_morphology_horizontal_kernel_ratio": (
                self.ranking_morphology_horizontal_kernel_ratio
            ),
            "ranking_morphology_vertical_kernel_ratio": (
                self.ranking_morphology_vertical_kernel_ratio
            ),
            "ranking_maximum_edge_alignment_ratio": (
                self.ranking_maximum_edge_alignment_ratio
            ),
            "ranking_maximum_gap_coefficient_of_variation": (
                self.ranking_maximum_gap_coefficient_of_variation
            ),
        }
        for name, positive_ratio in positive_ratios.items():
            if positive_ratio <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.viewport_content_top_maximum_ratio > (
            self.viewport_content_top_search_ratio
        ):
            raise ValueError(
                "Content-top maximum must not exceed its search-region ratio."
            )
        if self.level_morphology_iterations < 1:
            raise ValueError("Level morphology iterations must be positive.")
        if self.ranking_minimum_card_count < 2:
            raise ValueError("Ranking requires at least two cards.")
        if self.ranking_maximum_card_count < self.ranking_minimum_card_count:
            raise ValueError("Ranking maximum card count must not be below minimum.")


@dataclass(frozen=True, slots=True)
class _ViewportCandidate:
    """Retain complete private evidence for one vertical game viewport."""

    rect: CatsScreenRect
    score: float
    aspect_score: float
    left_boundary_score: float
    right_boundary_score: float
    internal_activity_score: float
    exterior_difference_score: float


@dataclass(frozen=True, slots=True)
class _ViewportContext:
    """Pair full-screenshot viewport geometry with its read-only image view."""

    rect: CatsScreenRect
    image: NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _ViewportSearch:
    """Return an accepted viewport and bounded best-effort diagnostics."""

    candidate: _ViewportCandidate | None
    best_score: float
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LevelButtonCandidate:
    """Retain primitive level-button evidence for deterministic selection."""

    rect: CatsScreenRect
    score: float
    orange_fill_ratio: float
    rectangularity: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


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


def _clamp_unit(value: float) -> float:
    """Clamp a normalized component into the inclusive unit interval."""

    return max(0.0, min(1.0, value))


def _triangular_score(
    value: float,
    minimum: float,
    preferred: float,
    maximum: float,
) -> float:
    """Score a value linearly around one preferred point inside hard bounds."""

    if value < minimum or value > maximum:
        return 0.0
    if value <= preferred:
        return _clamp_unit((value - minimum) / max(preferred - minimum, 1e-9))
    return _clamp_unit((maximum - value) / max(maximum - preferred, 1e-9))


def _threshold_score(value: float, minimum: float) -> float:
    """Normalize evidence from its hard minimum toward the ideal value one."""

    return _clamp_unit((value - minimum) / max(1.0 - minimum, 1e-9))


class OpenCvCatsScreenStateDetector(CatsScreenStateDetector):
    """Classify viewport overlays before full-screenshot board validation."""

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
        self._board_detector = board_detector or OpenCvBoardDetector(board_settings)
        self._grid_detector = grid_detector or OpenCvGridDetector(board_settings)
        legacy_geometry_was_injected = (
            board_detector is not None or grid_detector is not None
        )
        self._tile_grid_detector = (
            tile_grid_detector
            if tile_grid_detector is not None
            else (
                None if legacy_geometry_was_injected else OpenCvCatsTileGridDetector()
            )
        )

    def detect(self, screenshot: Screenshot) -> CatsScreenStateDetection:
        """Classify in viewport, LEVEL_COMPLETE, RANKING, BOARD, UNKNOWN order."""

        rejection_reasons: list[str] = []
        level_candidate: _LevelButtonCandidate | None = None
        ranking_cards: tuple[_RankingCard, ...] = ()
        ranking_score = 0.0
        try:
            viewport_search = self._find_game_viewport(screenshot)
            viewport_candidate = viewport_search.candidate
            if viewport_candidate is None:
                rejection_reasons.extend(viewport_search.rejection_reasons)
                rejection_reasons.append("no reliable Cats game viewport was found")
            else:
                context = self._viewport_context(screenshot, viewport_candidate.rect)
                level_candidate = self._find_level_button(context)
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

                ranking_cards, ranking_stack = self._find_ranking_stack(context)
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
            return self._detect_board_or_unknown(
                screenshot=screenshot,
                viewport_candidate=viewport_search.candidate,
                viewport_score=viewport_search.best_score,
                level_candidate=level_candidate,
                ranking_cards=ranking_cards,
                ranking_score=ranking_score,
                rejection_reasons=rejection_reasons,
            )
        except cv2.error as error:
            raise CatsScreenStateDetectionError(
                "OpenCV could not analyze Cats board or grid geometry."
            ) from error

    def _find_game_viewport(self, screenshot: Screenshot) -> _ViewportSearch:
        """Locate a tall 9:16-like content strip inside the BlueStacks window."""

        content_top = self._detect_content_top(screenshot.image)
        content_height = screenshot.height - content_top
        if (
            content_height / screenshot.height
            < self._settings.viewport_minimum_height_ratio
        ):
            return _ViewportSearch(
                candidate=None,
                best_score=0.0,
                rejection_reasons=(
                    "viewport candidate height was below the configured minimum",
                ),
            )

        content = screenshot.image[content_top:, :]
        lab = cast(NDArray[np.uint8], cv2.cvtColor(content, cv2.COLOR_BGR2LAB))
        boundary_profile = self._vertical_boundary_profile(lab)
        boundary_positions = self._boundary_positions(boundary_profile)
        candidates: list[_ViewportCandidate] = []
        saw_aspect_rejection = False
        for left, right in combinations(boundary_positions, 2):
            width = right - left
            aspect_ratio = width / content_height
            if not (
                self._settings.viewport_minimum_aspect_ratio
                <= aspect_ratio
                <= self._settings.viewport_maximum_aspect_ratio
            ):
                saw_aspect_rejection = True
                continue
            candidates.append(
                self._measure_viewport_candidate(
                    lab=lab,
                    content_top=content_top,
                    left=left,
                    right=right,
                    screenshot_width=screenshot.width,
                    screenshot_height=screenshot.height,
                    boundary_profile=boundary_profile,
                )
            )

        ordered = sorted(candidates, key=self._viewport_sort_key)
        full_content = next(
            (
                candidate
                for candidate in ordered
                if candidate.rect.x == 0
                and candidate.rect.x + candidate.rect.width == screenshot.width
            ),
            None,
        )
        if full_content is not None:
            better_two_sided = any(
                candidate.rect.x > 0
                and candidate.rect.x + candidate.rect.width < screenshot.width
                and candidate.score > full_content.score
                and candidate.left_boundary_score
                >= self._settings.viewport_minimum_boundary_score
                and candidate.right_boundary_score
                >= self._settings.viewport_minimum_boundary_score
                for candidate in ordered
            )
            if not better_two_sided:
                ordered.remove(full_content)
                ordered.insert(0, full_content)
        best_score = ordered[0].score if ordered else 0.0
        for candidate in ordered:
            left_is_edge = candidate.rect.x == 0
            right_is_edge = candidate.rect.x + candidate.rect.width == screenshot.width
            boundaries_reliable = (
                left_is_edge
                or candidate.left_boundary_score
                >= self._settings.viewport_minimum_boundary_score
            ) and (
                right_is_edge
                or candidate.right_boundary_score
                >= self._settings.viewport_minimum_boundary_score
            )
            if (
                candidate.score >= self._settings.viewport_minimum_score
                and candidate.internal_activity_score
                >= self._settings.viewport_minimum_internal_activity
                and boundaries_reliable
            ):
                return _ViewportSearch(candidate, best_score, ())

        reasons: list[str] = []
        if not candidates and saw_aspect_rejection:
            reasons.append("viewport candidate aspect ratio was outside range")
        elif ordered and (
            ordered[0].left_boundary_score
            < self._settings.viewport_minimum_boundary_score
            and ordered[0].right_boundary_score
            < self._settings.viewport_minimum_boundary_score
        ):
            reasons.append("viewport side boundaries were too weak")
        if ordered and (
            ordered[0].internal_activity_score
            < self._settings.viewport_minimum_internal_activity
        ):
            reasons.append("viewport internal activity was below threshold")
        if ordered and ordered[0].score < self._settings.viewport_minimum_score:
            reasons.append("viewport candidate score was below threshold")
        if not reasons:
            reasons.append("viewport side boundaries were too weak")
        return _ViewportSearch(None, best_score, tuple(reasons))

    def _detect_content_top(self, image: NDArray[np.uint8]) -> int:
        """Detect the title-bar/content boundary from horizontal LAB changes."""

        height = int(image.shape[0])
        search_bottom = max(
            2,
            min(
                height - 1,
                round(height * self._settings.viewport_content_top_search_ratio),
            ),
        )
        lab = cast(
            NDArray[np.uint8],
            cv2.cvtColor(image[: search_bottom + 1, :], cv2.COLOR_BGR2LAB),
        ).astype(np.float32)
        row_delta = np.linalg.norm(lab[1:] - lab[:-1], axis=2)
        profile = (
            0.40 * np.mean(row_delta, axis=1) + 0.60 * np.median(row_delta, axis=1)
        ) / 64.0
        minimum_y = max(
            1, round(height * self._settings.viewport_content_top_minimum_ratio)
        )
        maximum_y = min(
            search_bottom,
            max(
                minimum_y,
                round(height * self._settings.viewport_content_top_maximum_ratio),
            ),
        )
        search = profile[minimum_y - 1 : maximum_y]
        if search.size:
            relative_index = int(np.argmax(search))
            score = _clamp_unit(float(search[relative_index]))
            if score >= self._settings.viewport_minimum_content_top_boundary_score:
                return int(minimum_y + relative_index)
        fallback = round(height * self._settings.viewport_content_top_fallback_ratio)
        return int(max(minimum_y, min(maximum_y, fallback)))

    def _vertical_boundary_profile(
        self,
        lab: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        """Compare thin LAB strips on both sides of every possible x boundary."""

        height, width = lab.shape[:2]
        del height
        probe = max(1, round(width * self._settings.viewport_boundary_probe_ratio))
        profile = np.zeros(width + 1, dtype=np.float32)
        float_lab = lab.astype(np.float32)
        for x in range(probe, width - probe + 1):
            left = np.mean(float_lab[:, x - probe : x, :], axis=1)
            right = np.mean(float_lab[:, x : x + probe, :], axis=1)
            row_distance = np.linalg.norm(left - right, axis=1)
            profile[x] = _clamp_unit(
                (
                    0.30 * float(np.mean(row_distance))
                    + 0.30 * float(np.median(row_distance))
                    + 0.40 * float(np.percentile(row_distance, 25))
                )
                / 80.0
            )
        smoothing_size = max(
            1, round(width * self._settings.viewport_boundary_smoothing_ratio)
        )
        if smoothing_size % 2 == 0:
            smoothing_size += 1
        if smoothing_size > 1:
            kernel = np.full(smoothing_size, 1.0 / smoothing_size, dtype=np.float32)
            profile = np.asarray(
                np.convolve(profile, kernel, mode="same"), dtype=np.float32
            )
        return profile

    def _boundary_positions(
        self,
        profile: NDArray[np.float32],
    ) -> tuple[int, ...]:
        """Select well-spaced local maxima plus both screenshot edges."""

        width = len(profile) - 1
        minimum_separation = max(2, round(width * 0.015))
        raw_maxima = [
            x
            for x in range(1, width)
            if profile[x] >= profile[x - 1]
            and profile[x] >= profile[x + 1]
            and profile[x] >= self._settings.viewport_minimum_boundary_score * 0.50
        ]
        selected: list[int] = []
        for x in sorted(raw_maxima, key=lambda value: (-profile[value], value)):
            if all(abs(x - other) >= minimum_separation for other in selected):
                selected.append(x)
            if len(selected) >= 32:
                break
        return tuple(sorted((0, *selected, width)))

    def _measure_viewport_candidate(
        self,
        *,
        lab: NDArray[np.uint8],
        content_top: int,
        left: int,
        right: int,
        screenshot_width: int,
        screenshot_height: int,
        boundary_profile: NDArray[np.float32],
    ) -> _ViewportCandidate:
        """Combine aspect, boundaries, activity, exterior, height, and center."""

        height = lab.shape[0]
        width = right - left
        aspect_ratio = width / height
        aspect_score = _triangular_score(
            aspect_ratio,
            self._settings.viewport_minimum_aspect_ratio,
            self._settings.viewport_preferred_aspect_ratio,
            self._settings.viewport_maximum_aspect_ratio,
        )
        edge_evidence = 0.50 if left == 0 and right == screenshot_width else 0.10
        left_boundary = (
            edge_evidence if left == 0 else _clamp_unit(float(boundary_profile[left]))
        )
        right_boundary = (
            edge_evidence
            if right == screenshot_width
            else _clamp_unit(float(boundary_profile[right]))
        )
        internal_activity = self._internal_activity_score(lab[:, left:right])
        exterior_difference = self._exterior_difference_score(lab, left, right)
        height_ratio = height / screenshot_height
        height_score = _threshold_score(
            height_ratio, self._settings.viewport_minimum_height_ratio
        )
        center_ratio = (left + width / 2.0) / screenshot_width
        center_score = _clamp_unit(1.0 - abs(center_ratio - 0.5) / 0.5)
        score = _clamp_unit(
            0.34 * aspect_score
            + 0.12 * height_score
            + 0.12 * left_boundary
            + 0.12 * right_boundary
            + 0.15 * internal_activity
            + 0.10 * exterior_difference
            + 0.05 * center_score
        )
        return _ViewportCandidate(
            rect=CatsScreenRect(
                x=left,
                y=content_top,
                width=width,
                height=height,
            ),
            score=score,
            aspect_score=aspect_score,
            left_boundary_score=left_boundary,
            right_boundary_score=right_boundary,
            internal_activity_score=internal_activity,
            exterior_difference_score=exterior_difference,
        )

    @staticmethod
    def _internal_activity_score(image: NDArray[np.uint8]) -> float:
        """Measure texture and tonal diversity without using brightness as content."""

        grayscale = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_LAB2BGR))
        grayscale = cast(NDArray[np.uint8], cv2.cvtColor(grayscale, cv2.COLOR_BGR2GRAY))
        standard_deviation = float(np.std(grayscale)) / 52.0
        gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
        gradient = float(np.mean(cv2.magnitude(gradient_x, gradient_y))) / 42.0
        return _clamp_unit(0.60 * standard_deviation + 0.40 * gradient)

    def _exterior_difference_score(
        self,
        lab: NDArray[np.uint8],
        left: int,
        right: int,
    ) -> float:
        """Compare inner boundary strips with immediately adjacent exterior strips."""

        width = lab.shape[1]
        probe = max(
            1, round(width * self._settings.viewport_boundary_probe_ratio * 2.0)
        )
        scores: list[float] = []
        float_lab = lab.astype(np.float32)
        if left > 0:
            outer = float_lab[:, max(0, left - probe) : left, :]
            inner = float_lab[:, left : min(right, left + probe), :]
            scores.append(self._strip_difference(outer, inner))
        if right < width:
            inner = float_lab[:, max(left, right - probe) : right, :]
            outer = float_lab[:, right : min(width, right + probe), :]
            scores.append(self._strip_difference(inner, outer))
        return fmean(scores) if scores else 0.50

    @staticmethod
    def _strip_difference(
        first: NDArray[np.float32], second: NDArray[np.float32]
    ) -> float:
        """Return bounded row-wise LAB separation between two adjacent strips."""

        first_rows = np.mean(first, axis=1)
        second_rows = np.mean(second, axis=1)
        row_distance = np.linalg.norm(first_rows - second_rows, axis=1)
        return _clamp_unit(
            (
                0.30 * float(np.mean(row_distance))
                + 0.30 * float(np.median(row_distance))
                + 0.40 * float(np.percentile(row_distance, 25))
            )
            / 80.0
        )

    @staticmethod
    def _viewport_sort_key(
        candidate: _ViewportCandidate,
    ) -> tuple[float, float, int, float, float, int]:
        """Apply the documented deterministic viewport tie-break sequence."""

        boundary_mean = (
            candidate.left_boundary_score + candidate.right_boundary_score
        ) / 2.0
        return (
            -candidate.score,
            -candidate.aspect_score,
            -candidate.rect.height,
            -candidate.internal_activity_score,
            -boundary_mean,
            candidate.rect.x,
        )

    @staticmethod
    def _viewport_context(
        screenshot: Screenshot,
        rect: CatsScreenRect,
    ) -> _ViewportContext:
        """Create a private crop view without constructing or mutating Screenshot."""

        return _ViewportContext(
            rect=rect,
            image=screenshot.image[
                rect.y : rect.y + rect.height,
                rect.x : rect.x + rect.width,
            ],
        )

    def _find_level_button(
        self,
        viewport: _ViewportContext,
    ) -> _LevelButtonCandidate | None:
        """Find the best lower orange component relative to the game viewport."""

        hsv = cast(NDArray[np.uint8], cv2.cvtColor(viewport.image, cv2.COLOR_BGR2HSV))
        mask = cast(
            NDArray[np.uint8],
            cv2.inRange(
                hsv,
                (
                    self._settings.level_orange_hue_minimum,
                    self._settings.level_orange_saturation_minimum,
                    self._settings.level_orange_value_minimum,
                ),
                (self._settings.level_orange_hue_maximum, 255, 255),
            ),
        )
        region_start = round(
            viewport.rect.height * self._settings.level_region_start_y_ratio
        )
        mask[:region_start, :] = 0
        kernel = self._relative_kernel(
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
        orange_mask: NDArray[np.uint8],
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
        orange_fill = (
            cv2.countNonZero(orange_mask[y : y + height, x : x + width])
            / rectangle_area
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
            orange_fill, self._settings.level_button_minimum_orange_fill_ratio
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
        if orange_fill < self._settings.level_button_minimum_orange_fill_ratio:
            reasons.append("level button orange fill was below threshold")
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
            orange_fill_ratio=orange_fill,
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
            -candidate.orange_fill_ratio,
            -candidate.rect.width,
            -candidate.rect.center_y,
            candidate.rect.x,
        )

    def _find_ranking_stack(
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
        open_kernel = self._relative_kernel(width, height, 0.003)
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

        union = self._union_rect(tuple(card.rect for card in ordered))
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
        """Create a ranking result with a global action below the card union."""

        rectangles = tuple(card.rect for card in stack.cards)
        union = self._union_rect(rectangles)
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
            action_point=CatsScreenPoint(x=action_x, y=action_y),
            diagnostics=diagnostics,
        )

    def _detect_board_or_unknown(
        self,
        *,
        screenshot: Screenshot,
        viewport_candidate: _ViewportCandidate | None,
        viewport_score: float,
        level_candidate: _LevelButtonCandidate | None,
        ranking_cards: tuple[_RankingCard, ...],
        ranking_score: float,
        rejection_reasons: list[str],
    ) -> CatsScreenStateDetection:
        """Run Cats tile-grid primary, then generic fallback, on the full frame."""

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
                diagnostics = self._diagnostics(
                    viewport_candidate=viewport_candidate,
                    viewport_score=viewport_score,
                    level_candidate=level_candidate,
                    ranking_cards=ranking_cards,
                    ranking_score=ranking_score,
                    board_candidate=board_rect,
                    board_confidence=board_confidence,
                    grid_confidence=grid_confidence,
                    rows=rows,
                    columns=columns,
                    rejection_reasons=tuple(rejection_reasons),
                )
                return CatsScreenStateDetection(
                    state=CatsScreenState.BOARD,
                    confidence=min(board_confidence, grid_confidence),
                    action_point=None,
                    diagnostics=diagnostics,
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
                diagnostics = self._diagnostics(
                    viewport_candidate=viewport_candidate,
                    viewport_score=viewport_score,
                    level_candidate=level_candidate,
                    ranking_cards=ranking_cards,
                    ranking_score=ranking_score,
                    board_candidate=board_rect,
                    board_confidence=board_confidence,
                    grid_confidence=grid_confidence,
                    rows=rows,
                    columns=columns,
                    rejection_reasons=tuple(rejection_reasons),
                )
                return CatsScreenStateDetection(
                    state=CatsScreenState.BOARD,
                    confidence=min(board_confidence, grid_confidence),
                    action_point=None,
                    diagnostics=diagnostics,
                )

        diagnostics = self._diagnostics(
            viewport_candidate=viewport_candidate,
            viewport_score=viewport_score,
            level_candidate=level_candidate,
            ranking_cards=ranking_cards,
            ranking_score=ranking_score,
            board_candidate=board_rect,
            board_confidence=board_confidence,
            grid_confidence=grid_confidence,
            rows=rows,
            columns=columns,
            rejection_reasons=tuple(rejection_reasons),
        )
        return CatsScreenStateDetection(
            state=CatsScreenState.UNKNOWN,
            confidence=0.0,
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

    @staticmethod
    def _relative_kernel(width: int, height: int, ratio: float) -> NDArray[np.uint8]:
        """Build one odd scale-relative morphology kernel for any resolution."""

        size = max(3, round(min(width, height) * ratio))
        if size % 2 == 0:
            size += 1
        return cast(
            NDArray[np.uint8],
            cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)),
        )

    @staticmethod
    def _union_rect(rectangles: tuple[CatsScreenRect, ...]) -> CatsScreenRect:
        """Return the smallest half-open rectangle containing every input rect."""

        left = min(rect.x for rect in rectangles)
        top = min(rect.y for rect in rectangles)
        right = max(rect.x + rect.width for rect in rectangles)
        bottom = max(rect.y + rect.height for rect in rectangles)
        return CatsScreenRect(x=left, y=top, width=right - left, height=bottom - top)
