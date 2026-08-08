"""Viewport discovery and scoring for Cats screen-state classification."""

from dataclasses import dataclass
from itertools import combinations
from statistics import fmean
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

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
from logicforge.plugins.cats.screen_state import CatsScreenRect
from logicforge.vision.screenshot import Screenshot


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


class _ViewportAnalyzer:
    """Locate and score one game viewport without owning orchestration policy."""

    def __init__(self, settings: CatsScreenStateDetectionSettings) -> None:
        self._settings = settings

    def find_game_viewport(self, screenshot: Screenshot) -> _ViewportSearch:
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
    def viewport_context(
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
