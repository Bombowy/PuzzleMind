"""Deterministic synthetic Cats screens without private screenshot fixtures."""

from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.plugins.cats import CatsScreenState
from logicforge.vision.screenshot import Screenshot


def _screenshot(image: NDArray[np.uint8]) -> Screenshot:
    """Wrap generated BGR pixels in the production immutable transport model."""

    height, width = image.shape[:2]
    return Screenshot(
        image=image,
        width=width,
        height=height,
        timestamp=datetime(2026, 8, 6, tzinfo=UTC),
    )


def _draw_regular_board(
    image: NDArray[np.uint8],
    *,
    rows: int = 6,
    columns: int = 6,
) -> tuple[int, int, int, int]:
    """Draw one scale-relative regular board and return its exact geometry."""

    height, width = image.shape[:2]
    board_size = round(min(width * 0.88, height * 0.48))
    x = (width - board_size) // 2
    y = round(height * 0.43 - board_size / 2)
    right = x + board_size
    bottom = y + board_size
    cv2.rectangle(image, (x, y), (right, bottom), (225, 225, 225), cv2.FILLED)
    cv2.rectangle(image, (x, y), (right, bottom), (248, 248, 248), 5)
    for row in range(1, rows):
        separator_y = y + round(board_size * row / rows)
        cv2.line(image, (x, separator_y), (right, separator_y), (65, 65, 65), 3)
    for column in range(1, columns):
        separator_x = x + round(board_size * column / columns)
        cv2.line(image, (separator_x, y), (separator_x, bottom), (65, 65, 65), 3)
    return x, y, board_size, board_size


def _base_board_image(width: int, height: int) -> NDArray[np.uint8]:
    """Create a varied game frame with one visible regular grid."""

    image = np.full((height, width, 3), (44, 38, 31), dtype=np.uint8)
    for y in range(height):
        shade = round(24 * y / max(1, height - 1))
        image[y, :, :] = (44 + shade // 3, 38 + shade // 2, 31 + shade)
    _draw_regular_board(image)
    cv2.circle(
        image,
        (round(width * 0.18), round(height * 0.12)),
        max(5, round(width * 0.06)),
        (80, 125, 170),
        cv2.FILLED,
    )
    return image


def _dim(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Simulate a modal backdrop while retaining visible board structure."""

    return np.asarray(np.round(image.astype(np.float32) * 0.30), dtype=np.uint8)


def _draw_ranking_cards(
    image: NDArray[np.uint8],
    *,
    card_count: int,
    aligned: bool,
    interrupted: bool = True,
    card_width_ratio: float = 0.78,
) -> tuple[tuple[int, int, int, int], ...]:
    """Draw neutral, cream, and peach cards with mask-breaking details."""

    height, width = image.shape[:2]
    card_width = round(width * card_width_ratio)
    card_height = round(height * 0.078)
    gap = round(height * 0.035)
    start_y = round(height * 0.28)
    base_x = (width - card_width) // 2
    offsets = (0, round(width * 0.14), -round(width * 0.13), round(width * 0.10))
    colors = ((242, 242, 242), (205, 226, 245), (180, 208, 245))
    rectangles: list[tuple[int, int, int, int]] = []
    for index in range(card_count):
        x = base_x if aligned else base_x + offsets[index % len(offsets)]
        x = max(2, min(width - card_width - 2, x))
        y = start_y + index * (card_height + gap)
        cv2.rectangle(
            image,
            (x, y),
            (x + card_width, y + card_height),
            colors[index % len(colors)],
            cv2.FILLED,
        )
        if interrupted:
            avatar_size = round(card_height * 0.76)
            avatar_x = x + round(card_width * 0.08)
            avatar_y = y + (card_height - avatar_size) // 2
            cv2.circle(
                image,
                (avatar_x + avatar_size // 2, avatar_y + avatar_size // 2),
                avatar_size // 2,
                (48, 55, 68),
                cv2.FILLED,
            )
            cv2.rectangle(
                image,
                (x + round(card_width * 0.28), y + round(card_height * 0.25)),
                (x + round(card_width * 0.66), y + round(card_height * 0.42)),
                (55, 48, 45),
                cv2.FILLED,
            )
            cv2.rectangle(
                image,
                (x + round(card_width * 0.30), y + round(card_height * 0.59)),
                (x + round(card_width * 0.51), y + round(card_height * 0.72)),
                (72, 65, 60),
                cv2.FILLED,
            )
        rectangles.append((x, y, card_width, card_height))
    return tuple(rectangles)


def _draw_level_button(image: NDArray[np.uint8]) -> tuple[int, int, int, int]:
    """Draw a rounded orange button occupying about 70% of viewport width."""

    height, width = image.shape[:2]
    button_width = round(width * 0.70)
    button_height = round(height * 0.085)
    x = (width - button_width) // 2
    y = round(height * 0.86)
    radius = max(4, button_height // 2)
    orange = (0, 145, 255)
    cv2.rectangle(
        image,
        (x + radius, y),
        (x + button_width - radius, y + button_height),
        orange,
        cv2.FILLED,
    )
    cv2.circle(image, (x + radius, y + radius), radius, orange, cv2.FILLED)
    cv2.circle(
        image,
        (x + button_width - radius, y + radius),
        radius,
        orange,
        cv2.FILLED,
    )
    cv2.circle(
        image,
        (round(width * 0.18), round(height * 0.68)),
        max(2, round(min(width, height) * 0.008)),
        orange,
        cv2.FILLED,
    )
    return x, y, button_width, button_height


def synthetic_game_viewport(
    *,
    width: int,
    height: int,
    state: CatsScreenState,
    card_count: int = 3,
    aligned: bool = True,
    include_ranking_cards: bool = False,
    interrupted_cards: bool = True,
    card_width_ratio: float = 0.78,
) -> NDArray[np.uint8]:
    """Create one standalone game viewport for embedding in a full window."""

    if state is CatsScreenState.UNKNOWN:
        image = np.full((height, width, 3), (48, 39, 34), dtype=np.uint8)
        cv2.circle(
            image,
            (round(width * 0.28), round(height * 0.38)),
            round(width * 0.12),
            (95, 80, 70),
            5,
        )
        return image

    base = _base_board_image(width, height)
    if state is CatsScreenState.BOARD:
        return base
    image = _dim(base)
    if state is CatsScreenState.RANKING or include_ranking_cards:
        _draw_ranking_cards(
            image,
            card_count=card_count,
            aligned=aligned,
            interrupted=interrupted_cards,
            card_width_ratio=card_width_ratio,
        )
    if state is CatsScreenState.LEVEL_COMPLETE:
        _draw_level_button(image)
    return image


def synthetic_bluestacks_window(
    *,
    screenshot_width: int,
    screenshot_height: int,
    viewport_x: int,
    viewport_y: int,
    viewport_width: int,
    viewport_height: int,
    state: CatsScreenState,
    left_ad: bool = True,
    right_toolbar: bool = True,
    card_count: int = 3,
    aligned: bool = True,
    include_ranking_cards: bool = False,
    interrupted_cards: bool = True,
    card_width_ratio: float = 0.78,
) -> Screenshot:
    """Embed a Cats viewport among title bar, ad, margins, separators, and toolbar."""

    if (
        viewport_x < 0
        or viewport_y < 0
        or viewport_width <= 0
        or viewport_height <= 0
        or viewport_x + viewport_width > screenshot_width
        or viewport_y + viewport_height > screenshot_height
    ):
        raise ValueError("Viewport must be a positive rectangle inside the screenshot.")
    image = np.full(
        (screenshot_height, screenshot_width, 3), (6, 8, 25), dtype=np.uint8
    )
    image[:viewport_y, :] = (54, 45, 38)
    cv2.line(
        image,
        (0, max(0, viewport_y - 1)),
        (screenshot_width - 1, max(0, viewport_y - 1)),
        (105, 96, 88),
        max(1, round(screenshot_height * 0.002)),
    )
    content_bottom = viewport_y + viewport_height
    if left_ad and viewport_x > 0:
        margin = max(2, round(screenshot_width * 0.012))
        ad_left = margin
        ad_right = max(
            ad_left + 1,
            min(
                viewport_x - margin * 2,
                ad_left + round(viewport_width * 0.48),
            ),
        )
        ad_top = viewport_y + round(viewport_height * 0.26)
        ad_bottom = viewport_y + round(viewport_height * 0.79)
        cv2.rectangle(
            image, (ad_left, ad_top), (ad_right, ad_bottom), (60, 68, 135), cv2.FILLED
        )
        cv2.rectangle(
            image,
            (ad_left + margin, ad_top + margin),
            (ad_right - margin, ad_top + round((ad_bottom - ad_top) * 0.46)),
            (195, 205, 225),
            cv2.FILLED,
        )
        cv2.rectangle(
            image,
            (ad_left + margin, ad_top + round((ad_bottom - ad_top) * 0.60)),
            (ad_right - margin, ad_top + round((ad_bottom - ad_top) * 0.72)),
            (230, 234, 240),
            cv2.FILLED,
        )
        orange_y = ad_top + round((ad_bottom - ad_top) * 0.82)
        cv2.rectangle(
            image,
            (ad_left + margin, orange_y),
            (ad_right - margin, orange_y + round(viewport_height * 0.07)),
            (0, 145, 255),
            cv2.FILLED,
        )
    separator = max(1, round(screenshot_width * 0.004))
    if viewport_x > 0:
        image[
            viewport_y:content_bottom, max(0, viewport_x - separator) : viewport_x
        ] = 0

    viewport_image = synthetic_game_viewport(
        width=viewport_width,
        height=viewport_height,
        state=state,
        card_count=card_count,
        aligned=aligned,
        include_ranking_cards=include_ranking_cards,
        interrupted_cards=interrupted_cards,
        card_width_ratio=card_width_ratio,
    )
    image[
        viewport_y:content_bottom,
        viewport_x : viewport_x + viewport_width,
    ] = viewport_image

    viewport_right = viewport_x + viewport_width
    if viewport_right < screenshot_width:
        image[
            viewport_y:content_bottom,
            viewport_right : min(screenshot_width, viewport_right + separator),
        ] = 0
    if right_toolbar and viewport_right < screenshot_width:
        toolbar_left = max(
            viewport_right + separator,
            screenshot_width - round(screenshot_width * 0.04),
        )
        image[viewport_y:content_bottom, toolbar_left:] = (62, 53, 72)
        icon_radius = max(2, round((screenshot_width - toolbar_left) * 0.18))
        for index in range(12):
            center_y = viewport_y + round(viewport_height * (0.05 + index * 0.075))
            if center_y < content_bottom:
                cv2.circle(
                    image,
                    ((toolbar_left + screenshot_width) // 2, center_y),
                    icon_radius,
                    (190, 195, 210),
                    1,
                )
    return _screenshot(image)


def synthetic_board_screen(*, width: int = 560, height: int = 1000) -> Screenshot:
    """Create a narrow Cats board where the game nearly fills the screenshot."""

    return _screenshot(_base_board_image(width, height))


def synthetic_ranking_screen(
    *,
    width: int = 560,
    height: int = 1000,
    card_count: int = 3,
    aligned: bool = True,
    include_bubble: bool = False,
) -> Screenshot:
    """Create a narrow-window ranking screen for full-content fallback tests."""

    image = synthetic_game_viewport(
        width=width,
        height=height,
        state=CatsScreenState.RANKING,
        card_count=card_count,
        aligned=aligned,
    )
    if include_bubble:
        cv2.rectangle(
            image,
            (round(width * 0.31), round(height * 0.19)),
            (round(width * 0.69), round(height * 0.245)),
            (232, 232, 235),
            cv2.FILLED,
        )
    return _screenshot(image)


def synthetic_level_complete_screen(
    *,
    width: int = 560,
    height: int = 1000,
    include_ranking_cards: bool = False,
) -> Screenshot:
    """Create a narrow-window level-complete screen for viewport fallback."""

    return _screenshot(
        synthetic_game_viewport(
            width=width,
            height=height,
            state=CatsScreenState.LEVEL_COMPLETE,
            include_ranking_cards=include_ranking_cards,
        )
    )


def synthetic_unknown_screen(*, width: int = 560, height: int = 1000) -> Screenshot:
    """Create a non-empty narrow UI frame without a known state."""

    return _screenshot(
        synthetic_game_viewport(
            width=width,
            height=height,
            state=CatsScreenState.UNKNOWN,
        )
    )
