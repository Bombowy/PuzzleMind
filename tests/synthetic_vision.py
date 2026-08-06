"""Shared deterministic, non-private synthetic screenshots for vision tests."""

from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.vision.screenshot import Screenshot


def screenshot_from_image(image: NDArray[np.uint8]) -> Screenshot:
    """Wrap synthetic BGR pixels in the production immutable screenshot model."""

    height, width = image.shape[:2]
    return Screenshot(
        image=image,
        width=width,
        height=height,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


def custom_grid_screenshot(
    *,
    rows: int,
    columns: int,
    board_width: int = 400,
    board_height: int = 320,
    horizontal_positions: tuple[float, ...] | None = None,
    vertical_positions: tuple[float, ...] | None = None,
    horizontal_coverage: float = 1.0,
    vertical_coverage: float = 1.0,
) -> Screenshot:
    """Draw one configurable grid for shared board and public-grid regression tests."""

    image = np.full((600, 800, 3), 32, dtype=np.uint8)
    x = (800 - board_width) // 2
    y = (600 - board_height) // 2
    right = x + board_width
    bottom = y + board_height
    cv2.rectangle(image, (x, y), (right, bottom), (225, 225, 225), -1)
    cv2.rectangle(image, (x, y), (right, bottom), (245, 245, 245), 6)

    row_separators = horizontal_positions or tuple(
        index / rows for index in range(1, rows)
    )
    column_separators = vertical_positions or tuple(
        index / columns for index in range(1, columns)
    )
    horizontal_margin = round(board_width * (1.0 - horizontal_coverage) / 2.0)
    vertical_margin = round(board_height * (1.0 - vertical_coverage) / 2.0)
    for position in row_separators:
        separator_y = y + round(board_height * position)
        cv2.line(
            image,
            (x + horizontal_margin, separator_y),
            (right - horizontal_margin, separator_y),
            (70, 70, 70),
            3,
        )
    for position in column_separators:
        separator_x = x + round(board_width * position)
        cv2.line(
            image,
            (separator_x, y + vertical_margin),
            (separator_x, bottom - vertical_margin),
            (70, 70, 70),
            3,
        )
    return screenshot_from_image(image)


def advertisement_like_screenshot() -> Screenshot:
    """Create one geometry-plausible text/image card without a regular grid."""

    image = np.full((600, 800, 3), 32, dtype=np.uint8)
    x, y, width, height = 200, 140, 400, 320
    cv2.rectangle(image, (x, y), (x + width, y + height), (230, 230, 230), -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (250, 250, 250), 8)
    cv2.circle(image, (295, 250), 62, (110, 110, 110), 8)
    cv2.circle(image, (295, 250), 28, (170, 170, 170), -1)
    cv2.rectangle(image, (390, 205), (550, 235), (85, 85, 85), 3)
    cv2.rectangle(image, (390, 255), (520, 275), (100, 100, 100), 3)
    cv2.line(image, (245, 365), (560, 365), (90, 90, 90), 4)
    cv2.line(image, (260, 395), (470, 395), (120, 120, 120), 4)
    cv2.line(image, (515, 320), (565, 410), (80, 80, 80), 7)
    cv2.putText(
        image,
        "SPECIAL",
        (375, 330),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (65, 65, 65),
        2,
        cv2.LINE_AA,
    )
    return screenshot_from_image(image)
