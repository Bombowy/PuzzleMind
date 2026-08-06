"""Shared deterministic, non-private synthetic screenshots for vision tests."""

from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.vision.board_detector import BoardDetection
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


def weak_separator_grid_case(
    *,
    rows: int,
    columns: int,
    weakened_vertical_line_indices: tuple[int, ...] = (),
    weakened_horizontal_line_indices: tuple[int, ...] = (),
    weak_signal_present: bool = True,
    board_width: int = 540,
    board_height: int = 540,
) -> tuple[Screenshot, BoardDetection]:
    """Draw regular strong lines plus selected real, fragmented weak separators."""

    if rows < 3 or columns < 3:
        raise ValueError(
            "Synthetic weak grid requires at least three rows and columns."
        )
    if any(not 0 < index < columns for index in weakened_vertical_line_indices):
        raise ValueError("Weakened vertical line indices must be internal.")
    if any(not 0 < index < rows for index in weakened_horizontal_line_indices):
        raise ValueError("Weakened horizontal line indices must be internal.")

    screenshot_width = max(800, board_width + 360)
    screenshot_height = max(700, board_height + 220)
    board_x = (screenshot_width - board_width) // 2
    board_y = (screenshot_height - board_height) // 2
    right = board_x + board_width
    bottom = board_y + board_height
    image = np.full(
        (screenshot_height, screenshot_width, 3),
        32,
        dtype=np.uint8,
    )
    palette = (
        (220, 185, 245),
        (205, 225, 170),
        (245, 205, 165),
        (185, 215, 245),
    )
    cv2.rectangle(image, (board_x, board_y), (right, bottom), palette[0], -1)
    if weakened_vertical_line_indices and not weakened_horizontal_line_indices:
        for row in range(rows):
            top = board_y + round(board_height * row / rows)
            row_bottom = board_y + round(board_height * (row + 1) / rows)
            cv2.rectangle(
                image,
                (board_x, top),
                (right, row_bottom),
                palette[row % len(palette)],
                -1,
            )
    elif weakened_horizontal_line_indices and not weakened_vertical_line_indices:
        for column in range(columns):
            left = board_x + round(board_width * column / columns)
            column_right = board_x + round(board_width * (column + 1) / columns)
            cv2.rectangle(
                image,
                (left, board_y),
                (column_right, bottom),
                palette[column % len(palette)],
                -1,
            )
    cv2.rectangle(image, (board_x, board_y), (right, bottom), (245, 245, 245), 6)

    weakened_vertical = set(weakened_vertical_line_indices)
    weakened_horizontal = set(weakened_horizontal_line_indices)
    for row in range(1, rows):
        if row in weakened_horizontal:
            continue
        y = board_y + round(board_height * row / rows)
        cv2.line(image, (board_x, y), (right, y), (70, 70, 70), 3)
    for column in range(1, columns):
        if column in weakened_vertical:
            continue
        x = board_x + round(board_width * column / columns)
        cv2.line(image, (x, board_y), (x, bottom), (70, 70, 70), 3)

    if weak_signal_present:
        vertical_fragment_length = max(4, round(board_height * 0.04))
        horizontal_fragment_length = max(4, round(board_width * 0.04))
        selected_rows = tuple(range(1, rows, max(1, rows // 4)))[:4]
        selected_columns = tuple(range(1, columns, max(1, columns // 4)))[:4]
        for column in weakened_vertical:
            x = board_x + round(board_width * column / columns)
            for row in selected_rows:
                center_y = board_y + round(board_height * (row + 0.5) / rows)
                half_length = vertical_fragment_length // 2
                cv2.line(
                    image,
                    (x, center_y - half_length),
                    (x, center_y + half_length),
                    (70, 70, 70),
                    2,
                )
        for row in weakened_horizontal:
            y = board_y + round(board_height * row / rows)
            for column in selected_columns:
                center_x = board_x + round(board_width * (column + 0.5) / columns)
                half_length = horizontal_fragment_length // 2
                cv2.line(
                    image,
                    (center_x - half_length, y),
                    (center_x + half_length, y),
                    (70, 70, 70),
                    2,
                )
    return screenshot_from_image(image), BoardDetection(
        x=board_x,
        y=board_y,
        width=board_width,
        height=board_height,
        confidence=0.90,
    )


def live_like_9x9_weak_grid_case(
    weakened_vertical_line_index: int,
    *,
    weakened_horizontal_line_index: int | None = None,
    weak_signal_present: bool = True,
) -> tuple[Screenshot, BoardDetection]:
    """Create a square 9x9 Cats-like grid with selectable weak separators."""

    horizontal_indices = (
        (weakened_horizontal_line_index,)
        if weakened_horizontal_line_index is not None
        else ()
    )
    return weak_separator_grid_case(
        rows=9,
        columns=9,
        weakened_vertical_line_indices=(weakened_vertical_line_index,),
        weakened_horizontal_line_indices=horizontal_indices,
        weak_signal_present=weak_signal_present,
    )
