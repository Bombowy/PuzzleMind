"""Synthetic Cats-like tile lattices without a board outer contour."""

from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.vision.screenshot import Screenshot

CATS_TILE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (72, 84, 224),
    (82, 188, 92),
    (54, 202, 236),
    (218, 142, 82),
    (184, 92, 183),
    (52, 137, 237),
    (202, 202, 65),
    (158, 112, 235),
    (150, 181, 61),
    (221, 112, 125),
)


@dataclass(frozen=True, slots=True)
class SyntheticCatsTileGrid:
    """Pair generated pixels with exact lattice geometry and logical colors."""

    screenshot: Screenshot
    board_x: int
    board_y: int
    board_width: int
    board_height: int
    horizontal_lines: tuple[int, ...]
    vertical_lines: tuple[int, ...]
    tile_centers: tuple[tuple[tuple[int, int], ...], ...]
    color_indices: tuple[tuple[int, ...], ...]


def _rounded_rectangle(
    image: NDArray[np.uint8],
    left: int,
    top: int,
    right: int,
    bottom: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled rounded tile without creating a shared grid outline."""

    radius = max(2, min(right - left, bottom - top) // 8)
    cv2.rectangle(
        image,
        (left + radius, top),
        (right - radius, bottom),
        color,
        cv2.FILLED,
    )
    cv2.rectangle(
        image,
        (left, top + radius),
        (right, bottom - radius),
        color,
        cv2.FILLED,
    )
    for center in (
        (left + radius, top + radius),
        (right - radius, top + radius),
        (left + radius, bottom - radius),
        (right - radius, bottom - radius),
    ):
        cv2.circle(image, center, radius, color, cv2.FILLED, cv2.LINE_AA)


def _draw_cat_like_sprite(
    image: NDArray[np.uint8],
    center_x: int,
    center_y: int,
    tile_width: int,
    tile_height: int,
) -> None:
    """Draw a synthetic multicolor center sprite while preserving tile corners."""

    radius = max(3, round(min(tile_width, tile_height) * 0.30))
    ear_half_width = max(2, radius // 2)
    ear_height = max(3, round(radius * 0.75))
    brown = (55, 91, 137)
    dark = (28, 31, 36)
    white = (242, 244, 246)
    pink = (145, 120, 238)
    cv2.fillConvexPoly(
        image,
        np.asarray(
            (
                (center_x - radius + 1, center_y - radius + 2),
                (center_x - ear_half_width, center_y - radius - ear_height),
                (center_x - 1, center_y - radius + 3),
            ),
            dtype=np.int32,
        ),
        brown,
    )
    cv2.fillConvexPoly(
        image,
        np.asarray(
            (
                (center_x + 1, center_y - radius + 3),
                (center_x + ear_half_width, center_y - radius - ear_height),
                (center_x + radius - 1, center_y - radius + 2),
            ),
            dtype=np.int32,
        ),
        brown,
    )
    cv2.circle(image, (center_x, center_y), radius, dark, cv2.FILLED, cv2.LINE_AA)
    eye_y = center_y - radius // 4
    eye_offset = max(2, radius // 3)
    for eye_x in (center_x - eye_offset, center_x + eye_offset):
        cv2.circle(image, (eye_x, eye_y), max(1, radius // 5), white, cv2.FILLED)
        cv2.circle(image, (eye_x, eye_y), 1, dark, cv2.FILLED)
    cv2.ellipse(
        image,
        (center_x, center_y + radius // 3),
        (max(2, radius // 2), max(1, radius // 3)),
        0,
        0,
        360,
        white,
        cv2.FILLED,
        cv2.LINE_AA,
    )
    cv2.circle(
        image,
        (center_x, center_y + radius // 5),
        max(1, radius // 6),
        pink,
        cv2.FILLED,
    )


def synthetic_cats_tile_grid(
    *,
    rows: int,
    columns: int,
    screenshot_width: int = 916,
    screenshot_height: int = 1032,
    board_x: int | None = None,
    board_y: int = 266,
    pitch_x: int = 53,
    pitch_y: int = 53,
    tile_width: int = 47,
    tile_height: int = 47,
    palette: tuple[tuple[int, int, int], ...] = CATS_TILE_PALETTE,
    pastel_outer_column: bool = False,
    pastel_outer_row: bool = False,
    include_advertisement: bool = True,
    include_toolbar: bool = True,
    include_ui: bool = True,
    omitted_slots: frozenset[tuple[int, int]] = frozenset(),
    center_offsets: tuple[tuple[int, int, int, int], ...] = (),
    size_overrides: tuple[tuple[int, int, int, int], ...] = (),
    duplicate_slot: tuple[int, int] | None = None,
    cat_sprite_slots: frozenset[tuple[int, int]] = frozenset(),
) -> SyntheticCatsTileGrid:
    """Create separated rounded colored tiles on a cream game panel.

    Offset and size override records use ``(row, column, dx, dy_or_size)``. For
    ``size_overrides`` the last two values are replacement width and height.
    """

    if rows <= 0 or columns <= 0:
        raise ValueError("Synthetic lattice dimensions must be positive.")
    board_width = columns * pitch_x
    board_height = rows * pitch_y
    if board_x is None:
        board_x = screenshot_width - board_width - max(48, screenshot_width // 20)
    if (
        board_x < 0
        or board_y < 0
        or board_x + board_width > screenshot_width
        or board_y + board_height > screenshot_height
    ):
        raise ValueError("Synthetic Cats board must fit inside the screenshot.")

    image = np.full(
        (screenshot_height, screenshot_width, 3),
        (8, 9, 13),
        dtype=np.uint8,
    )
    title_height = max(16, round(screenshot_height * 0.032))
    image[:title_height, :] = (49, 44, 42)
    panel_left = max(0, board_x - round(pitch_x * 0.36))
    panel_right = min(screenshot_width, board_x + board_width + round(pitch_x * 0.36))
    image[title_height:, panel_left:panel_right] = (236, 238, 241)

    if include_advertisement and panel_left >= 150:
        ad_right = max(8, panel_left - 9)
        image[title_height:, 8:ad_right] = (214, 220, 226)
        _rounded_rectangle(
            image,
            22,
            title_height + 52,
            min(ad_right - 8, 146),
            title_height + 109,
            (74, 128, 231),
        )
        _rounded_rectangle(
            image,
            48,
            title_height + 151,
            min(ad_right - 16, 207),
            title_height + 207,
            (211, 116, 69),
        )
        cv2.circle(
            image,
            (min(ad_right - 30, 112), title_height + 290),
            27,
            (63, 196, 224),
            cv2.FILLED,
        )

    if include_toolbar and panel_right + 22 < screenshot_width:
        toolbar_left = screenshot_width - max(28, screenshot_width // 28)
        image[title_height:, toolbar_left:] = (42, 43, 45)
        for index in range(7):
            cy = title_height + 48 + index * max(45, screenshot_height // 14)
            cv2.circle(
                image,
                ((toolbar_left + screenshot_width) // 2, cy),
                5,
                (176, 178, 180),
                1,
            )

    if include_ui:
        cv2.circle(
            image,
            (panel_left + 42, max(title_height + 24, board_y - 63)),
            18,
            (61, 147, 233),
            cv2.FILLED,
        )
        _rounded_rectangle(
            image,
            board_x + board_width // 4,
            board_y + board_height + 38,
            board_x + 3 * board_width // 4,
            board_y + board_height + 82,
            (62, 140, 239),
        )

    offsets = {(row, column): (dx, dy) for row, column, dx, dy in center_offsets}
    sizes = {
        (row, column): (width, height) for row, column, width, height in size_overrides
    }
    centers: list[tuple[tuple[int, int], ...]] = []
    color_indices: list[tuple[int, ...]] = []
    for row in range(rows):
        center_row: list[tuple[int, int]] = []
        color_row: list[int] = []
        for column in range(columns):
            dx, dy = offsets.get((row, column), (0, 0))
            center_x = board_x + column * pitch_x + pitch_x // 2 + dx
            center_y = board_y + row * pitch_y + pitch_y // 2 + dy
            center_row.append((center_x, center_y))
            color_index = (row + column) % min(rows, columns, len(palette))
            color_row.append(color_index)
            if (row, column) in omitted_slots:
                if (row, column) in cat_sprite_slots:
                    color = palette[color_index]
                    patch_size = max(3, round(min(tile_width, tile_height) * 0.22))
                    patch_offset_x = max(
                        patch_size,
                        round(tile_width * 0.31),
                    )
                    patch_offset_y = max(
                        patch_size,
                        round(tile_height * 0.31),
                    )
                    for patch_center_x, patch_center_y in (
                        (center_x - patch_offset_x, center_y - patch_offset_y),
                        (center_x + patch_offset_x, center_y - patch_offset_y),
                        (center_x - patch_offset_x, center_y + patch_offset_y),
                        (center_x + patch_offset_x, center_y + patch_offset_y),
                    ):
                        half = patch_size // 2
                        cv2.rectangle(
                            image,
                            (patch_center_x - half, patch_center_y - half),
                            (patch_center_x + half, patch_center_y + half),
                            color,
                            cv2.FILLED,
                        )
                    _draw_cat_like_sprite(
                        image,
                        center_x,
                        center_y,
                        tile_width,
                        tile_height,
                    )
                continue
            current_width, current_height = sizes.get(
                (row, column),
                (tile_width, tile_height),
            )
            color = palette[color_index]
            if pastel_outer_column and column == columns - 1:
                color = (183, 239, 248)
            if pastel_outer_row and row == rows - 1:
                color = (238, 218, 188)
            left = center_x - current_width // 2
            top = center_y - current_height // 2
            _rounded_rectangle(
                image,
                left,
                top,
                left + current_width - 1,
                top + current_height - 1,
                color,
            )
            if (row, column) in cat_sprite_slots:
                _draw_cat_like_sprite(
                    image,
                    center_x,
                    center_y,
                    current_width,
                    current_height,
                )
        centers.append(tuple(center_row))
        color_indices.append(tuple(color_row))

    if duplicate_slot is not None:
        row, column = duplicate_slot
        center_x, center_y = centers[row][column]
        small_width = max(5, tile_width // 3)
        _rounded_rectangle(
            image,
            center_x - small_width,
            center_y - small_width,
            center_x - 2,
            center_y - 2,
            (55, 98, 224),
        )

    horizontal_lines = tuple(board_y + index * pitch_y for index in range(rows + 1))
    vertical_lines = tuple(board_x + index * pitch_x for index in range(columns + 1))
    screenshot = Screenshot(
        image=image,
        width=screenshot_width,
        height=screenshot_height,
        timestamp=datetime(2026, 8, 6, tzinfo=UTC),
    )
    return SyntheticCatsTileGrid(
        screenshot=screenshot,
        board_x=board_x,
        board_y=board_y,
        board_width=board_width,
        board_height=board_height,
        horizontal_lines=horizontal_lines,
        vertical_lines=vertical_lines,
        tile_centers=tuple(centers),
        color_indices=tuple(color_indices),
    )
