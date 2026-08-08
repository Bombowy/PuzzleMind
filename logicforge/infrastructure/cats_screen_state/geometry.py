"""Shared screen-state morphology and rectangle geometry."""

from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from logicforge.plugins.cats.screen_state import CatsScreenRect


def relative_kernel(width: int, height: int, ratio: float) -> NDArray[np.uint8]:
    """Build one odd scale-relative morphology kernel for any resolution."""

    size = max(3, round(min(width, height) * ratio))
    if size % 2 == 0:
        size += 1
    return cast(
        NDArray[np.uint8],
        cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)),
    )


def union_rect(rectangles: tuple[CatsScreenRect, ...]) -> CatsScreenRect:
    """Return the smallest half-open rectangle containing every input rect."""

    left = min(rect.x for rect in rectangles)
    top = min(rect.y for rect in rectangles)
    right = max(rect.x + rect.width for rect in rectangles)
    bottom = max(rect.y + rect.height for rect in rectangles)
    return CatsScreenRect(x=left, y=top, width=right - left, height=bottom - top)
