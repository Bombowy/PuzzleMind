"""Shared scale-relative geometry for color detector and debug rendering."""

from logicforge.config.settings import ColorDetectionSettings
from logicforge.vision.grid_detector import CellBounds

type SampleBounds = tuple[int, int, int, int]


class CornerSampleGeometryError(RuntimeError):
    """Report a cell too small for four positive inset patch rectangles."""


def corner_sample_bounds(
    cell: CellBounds,
    settings: ColorDetectionSettings,
) -> tuple[SampleBounds, ...]:
    """Return TL, TR, BL, BR half-open patches using the established rounding."""

    patch_width = round(cell.width * settings.corner_sample_patch_fraction)
    patch_height = round(cell.height * settings.corner_sample_patch_fraction)
    offset_x = round(cell.width * settings.corner_sample_offset_fraction)
    offset_y = round(cell.height * settings.corner_sample_offset_fraction)
    left = cell.x + offset_x
    right = cell.x + cell.width - offset_x - patch_width
    top = cell.y + offset_y
    bottom = cell.y + cell.height - offset_y - patch_height
    bounds = (
        (left, top, left + patch_width, top + patch_height),
        (right, top, right + patch_width, top + patch_height),
        (left, bottom, left + patch_width, bottom + patch_height),
        (right, bottom, right + patch_width, bottom + patch_height),
    )
    cell_right = cell.x + cell.width
    cell_bottom = cell.y + cell.height
    center_x = cell.x + cell.width / 2.0
    center_y = cell.y + cell.height / 2.0
    if patch_width <= 0 or patch_height <= 0:
        raise CornerSampleGeometryError(
            "has degenerate corner-patch dimensions after relative rounding"
        )
    if any(
        patch_left < cell.x
        or patch_top < cell.y
        or patch_right > cell_right
        or patch_bottom > cell_bottom
        or patch_right <= patch_left
        or patch_bottom <= patch_top
        for patch_left, patch_top, patch_right, patch_bottom in bounds
    ):
        raise CornerSampleGeometryError(
            "has a corner patch outside its half-open cell bounds"
        )
    if not (
        bounds[0][2] <= center_x
        and bounds[2][2] <= center_x
        and bounds[1][0] >= center_x
        and bounds[3][0] >= center_x
        and bounds[0][3] <= center_y
        and bounds[1][3] <= center_y
        and bounds[2][1] >= center_y
        and bounds[3][1] >= center_y
    ):
        raise CornerSampleGeometryError(
            "has a rounded corner patch crossing the cell center"
        )
    return bounds
