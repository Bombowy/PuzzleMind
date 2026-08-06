"""Capture once and diagnose existing Cats occupancy without input actions."""

import sys
from pathlib import Path
from typing import Final

from logicforge.infrastructure.opencv_cats_existing_cat_detector import (
    OpenCvCatsExistingCatDetector,
)
from logicforge.infrastructure.opencv_cats_existing_cat_renderer import (
    CatsExistingCatDebugRenderError,
    OpenCvCatsExistingCatDebugRenderer,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.infrastructure.opencv_color_detector import OpenCvColorDetector
from logicforge.infrastructure.windows import (
    MssWindowCapturer,
    Win32BlueStacksWindowLocator,
)
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
)
from logicforge.plugins.cats.tile_grid import (
    CatsTileGridDetection,
    CatsTileGridDetectionError,
)
from logicforge.vision.color_detector import ColorDetectionError, ColorDetectionResult
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowCaptureService,
    WindowInfo,
)

DEBUG_OUTPUT_PATH: Final = Path("artifacts/vision/cats_existing_cat_detection.png")


def print_detection_information(
    window: WindowInfo,
    screenshot: Screenshot,
    tile_grid: CatsTileGridDetection,
    colors: ColorDetectionResult,
    existing: CatsExistingCatDetection,
) -> None:
    """Print accepted cats plus the strongest rejected per-cell evidence."""

    board = tile_grid.board
    print(f"Window title: {window.title}")
    print(f"Screenshot resolution: {screenshot.width}x{screenshot.height} pixels")
    print("Grid:")
    print(f"  rows: {tile_grid.grid.rows}")
    print(f"  columns: {tile_grid.grid.columns}")
    print(f"  cell count: {len(tile_grid.grid.cells)}")
    print(
        f"  board bounds: x={board.x}, y={board.y}, "
        f"width={board.width}, height={board.height}"
    )
    print("Colors:")
    print(f"  color_count: {colors.color_count}")
    print("Existing cats:")
    print(f"  count: {len(existing.cats)}")
    accepted = {(cat.row, cat.column): cat for cat in existing.cats}
    for cat in existing.cats:
        print(
            f"  ({cat.row},{cat.column}) "
            f"original_color={colors.color_matrix[cat.row][cat.column]} "
            f"confidence={cat.confidence:.3f}"
        )
    suspicious = sorted(
        (cell for cell in existing.diagnostics.cells if not cell.accepted),
        key=lambda cell: (-cell.score, cell.row, cell.column),
    )[:5]
    print("Accepted and strongest rejected cell metrics:")
    for cell in (
        *(
            cell
            for cell in existing.diagnostics.cells
            if (cell.row, cell.column) in accepted
        ),
        *suspicious,
    ):
        print(
            f"  ({cell.row},{cell.column}) accepted={cell.accepted} "
            f"foreground_ratio={cell.foreground_ratio:.3f} "
            f"largest_component_ratio={cell.largest_component_ratio:.3f} "
            f"component_width_ratio={cell.component_width_ratio:.3f} "
            f"component_height_ratio={cell.component_height_ratio:.3f} "
            f"center_offset_ratio={cell.center_offset_ratio:.3f} "
            f"score={cell.score:.3f}"
        )
    print(f"Debug output path: {DEBUG_OUTPUT_PATH.as_posix()}")


def main() -> int:
    """Run locate/capture/grid/colors/cats/render exactly once and zero clicks."""

    capture_service = WindowCaptureService(
        locator=Win32BlueStacksWindowLocator(),
        capturer=MssWindowCapturer(),
    )
    try:
        window = capture_service.locate_window()
        screenshot = capture_service.capture_window(window, debug=False)
    except WindowCaptureError as error:
        print(f"BlueStacks capture failed: {error}", file=sys.stderr)
        return 1
    try:
        tile_grid = OpenCvCatsTileGridDetector().detect(screenshot)
    except CatsTileGridDetectionError as error:
        print(f"Cats tile-grid detection failed: {error}", file=sys.stderr)
        return 2
    try:
        colors = OpenCvColorDetector().detect(screenshot, tile_grid.grid)
    except ColorDetectionError as error:
        print(f"Cats color detection failed: {error}", file=sys.stderr)
        return 3
    try:
        existing = OpenCvCatsExistingCatDetector().detect(
            screenshot,
            tile_grid.grid,
            colors,
        )
    except CatsExistingCatDetectionError as error:
        print(f"Existing cat detection failed: {error}", file=sys.stderr)
        return 4
    try:
        saved = OpenCvCatsExistingCatDebugRenderer().save_debug_overlay(
            screenshot,
            tile_grid.grid,
            existing,
            DEBUG_OUTPUT_PATH,
            debug=True,
        )
    except CatsExistingCatDebugRenderError as error:
        print(f"Existing cat debug rendering failed: {error}", file=sys.stderr)
        return 5
    if saved is None:
        print("Existing cat debug rendering produced no output.", file=sys.stderr)
        return 5
    print_detection_information(window, screenshot, tile_grid, colors, existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
