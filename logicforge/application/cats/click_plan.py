"""Backend-neutral Cats click planning and port-based execution."""

import time
from collections.abc import Callable, Sequence

from logicforge.application.cats.models import CatClickTarget
from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint
from logicforge.core import Board
from logicforge.vision.grid_detector import CellBounds, GridDetection
from logicforge.vision.window_capture import WindowInfo

type SleepFunction = Callable[[float], None]


class CatClickPlanError(RuntimeError):
    """Report inconsistent logical or detected geometry during click mapping."""


class CatClickExecutionError(RuntimeError):
    """Report invalid or failed orchestration of an explicit click plan."""


def collect_cat_coordinates(board: Board) -> tuple[tuple[int, int], ...]:
    """Return every confirmed cat in deterministic zero-based row-major order."""

    return tuple(
        (row, column)
        for row, values in enumerate(board.cells)
        for column in range(len(values))
        if board.is_cat(row, column)
    )


def get_grid_cell(
    grid: GridDetection,
    row: int,
    column: int,
) -> CellBounds:
    """Return one row-major cell or raise a typed consistency error."""

    requested_coordinates = f"row={row}, column={column}"
    if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) is outside the "
            f"detected {grid.rows}x{grid.columns} grid."
        )

    index = row * grid.columns + column
    try:
        cell = grid.cells[index]
    except IndexError as error:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) has no row-major "
            f"entry at index {index}."
        ) from error

    if cell.row != row or cell.column != column:
        raise CatClickPlanError(
            f"Requested grid cell ({requested_coordinates}) resolved to "
            f"cell row={cell.row}, column={cell.column} at row-major index {index}."
        )
    return cell


def create_cat_click_target(
    window: WindowInfo,
    grid: GridDetection,
    row: int,
    column: int,
) -> CatClickTarget:
    """Map one logical cat coordinate to screenshot and desktop centers."""

    cell = get_grid_cell(grid, row, column)
    return CatClickTarget(
        row=row,
        column=column,
        screenshot_x=cell.center_x,
        screenshot_y=cell.center_y,
        desktop_x=window.bounds.x + cell.center_x,
        desktop_y=window.bounds.y + cell.center_y,
    )


def build_cat_click_plan(
    board: Board,
    grid: GridDetection,
    window: WindowInfo,
    *,
    existing_cat_coordinates: Sequence[tuple[int, int]] = (),
) -> tuple[CatClickTarget, ...]:
    """Map only new K cells after validating dimensions and exclusions first."""

    if len(board.cells) != grid.rows:
        raise CatClickPlanError(
            f"Board has {len(board.cells)} rows but detected grid has "
            f"{grid.rows} rows."
        )
    for row, values in enumerate(board.cells):
        if len(values) != grid.columns:
            raise CatClickPlanError(
                f"Board row {row} has {len(values)} columns but detected grid "
                f"has {grid.columns} columns."
            )

    excluded = tuple(existing_cat_coordinates)
    if len(excluded) != len(set(excluded)):
        raise CatClickPlanError("Existing cat coordinates contain duplicates.")
    for row, column in excluded:
        if row < 0 or column < 0 or row >= grid.rows or column >= grid.columns:
            raise CatClickPlanError(
                f"Existing cat coordinate ({row}, {column}) is outside the "
                f"detected {grid.rows}x{grid.columns} grid."
            )
        if not board.is_cat(row, column):
            raise CatClickPlanError(
                f"Existing cat coordinate ({row}, {column}) is not K on final Board."
            )
    excluded_set = set(excluded)
    new_cats = tuple(
        coordinate
        for coordinate in collect_cat_coordinates(board)
        if coordinate not in excluded_set
    )
    return tuple(
        create_cat_click_target(window, grid, row, column) for row, column in new_cats
    )


def execute_cat_click_plan(
    targets: tuple[CatClickTarget, ...],
    mouse_controller: MouseController,
    *,
    click_delay_seconds: float = 0.01,
    sleep_function: SleepFunction = time.sleep,
) -> int:
    """Double-click every target in order with one delay between all clicks."""

    if click_delay_seconds < 0:
        raise CatClickExecutionError(
            "Click delay must be greater than or equal to zero seconds."
        )

    for target_index, target in enumerate(targets):
        point = ScreenPoint(x=target.desktop_x, y=target.desktop_y)
        mouse_controller.click(point, MouseButton.LEFT)
        sleep_function(click_delay_seconds)
        mouse_controller.click(point, MouseButton.LEFT)
        if target_index < len(targets) - 1:
            sleep_function(click_delay_seconds)

    return len(targets)
