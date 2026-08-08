"""Cats solve click_plan tests."""

import pytest

from cats_solve_test_support import (
    _color_result,
    _logical_board,
    _offset_window,
    _rectangular_grid_detection,
    _rectangular_logical_board,
)
from logicforge.application import cats as cats_app
from logicforge.application.cats import click_plan as cats_click_plan
from logicforge.application.cats.models import CatsSolveStatus
from logicforge.core import Board
from logicforge.vision.grid_detector import (
    GridDetection,
)
from logicforge.vision.window_capture import (
    WindowInfo,
)


def test_collect_cat_coordinates_returns_only_cats() -> None:
    """Exclude unresolved and blocked values from the diagnostic coordinate list."""

    board = _logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 0)

    assert cats_app.collect_cat_coordinates(board) == ((0, 1),)


def test_collect_cat_coordinates_uses_row_major_order() -> None:
    """Return zero-based coordinates deterministically regardless of mutation order."""

    board = _logical_board()
    board.set_cat(1, 1)
    board.set_cat(0, 1)
    board.set_cat(1, 0)

    assert cats_app.collect_cat_coordinates(board) == ((0, 1), (1, 0), (1, 1))


def test_count_unresolved_cells_counts_only_color_ids() -> None:
    """Count every remaining C<n> entry through the Board query contract."""

    board = _logical_board()
    board.set_cat(0, 0)

    assert cats_app.count_unresolved_cells(board) == 3


def test_cat_and_blocked_cells_are_not_unresolved() -> None:
    """Exclude both terminal Board states from the unresolved count."""

    board = _logical_board()
    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    board.set_cat(1, 0)
    board.set_blocked(1, 1)

    assert cats_app.count_unresolved_cells(board) == 0


def test_classify_result_returns_complete_for_terminal_board() -> None:
    """Report COMPLETE only when no logical color candidate remains."""

    board = _logical_board()
    board.set_cat(0, 0)
    board.set_blocked(0, 1)
    board.set_blocked(1, 0)
    board.set_cat(1, 1)

    assert cats_app.classify_result(board) is CatsSolveStatus.COMPLETE


def test_classify_result_returns_stalled_with_unknown_cell() -> None:
    """Report STALLED whenever at least one C<n> remains after deductions."""

    board = _logical_board()
    board.set_cat(0, 0)

    assert cats_app.classify_result(board) is CatsSolveStatus.STALLED


def test_cats_solve_status_preserves_existing_cli_values() -> None:
    """Close the production status API while retaining exact terminal text."""

    assert tuple(status.value for status in CatsSolveStatus) == (
        "COMPLETE",
        "STALLED",
        "UNSAT",
        "AMBIGUOUS",
        "SEARCH_LIMIT",
    )


def test_get_grid_cell_returns_first_cell() -> None:
    """Resolve row zero, column zero directly from row-major geometry."""

    grid = _rectangular_grid_detection()

    assert cats_app.get_grid_cell(grid, 0, 0) is grid.cells[0]


def test_get_grid_cell_returns_last_cell() -> None:
    """Resolve the final logical coordinate without scanning all cells."""

    grid = _rectangular_grid_detection()

    assert cats_app.get_grid_cell(grid, 1, 2) is grid.cells[-1]


def test_get_grid_cell_uses_row_major_index() -> None:
    """Map coordinates through row * columns + column."""

    grid = _rectangular_grid_detection()

    assert cats_app.get_grid_cell(grid, 1, 1) is grid.cells[4]


def test_get_grid_cell_rejects_negative_row() -> None:
    """Reject negative logical rows with requested coordinates in the error."""

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=-1, column=0",
    ):
        cats_app.get_grid_cell(_rectangular_grid_detection(), -1, 0)


def test_get_grid_cell_rejects_negative_column() -> None:
    """Reject negative logical columns with requested coordinates in the error."""

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=0, column=-1",
    ):
        cats_app.get_grid_cell(_rectangular_grid_detection(), 0, -1)


def test_get_grid_cell_rejects_row_outside_grid() -> None:
    """Reject a row equal to the public row count."""

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=2, column=0",
    ):
        cats_app.get_grid_cell(_rectangular_grid_detection(), 2, 0)


def test_get_grid_cell_rejects_column_outside_grid() -> None:
    """Reject a column equal to the public column count."""

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=0, column=3",
    ):
        cats_app.get_grid_cell(_rectangular_grid_detection(), 0, 3)


def test_get_grid_cell_rejects_inconsistent_cell_coordinates() -> None:
    """Fail closed if supposedly row-major public geometry is corrupted."""

    grid = _rectangular_grid_detection()
    object.__setattr__(grid.cells[4], "column", 0)

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=1, column=1",
    ):
        cats_app.get_grid_cell(grid, 1, 1)


def test_get_grid_cell_rejects_missing_row_major_entry() -> None:
    """Translate defensive tuple corruption into the local typed error."""

    grid = _rectangular_grid_detection()
    object.__setattr__(grid, "cells", grid.cells[:-1])

    with pytest.raises(
        cats_app.CatClickPlanError,
        match=r"row=1, column=2",
    ):
        cats_app.get_grid_cell(grid, 1, 2)


def test_create_target_preserves_logical_coordinates() -> None:
    """Carry zero-based row and column into the immutable dry-run target."""

    target = cats_app.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert (target.row, target.column) == (1, 2)


def test_create_target_uses_exact_cell_center_x() -> None:
    """Use CellBounds.center_x instead of recalculating board geometry."""

    target = cats_app.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.screenshot_x == 300


def test_create_target_uses_exact_cell_center_y() -> None:
    """Use CellBounds.center_y instead of recalculating board geometry."""

    target = cats_app.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.screenshot_y == 250


def test_create_target_adds_window_x_to_screenshot_center() -> None:
    """Translate screenshot x through the window's virtual-desktop origin."""

    target = cats_app.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.desktop_x == 400 + 300


def test_create_target_adds_window_y_to_screenshot_center() -> None:
    """Translate screenshot y through the window's virtual-desktop origin."""

    target = cats_app.create_cat_click_target(
        _offset_window(),
        _rectangular_grid_detection(),
        1,
        2,
    )

    assert target.desktop_y == 300 + 250


def test_create_target_maps_shifted_window() -> None:
    """Map a non-origin BlueStacks window using direct offset addition."""

    target = cats_app.create_cat_click_target(
        _offset_window(x=300, y=200),
        _rectangular_grid_detection(),
        1,
        1,
    )

    assert (target.screenshot_x, target.screenshot_y) == (200, 250)
    assert (target.desktop_x, target.desktop_y) == (500, 450)


def test_create_target_accepts_negative_window_position() -> None:
    """Preserve valid negative desktop coordinates on a secondary monitor."""

    target = cats_app.create_cat_click_target(
        _offset_window(x=-1000, y=50),
        _rectangular_grid_detection(),
        0,
        0,
    )

    assert (target.desktop_x, target.desktop_y) == (-900, 200)


def test_build_click_plan_contains_only_cat_cells() -> None:
    """Map confirmed K while excluding every other logical state."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 0)

    plan = cats_app.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple((target.row, target.column) for target in plan) == ((0, 1),)


def test_build_click_plan_excludes_blocked_cells() -> None:
    """Never create a click target for X."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    board.set_blocked(1, 2)

    plan = cats_app.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert all((target.row, target.column) != (1, 2) for target in plan)


def test_build_click_plan_excludes_unresolved_colors() -> None:
    """Never create a click target for C<n>."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)

    plan = cats_app.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert all((target.row, target.column) != (1, 2) for target in plan)


def test_build_click_plan_orders_multiple_cats_row_major() -> None:
    """Retain collect_cat_coordinates ordering regardless of mutation order."""

    board = _rectangular_logical_board()
    board.set_cat(1, 2)
    board.set_cat(0, 1)
    board.set_cat(1, 0)

    plan = cats_app.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple((target.row, target.column) for target in plan) == (
        (0, 1),
        (1, 0),
        (1, 2),
    )


def test_build_click_plan_returns_empty_tuple_without_cats() -> None:
    """Represent a valid no-click dry run without sentinel values."""

    plan = cats_app.build_cat_click_plan(
        _rectangular_logical_board(),
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert plan == ()


def test_build_click_plan_rejects_board_row_count_mismatch() -> None:
    """Validate all Board dimensions before producing any targets."""

    board = Board(_color_result((("C0", "C1", "C2"),)))

    with pytest.raises(cats_app.CatClickPlanError, match="1 rows"):
        cats_app.build_cat_click_plan(
            board,
            _rectangular_grid_detection(),
            _offset_window(),
        )


def test_build_click_plan_rejects_board_column_count_mismatch() -> None:
    """Reject rows that do not match the detected public grid width."""

    with pytest.raises(cats_app.CatClickPlanError, match="2 columns"):
        cats_app.build_cat_click_plan(
            _logical_board(),
            _rectangular_grid_detection(),
            _offset_window(),
        )


def test_dimension_error_creates_no_partial_click_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish complete dimension validation before mapping the first K."""

    board = _rectangular_logical_board()
    board.set_cat(0, 0)
    board.cells[1].pop()
    mapping_calls: list[tuple[int, int]] = []

    def record_mapping(
        window: WindowInfo,
        grid: GridDetection,
        row: int,
        column: int,
    ) -> cats_app.CatClickTarget:
        """Fail the test if mapping starts before shape validation ends."""

        del window, grid
        mapping_calls.append((row, column))
        raise AssertionError("partial click plan was started")

    monkeypatch.setattr(cats_click_plan, "create_cat_click_target", record_mapping)

    with pytest.raises(cats_app.CatClickPlanError):
        cats_app.build_cat_click_plan(
            board,
            _rectangular_grid_detection(),
            _offset_window(),
        )
    assert mapping_calls == []


def test_build_click_plan_does_not_mutate_board() -> None:
    """Keep the sole logical Board unchanged during coordinate mapping."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    expected = tuple(tuple(row) for row in board.cells)

    cats_app.build_cat_click_plan(
        board,
        _rectangular_grid_detection(),
        _offset_window(),
    )

    assert tuple(tuple(row) for row in board.cells) == expected


def test_build_click_plan_does_not_mutate_grid() -> None:
    """Treat immutable GridDetection as read-only geometry input."""

    board = _rectangular_logical_board()
    board.set_cat(0, 1)
    grid = _rectangular_grid_detection()
    expected = grid

    cats_app.build_cat_click_plan(board, grid, _offset_window())

    assert grid == expected
