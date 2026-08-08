"""Cats solve solving tests."""

from collections.abc import Callable

import pytest

from cats_solve_test_support import (
    _ambiguous_exact_result,
    _board_detection,
    _CaptureService,
    _click_targets,
    _color_result,
    _grid_detection,
    _logical_board,
    _offset_window,
    _set_complete_result,
    _set_stalled_result,
    _square_grid_detection,
)
from logicforge.application import cats as cats_app
from logicforge.application.cats import solving as cats_solving
from logicforge.application.cats.models import CatsSolveStatus
from logicforge.core import Board
from logicforge.plugins.cats.board_actions import place_cat
from logicforge.plugins.cats.exact_search import (
    CatsExactSearchResult,
    CatsExactSearchStatus,
)
from logicforge.plugins.cats.existing_cat import (
    CatsExistingCatDetection,
    CatsExistingCatDiagnostics,
    CatsExistingCatObservation,
)
from logicforge.vision.board_detector import (
    BoardDetection,
)
from logicforge.vision.color_detector import (
    ColorDetectionResult,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowBounds,
    WindowInfo,
)


@pytest.mark.parametrize(
    ("rule_application", "expected_status", "expected_plan_calls"),
    (
        (_set_complete_result, CatsSolveStatus.COMPLETE, 1),
        (_set_stalled_result, CatsSolveStatus.AMBIGUOUS, 0),
    ),
)
def test_solve_analyzed_board_creates_solves_and_maps_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    rule_application: Callable[[Board], int],
    expected_status: CatsSolveStatus,
    expected_plan_calls: int,
) -> None:
    """Create one Board, run one rule loop, and build one plan per analysis."""

    actual_board_type = Board
    calls = {"board": 0, "rules": 0, "plan": 0}

    def create_board(color_result: ColorDetectionResult) -> Board:
        calls["board"] += 1
        return actual_board_type(color_result)

    def run_rules(board: Board) -> int:
        calls["rules"] += 1
        return rule_application(board)

    expected_plan = _click_targets()

    def build_plan(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[cats_app.CatClickTarget, ...]:
        del board
        calls["plan"] += 1
        assert grid == _grid_detection()
        assert window == _offset_window()
        return expected_plan

    monkeypatch.setattr(cats_solving, "Board", create_board)
    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", run_rules)
    monkeypatch.setattr(cats_solving, "build_cat_click_plan", build_plan)
    monkeypatch.setattr(
        cats_solving,
        "solve_cats_exact",
        lambda *args, **kwargs: _ambiguous_exact_result(),
    )
    board_input = cats_app.CatsBoardInput(
        detected_board=_board_detection(),
        grid=_grid_detection(),
        color_result=_color_result(),
    )

    solved = cats_app.solve_analyzed_cats_board(_offset_window(), board_input)

    assert calls == {"board": 1, "rules": 1, "plan": expected_plan_calls}
    assert solved.board_input is board_input
    assert solved.successful_applications in (1, 2)
    if expected_plan_calls:
        assert solved.click_plan is expected_plan
    else:
        assert solved.click_plan == ()
    assert solved.status == expected_status


def test_stalled_solve_runs_one_exact_search_on_same_board_and_original_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the one logical Board and immutable colors into one fallback call."""

    matrix = tuple(tuple(f"C{column}" for column in range(4)) for _ in range(4))
    color_result = _color_result(matrix)
    grid = _square_grid_detection(4)
    board_input = cats_app.CatsBoardInput(
        detected_board=BoardDetection(0, 0, 80, 80, 0.95),
        grid=grid,
        color_result=color_result,
    )
    solution = ((0, 1), (1, 3), (2, 0), (3, 2))
    created_boards: list[Board] = []
    search_calls: list[tuple[Board, object, int]] = []
    actual_board = Board

    def create_board(result: ColorDetectionResult) -> Board:
        board = actual_board(result)
        created_boards.append(board)
        return board

    exact_result = CatsExactSearchResult(
        status=CatsExactSearchStatus.UNIQUE,
        solution=solution,
        solutions_found=1,
        search_nodes=8,
        propagation_steps=11,
    )

    def search(
        board: Board,
        original_matrix: object,
        *,
        maximum_search_nodes: int,
    ) -> CatsExactSearchResult:
        search_calls.append((board, original_matrix, maximum_search_nodes))
        assert board.cells == [list(row) for row in matrix]
        assert original_matrix is color_result.color_matrix
        return exact_result

    monkeypatch.setattr(cats_solving, "Board", create_board)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        lambda board: 0,
    )
    monkeypatch.setattr(cats_solving, "solve_cats_exact", search)

    solved = cats_app.solve_analyzed_cats_board(
        _offset_window(),
        board_input,
    )

    assert len(created_boards) == 1
    assert search_calls == [(created_boards[0], color_result.color_matrix, 250_000)]
    assert solved.logical_board is created_boards[0]
    assert solved.exact_search_result is exact_result
    assert solved.status is CatsSolveStatus.COMPLETE
    assert (
        tuple((target.row, target.column) for target in solved.click_plan) == solution
    )


def test_rule_complete_board_does_not_invoke_exact_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the seven rules as the preferred successful path."""

    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    def forbidden_search(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("exact search must not run after rule completion")

    monkeypatch.setattr(cats_solving, "solve_cats_exact", forbidden_search)
    board_input = cats_app.CatsBoardInput(
        detected_board=_board_detection(),
        grid=_grid_detection(),
        color_result=_color_result(),
    )

    solved = cats_app.solve_analyzed_cats_board(_offset_window(), board_input)

    assert solved.status is CatsSolveStatus.COMPLETE
    assert solved.exact_search_result is None


@pytest.mark.parametrize(
    ("search_status", "solutions_found", "solved_status"),
    (
        (CatsExactSearchStatus.UNSAT, 0, CatsSolveStatus.UNSAT),
        (CatsExactSearchStatus.AMBIGUOUS, 2, CatsSolveStatus.AMBIGUOUS),
        (CatsExactSearchStatus.LIMIT_REACHED, 0, CatsSolveStatus.SEARCH_LIMIT),
    ),
)
def test_unresolved_exact_statuses_build_no_click_plan(
    monkeypatch: pytest.MonkeyPatch,
    search_status: CatsExactSearchStatus,
    solutions_found: int,
    solved_status: CatsSolveStatus,
) -> None:
    """Do not map even a partial K set after an inconclusive fallback."""

    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )
    monkeypatch.setattr(
        cats_solving,
        "solve_cats_exact",
        lambda *args, **kwargs: CatsExactSearchResult(
            status=search_status,
            solution=None,
            solutions_found=solutions_found,
            search_nodes=3,
            propagation_steps=1,
        ),
    )

    def forbidden_plan(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("non-COMPLETE fallback must not build a click plan")

    monkeypatch.setattr(cats_solving, "build_cat_click_plan", forbidden_plan)
    board_input = cats_app.CatsBoardInput(
        detected_board=_board_detection(),
        grid=_grid_detection(),
        color_result=_color_result(),
    )

    solved = cats_app.solve_analyzed_cats_board(_offset_window(), board_input)

    assert solved.status == solved_status
    assert solved.click_plan == ()


def test_exact_search_keeps_existing_cat_fixed_and_excludes_its_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply one fixed K to search while mapping only the three new cats."""

    matrix = tuple(tuple(f"C{column}" for column in range(4)) for _ in range(4))
    grid = _square_grid_detection(4)
    existing = CatsExistingCatObservation(row=0, column=1, confidence=0.96)
    board_input = cats_app.CatsBoardInput(
        detected_board=BoardDetection(0, 0, 80, 80, 0.95),
        grid=grid,
        color_result=_color_result(matrix),
        existing_cat_detection=CatsExistingCatDetection(
            cats=(existing,),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        lambda board: 0,
    )

    solved = cats_app.solve_analyzed_cats_board(_offset_window(), board_input)

    assert solved.status is CatsSolveStatus.COMPLETE
    assert solved.exact_search_result is not None
    assert solved.exact_search_result.solution == (
        (0, 1),
        (1, 3),
        (2, 0),
        (3, 2),
    )
    assert tuple((target.row, target.column) for target in solved.click_plan) == (
        (1, 3),
        (2, 0),
        (3, 2),
    )


def test_existing_cat_is_placed_before_rules_and_excluded_from_click_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one Board and produce the requested five new targets from six final K."""

    size = 6
    lines = tuple(index * 20 for index in range(size + 1))
    grid = GridDetection(
        horizontal_lines=lines,
        vertical_lines=lines,
        rows=size,
        columns=size,
        cells=tuple(
            CellBounds(
                row=row,
                column=column,
                x=lines[column],
                y=lines[row],
                width=20,
                height=20,
                center_x=lines[column] + 10,
                center_y=lines[row] + 10,
            )
            for row in range(size)
            for column in range(size)
        ),
        confidence=0.95,
    )
    colors = _color_result(
        tuple(tuple(f"C{column}" for column in range(size)) for _ in range(size))
    )
    existing = CatsExistingCatDetection(
        cats=(CatsExistingCatObservation(1, 0, 0.91),),
        diagnostics=CatsExistingCatDiagnostics(cells=()),
    )
    board_input = cats_app.CatsBoardInput(
        detected_board=BoardDetection(0, 0, 120, 120, 0.95),
        grid=grid,
        color_result=colors,
        existing_cat_detection=existing,
    )
    expected_final = ((0, 2), (1, 0), (2, 4), (3, 1), (4, 5), (5, 3))

    def complete(board: Board) -> int:
        assert board.is_cat(1, 0)
        for row, column in expected_final:
            if (row, column) != (1, 0):
                place_cat(board, row, column)
        return 5

    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", complete)
    solved = cats_app.solve_analyzed_cats_board(
        WindowInfo("BlueStacks", WindowBounds(100, 200, 120, 120)),
        board_input,
    )

    assert cats_app.collect_cat_coordinates(solved.logical_board) == expected_final
    assert tuple((target.row, target.column) for target in solved.click_plan) == (
        (0, 2),
        (2, 4),
        (3, 1),
        (4, 5),
        (5, 3),
    )
    assert solved.logical_board.get(1, 0) == "K"


def test_click_plan_validates_existing_coordinates_before_mapping() -> None:
    board = _logical_board()
    board.set_cat(0, 0)
    with pytest.raises(cats_app.CatClickPlanError, match="duplicates"):
        cats_app.build_cat_click_plan(
            board,
            _grid_detection(),
            _offset_window(),
            existing_cat_coordinates=((0, 0), (0, 0)),
        )
    with pytest.raises(cats_app.CatClickPlanError, match="outside"):
        cats_app.build_cat_click_plan(
            board,
            _grid_detection(),
            _offset_window(),
            existing_cat_coordinates=((2, 0),),
        )
    with pytest.raises(cats_app.CatClickPlanError, match="not K"):
        cats_app.build_cat_click_plan(
            board,
            _grid_detection(),
            _offset_window(),
            existing_cat_coordinates=((0, 1),),
        )


def test_solve_captured_board_composes_analysis_then_solve_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose one convenient composition point shared with autoplay."""

    board_input = cats_app.CatsBoardInput(
        detected_board=_board_detection(),
        grid=_grid_detection(),
        color_result=_color_result(),
    )
    solved = cats_app.CatsSolvedBoard(
        board_input=board_input,
        logical_board=_logical_board(),
        successful_applications=0,
        click_plan=(),
        status=CatsSolveStatus.STALLED,
    )
    calls = {"analyze": 0, "solve": 0}

    def analyze(screenshot: Screenshot) -> cats_app.CatsBoardInput:
        calls["analyze"] += 1
        assert screenshot is _CaptureService.screenshot
        return board_input

    def solve(
        window: WindowInfo,
        analyzed: cats_app.CatsBoardInput,
    ) -> cats_app.CatsSolvedBoard:
        calls["solve"] += 1
        assert window == _offset_window()
        assert analyzed is board_input
        return solved

    monkeypatch.setattr(cats_solving, "solve_analyzed_cats_board", solve)

    result = cats_app.solve_captured_cats_board(
        _offset_window(),
        _CaptureService.screenshot,
        analyze_board=analyze,
    )

    assert result is solved
    assert calls == {"analyze": 1, "solve": 1}
