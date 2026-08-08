"""Cats autoplay validation tests."""

from dataclasses import replace

import pytest

from cats_autoplay_test_support import (
    COLUMN_COLOR_MATRIX,
    ROW_COLOR_MATRIX,
    SIX_CAT_COLUMNS,
    FakeCatsScreenStateDetector,
    FakeClock,
    FakeFailureRenderer,
    FakeMouseController,
    FakeSolver,
    FakeWindowCapturer,
    FakeWindowLocator,
    _board_input,
    _detection,
    _geometry_board_input,
    _replace_board_values,
    _runner,
    _settings,
    _solved_board,
    _window,
)
from logicforge.application import cats as cats_app
from logicforge.application.cats import autoplay as cats_autoplay
from logicforge.application.cats.models import CatsSolveStatus
from logicforge.automation.mouse import MouseButton, ScreenPoint
from logicforge.plugins.cats import (
    CatsExistingCatDetection,
    CatsExistingCatDetectionError,
    CatsExistingCatDiagnostics,
    CatsExistingCatObservation,
    CatsScreenState,
)
from logicforge.vision.screenshot import Screenshot


def test_complete_validation_happens_before_first_cat_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never begin a click plan until full solution validation returns."""

    events: list[str] = []
    original_validate = cats_app.validate_complete_cats_solution

    def validate(solved: cats_app.CatsSolvedBoard) -> None:
        events.append("validate")
        original_validate(solved)

    class EventMouse(FakeMouseController):
        def click(
            self,
            point: ScreenPoint,
            button: MouseButton = MouseButton.LEFT,
        ) -> None:
            events.append("click")
            super().click(point, button)

    monkeypatch.setattr(cats_autoplay, "validate_complete_cats_solution", validate)
    runner, *_ = _runner(
        (_detection(CatsScreenState.BOARD),),
        mouse=EventMouse(),
    )

    runner.run()

    assert events[0] == "validate"
    assert events[1] == "click"


@pytest.mark.parametrize(
    "status",
    (
        CatsSolveStatus.STALLED,
        CatsSolveStatus.UNSAT,
        CatsSolveStatus.AMBIGUOUS,
        CatsSolveStatus.SEARCH_LIMIT,
    ),
)
def test_unresolved_solver_status_raises_before_any_click_and_saves_one_overlay(
    status: CatsSolveStatus,
) -> None:
    """Stop every unresolved proof status without emitting a pointer action."""

    runner, _, _, _, mouse, _, renderer, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        statuses=(status,),
    )

    with pytest.raises(cats_app.CatsSolutionValidationError, match=status.value):
        runner.run()
    runner.save_failure_overlay()
    runner.save_failure_overlay()

    assert mouse.clicks == []
    assert len(renderer.calls) == 1


@pytest.mark.parametrize(
    "cats",
    (
        ((0, 1), (1, 3), (2, 0)),
        ((0, 1), (0, 2), (1, 3), (2, 0), (3, 2)),
        ((0, 1), (1, 3), (3, 2)),
        ((0, 1), (1, 3), (2, 0), (2, 2), (3, 2)),
    ),
)
def test_validation_rejects_missing_or_duplicate_row_or_column_cat(
    cats: tuple[tuple[int, int], ...],
) -> None:
    """Require exactly one K in every row and every column."""

    solved = _replace_board_values(_solved_board(), cats)

    with pytest.raises(cats_app.CatsSolutionValidationError):
        cats_app.validate_complete_cats_solution(solved)


def test_validation_rejects_duplicate_or_missing_original_color() -> None:
    """Require exactly one K for every immutable original color identifier."""

    matrix = [list(row) for row in COLUMN_COLOR_MATRIX]
    matrix[0][1] = "C0"
    matrix[0][0] = "C1"
    invalid_input = _board_input(tuple(tuple(row) for row in matrix))
    solved = _solved_board(invalid_input)

    with pytest.raises(cats_app.CatsSolutionValidationError, match="Original color"):
        cats_app.validate_complete_cats_solution(solved)


def test_validation_rejects_touching_cats() -> None:
    """Reject orthogonal or diagonal adjacency after row/column validity."""

    solved = _solved_board(cat_columns=(0, 1, 3, 2))

    with pytest.raises(cats_app.CatsSolutionValidationError, match="touch"):
        cats_app.validate_complete_cats_solution(solved)


@pytest.mark.parametrize("value", ("C0", "?"))
def test_validation_rejects_unresolved_or_unsupported_value(value: str) -> None:
    """Allow only K and X in a claimed complete solution."""

    solved = _solved_board()
    solved.logical_board.cells[0][0] = value

    with pytest.raises(cats_app.CatsSolutionValidationError):
        cats_app.validate_complete_cats_solution(solved)


def test_validation_rejects_click_plan_mismatch_and_duplicate() -> None:
    """Require a unique row-major plan exactly equal to all K coordinates."""

    solved = _solved_board()
    missing = replace(solved, click_plan=solved.click_plan[:-1])
    duplicate = replace(
        solved,
        click_plan=(*solved.click_plan, solved.click_plan[0]),
    )

    with pytest.raises(cats_app.CatsSolutionValidationError, match="match"):
        cats_app.validate_complete_cats_solution(missing)
    with pytest.raises(cats_app.CatsSolutionValidationError, match="duplicate"):
        cats_app.validate_complete_cats_solution(duplicate)


def test_complete_six_by_six_with_one_existing_cat_has_five_new_targets() -> None:
    board_input = _geometry_board_input(6, 6, 6)
    board_input = replace(
        board_input,
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(1, 0, 0.93),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    solved = _solved_board(board_input, cat_columns=SIX_CAT_COLUMNS)

    cats_app.validate_complete_cats_solution(solved)

    assert len(cats_app.collect_cat_coordinates(solved.logical_board)) == 6
    assert len(solved.click_plan) == 5
    assert (1, 0) not in tuple(
        (target.row, target.column) for target in solved.click_plan
    )


def test_autoplay_executes_only_new_cats_when_one_is_existing() -> None:
    board_input = replace(
        _geometry_board_input(6, 6, 6),
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(1, 0, 0.93),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(board_input,),
    )

    summary = runner.run()

    assert summary.low_level_cat_clicks == 10
    assert len(mouse.clicks) == 10
    existing_cell = board_input.grid.cells[1 * 6]
    existing_desktop = ScreenPoint(
        _window().bounds.x + existing_cell.center_x,
        _window().bounds.y + existing_cell.center_y,
    )
    assert all(point != existing_desktop for point, _ in mouse.clicks)


def test_several_existing_cats_reduce_autoplay_click_count() -> None:
    existing_coordinates = ((1, 0), (3, 1), (5, 3))
    board_input = replace(
        _geometry_board_input(6, 6, 6),
        existing_cat_detection=CatsExistingCatDetection(
            cats=tuple(
                CatsExistingCatObservation(row, column, 0.9)
                for row, column in existing_coordinates
            ),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    runner, _, _, _, mouse, _, _, _, _ = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(board_input,),
    )

    summary = runner.run()

    assert summary.low_level_cat_clicks == 6
    assert len(mouse.clicks) == 6


def test_existing_cat_missing_from_final_k_fails_validation() -> None:
    solved = _solved_board()
    invalid_input = replace(
        solved.board_input,
        existing_cat_detection=CatsExistingCatDetection(
            cats=(CatsExistingCatObservation(0, 0, 0.9),),
            diagnostics=CatsExistingCatDiagnostics(cells=()),
        ),
    )
    invalid = replace(solved, board_input=invalid_input)
    with pytest.raises(cats_app.CatsSolutionValidationError, match="not K"):
        cats_app.validate_complete_cats_solution(invalid)


def test_duplicate_existing_coordinates_fail_validation() -> None:
    solved = _solved_board()
    detection = CatsExistingCatDetection(
        cats=(CatsExistingCatObservation(0, 1, 0.9),),
        diagnostics=CatsExistingCatDiagnostics(cells=()),
    )
    object.__setattr__(
        detection,
        "cats",
        (
            CatsExistingCatObservation(0, 1, 0.9),
            CatsExistingCatObservation(0, 1, 0.9),
        ),
    )
    invalid = replace(
        solved,
        board_input=replace(solved.board_input, existing_cat_detection=detection),
    )
    with pytest.raises(cats_app.CatsSolutionValidationError, match="duplicate"):
        cats_app.validate_complete_cats_solution(invalid)


def test_existing_cat_detection_contradiction_emits_zero_clicks() -> None:
    locator = FakeWindowLocator()
    capturer = FakeWindowCapturer()
    detector = FakeCatsScreenStateDetector((_detection(CatsScreenState.BOARD),))
    mouse = FakeMouseController()
    clock = FakeClock()

    def reject_analysis(screenshot: Screenshot) -> cats_app.CatsBoardInput:
        del screenshot
        diagnostics = CatsExistingCatDiagnostics(
            cells=(),
            rejection_reasons=("multiple existing cats were detected in one row",),
        )
        raise CatsExistingCatDetectionError("synthetic contradiction", diagnostics)

    runner = cats_app.CatsAutoplayRunner(
        settings=_settings(timeout=20.0, board_retry=0.3),
        locator=locator,
        capturer=capturer,
        detector=detector,
        renderer=FakeFailureRenderer(),
        mouse_controller=mouse,
        sleep_function=clock.sleep,
        monotonic_function=clock.monotonic,
        analyze_board=reject_analysis,
        solve_board=FakeSolver(),
    )

    with pytest.raises(CatsExistingCatDetectionError):
        runner.run()
    assert mouse.clicks == []
    assert len(capturer.calls) == 4


def test_validation_rejects_inconsistent_rows_columns_and_color_count() -> None:
    """Fail when one-cat row, column, and color counts cannot all agree."""

    solved = _solved_board()
    object.__setattr__(solved.board_input.color_result, "color_count", 3)

    with pytest.raises(cats_app.CatsSolutionValidationError, match="equal"):
        cats_app.validate_complete_cats_solution(solved)


def test_eight_cat_targets_emit_sixteen_row_major_low_level_clicks() -> None:
    """Retain existing two-click orchestration and target ordering."""

    first = _solved_board()
    second = _solved_board(_board_input(ROW_COLOR_MATRIX))
    targets = (*first.click_plan, *second.click_plan)
    mouse = FakeMouseController()
    clock = FakeClock()

    executed = cats_app.execute_cat_click_plan(
        targets,
        mouse,
        click_delay_seconds=0.01,
        sleep_function=clock.sleep,
    )

    assert executed == 8
    assert len(mouse.clicks) == 16
    assert [click[0] for click in mouse.clicks[::2]] == [
        ScreenPoint(target.desktop_x, target.desktop_y) for target in targets
    ]
    assert set(clock.sleeps) == {0.01}
