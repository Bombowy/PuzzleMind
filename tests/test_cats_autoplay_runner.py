"""Cats autoplay runner tests."""

import pytest

from cats_autoplay_test_support import (
    COLUMN_COLOR_MATRIX,
    ROW_COLOR_MATRIX,
    _board_input,
    _detection,
    _geometry_board_input,
    _runner,
    _settings,
)
from logicforge.application.cats import validation as cats_validation
from logicforge.plugins.cats import (
    CatsScreenState,
)


@pytest.mark.parametrize(
    "state",
    (
        CatsScreenState.BOARD,
        CatsScreenState.RANKING,
        CatsScreenState.LEVEL_COMPLETE,
        CatsScreenState.UNKNOWN,
    ),
)
def test_dry_run_captures_and_classifies_exactly_once(state: CatsScreenState) -> None:
    """Never enter polling when explicit execution is absent."""

    components = _runner((_detection(state),), settings=_settings(execute=False))
    runner, _, capturer, detector, mouse, clock, _, analyzer, _ = components

    summary = runner.run()

    assert len(capturer.calls) == 1
    assert detector.calls == 1
    assert mouse.clicks == []
    assert clock.sleeps == []
    assert analyzer.calls == (1 if state is CatsScreenState.BOARD else 0)
    assert summary.final_screen_state is state


@pytest.mark.parametrize(
    ("state", "expected_text"),
    (
        (CatsScreenState.BOARD, "complete Cats click plan validated"),
        (CatsScreenState.RANKING, "RANKING single click"),
        (CatsScreenState.LEVEL_COMPLETE, "LEVEL_COMPLETE single click"),
        (CatsScreenState.UNKNOWN, "synthetic unknown state"),
    ),
)
def test_dry_run_prints_state_specific_plan(
    state: CatsScreenState,
    expected_text: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Explain one action without any mouse controller use."""

    runner, *_ = _runner((_detection(state),), settings=_settings(execute=False))

    runner.run()

    assert expected_text in capsys.readouterr().out


def test_board_ranking_level_board_sequence_solves_two_levels() -> None:
    """React to the actual optional-ranking path rather than a fixed screen count."""

    states = tuple(
        _detection(state)
        for state in (
            CatsScreenState.BOARD,
            CatsScreenState.RANKING,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        )
    )
    components = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )
    runner, _, capturer, _, mouse, _, _, analyzer, solver = components

    summary = runner.run()

    assert summary.solved_levels == 2
    assert summary.ranking_clicks == 1
    assert summary.level_button_clicks == 1
    assert summary.low_level_cat_clicks == 16
    assert len(capturer.calls) == 4
    assert analyzer.calls == 2
    assert len(solver.calls) == 2
    assert len(mouse.clicks) == 18


def test_board_level_board_sequence_works_without_ranking() -> None:
    """Allow LEVEL_COMPLETE to follow a solved board directly."""

    states = tuple(
        _detection(state)
        for state in (
            CatsScreenState.BOARD,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        )
    )
    runner, *_, analyzer, solver = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    summary = runner.run()

    assert summary.solved_levels == 2
    assert summary.ranking_clicks == 0
    assert summary.level_button_clicks == 1
    assert analyzer.calls == 2
    assert len(solver.calls) == 2


@pytest.mark.parametrize(
    "initial_states",
    (
        (
            CatsScreenState.RANKING,
            CatsScreenState.LEVEL_COMPLETE,
            CatsScreenState.BOARD,
        ),
        (CatsScreenState.LEVEL_COMPLETE, CatsScreenState.BOARD),
    ),
)
def test_autoplay_can_start_on_transition_overlay(
    initial_states: tuple[CatsScreenState, ...],
) -> None:
    """Start from either accepted transition state and reach the next board."""

    runner, *_ = _runner(tuple(_detection(state) for state in initial_states))

    summary = runner.run()

    assert summary.solved_levels == 1
    assert summary.level_button_clicks == 1
    assert summary.ranking_clicks == (
        1 if CatsScreenState.RANKING in initial_states else 0
    )


def test_repeated_board_waiting_for_transition_does_not_resolve() -> None:
    """Run the solver only once while the just-clicked board remains visible."""

    states = (
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
    )
    runner, *_, analyzer, solver = _runner(
        states,
        settings=_settings(max_levels=2),
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    runner.run()

    assert analyzer.calls == 2
    assert len(solver.calls) == 2


def test_new_board_delay_and_old_fingerprint_guard_solver() -> None:
    """Wait after level click and reject the old immutable matrix before solving."""

    states = (
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.LEVEL_COMPLETE),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
        _detection(CatsScreenState.BOARD),
    )
    clock_settings = _settings(max_levels=2, new_board_delay=0.2)
    runner, *_, clock, _, analyzer, solver = _runner(
        states,
        settings=clock_settings,
        board_inputs=(
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(COLUMN_COLOR_MATRIX),
            _board_input(ROW_COLOR_MATRIX),
        ),
    )

    summary = runner.run()

    assert summary.solved_levels == 2
    assert clock.sleeps.count(0.1) >= 2
    assert analyzer.calls == 3
    assert len(solver.calls) == 2


@pytest.mark.parametrize("size", (5, 8, 9))
def test_cats_input_geometry_guard_accepts_consistent_square_sizes(size: int) -> None:
    """Permit arbitrary square Cats dimensions when color count and matrix agree."""

    cats_validation.validate_cats_board_input_geometry(
        _geometry_board_input(size, size, size)
    )


def test_cats_input_geometry_guard_rejects_rectangular_tile_lattice() -> None:
    """Keep square Cats validity separate from rectangular lattice geometry."""

    with pytest.raises(cats_validation.CatsBoardGeometryMismatchError):
        cats_validation.validate_cats_board_input_geometry(
            _geometry_board_input(6, 9, 9)
        )


def test_generic_refined_9x9_reaches_guard_and_solver_without_autoplay_repair() -> None:
    """Consume final generic vision geometry exactly once through existing ports."""

    refined_input = _geometry_board_input(9, 9, 9)
    runner, _, _, _, mouse, _, _, analyzer, solver = _runner(
        (_detection(CatsScreenState.BOARD),),
        board_inputs=(refined_input,),
    )

    summary = runner.run()

    assert analyzer.calls == 1
    assert len(solver.calls) == 1
    assert solver.calls[0][1] is refined_input
    assert summary.solved_levels == 1
    assert len(mouse.clicks) == 18
