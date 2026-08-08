"""Cats solve cli_errors tests."""

from inspect import getsource

import pytest
from scripts import solve_bluestacks_cats as solve_script

from cats_solve_test_support import (
    _board_error,
    _color_error,
    _configure_pipeline,
    _FakeMouseController,
    _grid_error,
    _set_complete_result,
    _set_stalled_result,
)
from logicforge.application.cats import solving as cats_solving
from logicforge.automation.mouse import MouseController
from logicforge.core import Board, BoardStateError
from logicforge.infrastructure.windows import MouseAutomationError
from logicforge.vision.grid_detector import (
    GridDetection,
)
from logicforge.vision.window_capture import (
    WindowCaptureError,
    WindowInfo,
)


@pytest.mark.parametrize(
    "execution_error",
    [
        MouseAutomationError("synthetic native failure"),
        solve_script.CatClickExecutionError("synthetic execution failure"),
    ],
)
def test_click_execution_errors_return_seven_and_actionable_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    execution_error: RuntimeError,
) -> None:
    """Translate only typed execution failures at the outer script boundary."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    def fail_execution(
        targets: tuple[solve_script.CatClickTarget, ...],
        mouse_controller: MouseController,
        *,
        click_delay_seconds: float = 0.01,
    ) -> int:
        del targets, mouse_controller, click_delay_seconds
        raise execution_error

    monkeypatch.setattr(solve_script, "Win32MouseController", _FakeMouseController)
    monkeypatch.setattr(solve_script, "execute_cat_click_plan", fail_execution)

    assert solve_script.main(("--execute-clicks",)) == 7
    error_output = capsys.readouterr().err
    assert "Cats click execution failed" in error_output
    assert str(execution_error) in error_output


def test_complete_main_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a fully finalized deduction result as successful execution."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    assert solve_script.main() == 0


def test_stalled_main_also_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed without clicks when the fallback proves ambiguity."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )

    assert solve_script.main() == 0


def test_output_contains_initial_immutable_color_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the detector result directly even after the Board has been mutated."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Initial board:\nC0 C1\nC2 C3" in output


def test_output_contains_final_mutable_board_matrix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print finalized K/X values from the same Board consumed by the rule loop."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Final board:\nK X\nX K" in output


def test_output_contains_successful_application_count(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report successful apply calls rather than inferring a cell-change count."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )

    solve_script.main()

    assert "Successful rule applications: 2" in capsys.readouterr().out


def test_output_contains_exact_search_status_and_effort(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose bounded fallback diagnostics without printing its branch tree."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_stalled_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "[solver] rules stalled; exact search started" in output
    assert "Exact search: AMBIGUOUS" in output
    assert "Search nodes: 4" in output
    assert "Propagation steps: 1" in output
    assert "Status: AMBIGUOUS" in output


def test_output_contains_all_cat_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print every K using stable zero-based row-major coordinates."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    solve_script.main()

    output = capsys.readouterr().out
    assert "Cats found: 2" in output
    assert "K: row=0, column=0" in output
    assert "K: row=1, column=1" in output


def test_window_capture_error_returns_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated capture-stage exit code and message."""

    _configure_pipeline(
        monkeypatch,
        capture_error=WindowCaptureError("synthetic capture failure"),
    )

    assert solve_script.main() == 1
    assert "synthetic capture failure" in capsys.readouterr().err


def test_board_detection_error_returns_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated board-stage exit code and message."""

    _configure_pipeline(monkeypatch, board_error=_board_error())

    assert solve_script.main() == 2
    assert "synthetic board failure" in capsys.readouterr().err


def test_grid_detection_error_returns_three(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated grid-stage exit code and message."""

    _configure_pipeline(monkeypatch, grid_error=_grid_error())

    assert solve_script.main() == 3
    assert "synthetic grid failure" in capsys.readouterr().err


def test_color_detection_error_returns_four(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preserve the dedicated color-stage exit code and message."""

    _configure_pipeline(monkeypatch, color_error=_color_error())

    assert solve_script.main() == 4
    assert "synthetic color failure" in capsys.readouterr().err


def test_board_state_error_from_rules_returns_five(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Translate only the expected deduction contradiction at the CLI boundary."""

    _configure_pipeline(monkeypatch)

    def fail_deduction(board: Board) -> int:
        raise BoardStateError(f"synthetic contradiction at {board.get(0, 0)}")

    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        fail_deduction,
    )

    assert solve_script.main() == 5
    assert "Cats deduction failed" in capsys.readouterr().err


def test_board_state_error_does_not_print_false_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Avoid presenting a contradiction as either successful terminal status."""

    _configure_pipeline(monkeypatch)

    def fail_deduction(board: Board) -> int:
        raise BoardStateError(f"synthetic contradiction at {board.get(0, 0)}")

    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        fail_deduction,
    )

    solve_script.main()
    captured = capsys.readouterr()

    assert "Status: COMPLETE" not in captured.out
    assert "Status: STALLED" not in captured.out


def test_click_plan_error_returns_six(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose geometry inconsistency through its dedicated script exit code."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    def fail_mapping(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Raise the expected local mapping error at the script boundary."""

        del board, grid, window
        raise solve_script.CatClickPlanError("synthetic click-plan failure")

    monkeypatch.setattr(cats_solving, "build_cat_click_plan", fail_mapping)

    assert solve_script.main() == 6


def test_click_plan_error_prints_actionable_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Describe the failed mapping without masking unexpected exceptions."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    def fail_mapping(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Raise the typed failure used by the CLI error branch."""

        del board, grid, window
        raise solve_script.CatClickPlanError("synthetic click-plan failure")

    monkeypatch.setattr(cats_solving, "build_cat_click_plan", fail_mapping)

    solve_script.main()

    error_output = capsys.readouterr().err
    assert "Cats click-plan mapping failed" in error_output
    assert "synthetic click-plan failure" in error_output


def test_script_uses_no_external_mouse_or_subprocess_technology() -> None:
    """Keep execution behind the controlled port and Win32 adapter only."""

    source = getsource(solve_script)

    for forbidden_text in (
        "pyautogui",
        "pynput",
        "subprocess",
    ):
        assert forbidden_text not in source


def test_script_does_not_emit_win32_events_directly() -> None:
    """Keep SetCursorPos and mouse_event confined to the infrastructure adapter."""

    source = getsource(solve_script)

    assert "SetCursorPos" not in source
    assert "mouse_event" not in source


def test_script_contains_no_prompt_countdown_or_target_limit() -> None:
    """Execute an opted-in full plan immediately without hidden interaction gates."""

    source = getsource(solve_script)

    for forbidden_text in ("\ninput(", "countdown", "max-clicks", "move-only"):
        assert forbidden_text not in source.casefold()
