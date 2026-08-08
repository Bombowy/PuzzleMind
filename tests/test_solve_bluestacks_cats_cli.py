"""Cats solve cli tests."""

import pytest
from scripts import solve_bluestacks_cats as solve_script

from cats_solve_test_support import (
    _CaptureService,
    _configure_pipeline,
    _FakeMouseController,
    _grid_detection,
    _GridDetector,
    _offset_window,
    _set_complete_result,
)
from logicforge.application.cats import solving as cats_solving
from logicforge.automation.mouse import MouseController, ScreenPoint
from logicforge.core import Board
from logicforge.vision.board_detector import (
    BoardDetection,
)
from logicforge.vision.color_detector import (
    ColorDetectionResult,
)
from logicforge.vision.grid_detector import (
    GridDetection,
)
from logicforge.vision.screenshot import Screenshot
from logicforge.vision.window_capture import (
    WindowInfo,
)


def test_main_creates_one_board_and_calls_rule_loop_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the sole newly constructed Board to exactly one rule-loop invocation."""

    _configure_pipeline(monkeypatch)
    created_boards: list[Board] = []
    solved_boards: list[Board] = []

    def create_board(result: ColorDetectionResult) -> Board:
        board = Board(result)
        created_boards.append(board)
        return board

    def solve(board: Board) -> int:
        solved_boards.append(board)
        return _set_complete_result(board)

    monkeypatch.setattr(cats_solving, "Board", create_board)
    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", solve)

    assert solve_script.main() == 0
    assert len(created_boards) == 1
    assert solved_boards == created_boards


def test_main_builds_click_plan_after_deduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map final K state only after the single Cats rule-loop invocation."""

    _configure_pipeline(monkeypatch)
    events: list[str] = []

    def solve(board: Board) -> int:
        """Record deduction and create one confirmed cat for the plan."""

        events.append("deduction")
        return _set_complete_result(board)

    def build_plan(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Record that mapping observes the already-mutated Board."""

        del grid, window
        events.append("click-plan")
        assert board.is_cat(0, 0)
        return ()

    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", solve)
    monkeypatch.setattr(cats_solving, "build_cat_click_plan", build_plan)

    assert solve_script.main() == 0
    assert events == ["deduction", "click-plan"]


def test_main_reuses_same_window_and_grid_for_click_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass pipeline geometry objects directly to mapping without re-detection."""

    _configure_pipeline(monkeypatch)
    located_windows: list[WindowInfo] = []
    detected_grids: list[GridDetection] = []

    def locate_window(service: _CaptureService) -> WindowInfo:
        """Return and retain one unique WindowInfo instance."""

        del service
        window = _offset_window(x=300, y=200)
        located_windows.append(window)
        return window

    def detect_grid(
        detector: _GridDetector,
        screenshot: Screenshot,
        board: BoardDetection,
    ) -> GridDetection:
        """Return and retain one unique GridDetection instance."""

        del detector, screenshot, board
        grid = _grid_detection()
        detected_grids.append(grid)
        return grid

    def inspect_plan_inputs(
        board: Board,
        grid: GridDetection,
        window: WindowInfo,
    ) -> tuple[solve_script.CatClickTarget, ...]:
        """Verify identity of the already-produced pipeline geometry."""

        del board
        assert window is located_windows[0]
        assert grid is detected_grids[0]
        return ()

    monkeypatch.setattr(_CaptureService, "locate_window", locate_window)
    monkeypatch.setattr(_GridDetector, "detect", detect_grid)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )
    monkeypatch.setattr(cats_solving, "build_cat_click_plan", inspect_plan_inputs)

    assert solve_script.main() == 0
    assert len(located_windows) == 1
    assert len(detected_grids) == 1


def test_main_prints_dry_run_click_plan(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Include mapped K centers in successful script diagnostics."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    assert solve_script.main() == 0

    output = capsys.readouterr().out
    assert "Planned cat click targets: 2" in output
    assert "CLICK: row=0, column=0, screenshot=(20, 30), desktop=(20, 30)" in output


def test_cli_defaults_to_dry_run_and_ten_millisecond_delay() -> None:
    """Keep execution opt-in while retaining the operational delay default."""

    arguments = solve_script.parse_arguments(())

    assert arguments.execute_clicks is False
    assert arguments.click_delay_ms == 10


@pytest.mark.parametrize("delay_ms", [0, 25])
def test_cli_accepts_non_negative_delay(delay_ms: int) -> None:
    """Accept zero and arbitrary positive integer millisecond delays."""

    arguments = solve_script.parse_arguments(("--click-delay-ms", str(delay_ms)))

    assert arguments.click_delay_ms == delay_ms


def test_cli_rejects_negative_delay() -> None:
    """Use argparse's standard non-zero parse failure for an unsafe value."""

    with pytest.raises(SystemExit) as error_info:
        solve_script.parse_arguments(("--click-delay-ms", "-1"))

    assert error_info.value.code == 2


def test_dry_run_does_not_construct_or_call_mouse_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the previous no-input behavior unless execution is explicit."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )

    def reject_controller_creation() -> _FakeMouseController:
        raise AssertionError("dry-run must not create a mouse controller")

    monkeypatch.setattr(
        solve_script,
        "Win32MouseController",
        reject_controller_creation,
    )

    assert solve_script.main(()) == 0


def test_execute_clicks_runs_complete_plan_with_one_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute every mapped K twice through exactly one adapter instance."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )
    controller = _FakeMouseController()
    controller_creations: list[None] = []

    def create_controller() -> _FakeMouseController:
        controller_creations.append(None)
        return controller

    monkeypatch.setattr(solve_script, "Win32MouseController", create_controller)

    assert solve_script.main(("--execute-clicks", "--click-delay-ms", "0")) == 0
    assert controller_creations == [None]
    assert [point for point, _ in controller.clicks] == [
        ScreenPoint(20, 30),
        ScreenPoint(20, 30),
        ScreenPoint(40, 50),
        ScreenPoint(40, 50),
    ]


@pytest.mark.parametrize(
    ("delay_ms", "expected_seconds"),
    [(10, 0.01), (0, 0.0), (25, 0.025)],
)
def test_main_converts_cli_delay_to_seconds(
    monkeypatch: pytest.MonkeyPatch,
    delay_ms: int,
    expected_seconds: float,
) -> None:
    """Convert integer milliseconds exactly once at the execution boundary."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )
    received_delays: list[float] = []

    def record_execution(
        targets: tuple[solve_script.CatClickTarget, ...],
        mouse_controller: MouseController,
        *,
        click_delay_seconds: float = 0.01,
    ) -> int:
        del mouse_controller
        received_delays.append(click_delay_seconds)
        return len(targets)

    monkeypatch.setattr(solve_script, "Win32MouseController", _FakeMouseController)
    monkeypatch.setattr(solve_script, "execute_cat_click_plan", record_execution)

    arguments = ["--execute-clicks"]
    if delay_ms != 10:
        arguments.extend(("--click-delay-ms", str(delay_ms)))

    assert solve_script.main(arguments) == 0
    assert received_delays == [expected_seconds]


def test_execute_empty_plan_does_not_construct_native_adapter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return success and print zero without resolving any native dependency."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", lambda board: 0)

    def reject_controller_creation() -> _FakeMouseController:
        raise AssertionError("empty plan must not create a mouse controller")

    monkeypatch.setattr(
        solve_script,
        "Win32MouseController",
        reject_controller_creation,
    )

    assert solve_script.main(("--execute-clicks",)) == 0
    assert "Executed cat double-click targets: 0" in capsys.readouterr().out


def test_execute_success_prints_target_click_and_delay_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report targets, low-level clicks, and configured delay after execution."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )
    monkeypatch.setattr(solve_script, "Win32MouseController", _FakeMouseController)

    assert solve_script.main(("--execute-clicks", "--click-delay-ms", "0")) == 0

    output = capsys.readouterr().out
    assert "Executed cat double-click targets: 2" in output
    assert "Low-level left clicks emitted: 4" in output
    assert "Click delay: 0 ms" in output


def test_execute_success_prints_default_ten_millisecond_delay(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Make the default operational pause explicit in successful diagnostics."""

    _configure_pipeline(monkeypatch)
    monkeypatch.setattr(
        cats_solving,
        "apply_cats_rules_until_stalled",
        _set_complete_result,
    )
    monkeypatch.setattr(solve_script, "Win32MouseController", _FakeMouseController)

    def execute_without_real_sleep(
        targets: tuple[solve_script.CatClickTarget, ...],
        mouse_controller: MouseController,
        *,
        click_delay_seconds: float = 0.01,
    ) -> int:
        del mouse_controller
        assert click_delay_seconds == 0.01
        return len(targets)

    monkeypatch.setattr(
        solve_script,
        "execute_cat_click_plan",
        execute_without_real_sleep,
    )

    assert solve_script.main(("--execute-clicks",)) == 0
    assert "Click delay: 10 ms" in capsys.readouterr().out


def test_execute_mode_captures_once_and_solves_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add input emission without repeating capture, Board, or rule-loop work."""

    _configure_pipeline(monkeypatch)
    capture_calls: list[WindowInfo] = []
    solve_calls: list[Board] = []
    board_creations: list[Board] = []

    def capture_once(
        service: _CaptureService,
        window: WindowInfo,
        *,
        debug: bool = False,
    ) -> Screenshot:
        del service, debug
        capture_calls.append(window)
        return _CaptureService.screenshot

    def create_board(result: ColorDetectionResult) -> Board:
        board = Board(result)
        board_creations.append(board)
        return board

    def solve_once(board: Board) -> int:
        solve_calls.append(board)
        return _set_complete_result(board)

    monkeypatch.setattr(_CaptureService, "capture_window", capture_once)
    monkeypatch.setattr(cats_solving, "Board", create_board)
    monkeypatch.setattr(cats_solving, "apply_cats_rules_until_stalled", solve_once)
    monkeypatch.setattr(solve_script, "Win32MouseController", _FakeMouseController)

    assert solve_script.main(("--execute-clicks", "--click-delay-ms", "0")) == 0
    assert len(capture_calls) == 1
    assert len(board_creations) == 1
    assert solve_calls == board_creations
