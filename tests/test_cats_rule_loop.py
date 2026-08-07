"""Tests for deterministic fixed-point execution of current Cats rules."""

from collections.abc import Callable

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import apply_cats_rules_until_stalled
from logicforge.plugins.cats.rule_loop import DEFAULT_CATS_RULES
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)


class _ScriptedRule:
    """Return a finite outcome script while recording deterministic call order."""

    __slots__ = ("_calls", "_name", "_next_outcome", "_on_true", "_outcomes")

    def __init__(
        self,
        name: str,
        outcomes: tuple[bool, ...],
        calls: list[str],
        *,
        on_true: Callable[[Board], None] | None = None,
    ) -> None:
        """Store an immutable result script and an optional real mutation hook."""

        self._name = name
        self._outcomes = outcomes
        self._calls = calls
        self._on_true = on_true
        self._next_outcome = 0

    def apply(self, board: Board) -> bool:
        """Record the call and return the next result, then False indefinitely."""

        self._calls.append(self._name)
        if self._next_outcome >= len(self._outcomes):
            return False
        outcome = self._outcomes[self._next_outcome]
        self._next_outcome += 1
        if outcome and self._on_true is not None:
            self._on_true(board)
        return outcome


class _FailingRule:
    """Raise the board contradiction used to verify fail-fast propagation."""

    __slots__ = ("_calls", "_name")

    def __init__(self, name: str, calls: list[str]) -> None:
        """Retain only shared diagnostic call history, never the Board itself."""

        self._name = name
        self._calls = calls

    def apply(self, board: Board) -> bool:
        """Record one invocation and surface a deterministic contradiction."""

        del board
        self._calls.append(self._name)
        raise BoardStateError("synthetic Cats rule contradiction")


def _board_from_values(values: tuple[tuple[str, ...], ...]) -> Board:
    """Build Board transport, then configure terminal values as test-only state."""

    rows = len(values)
    columns = len(values[0])
    base_matrix = tuple(tuple("C0" for _ in row) for row in values)
    observations = tuple(
        ColorObservation(
            row=row,
            column=column,
            color_id="C0",
            confidence=1.0,
            representative_lab=(120.0, 130.0, 140.0),
        )
        for row in range(rows)
        for column in range(columns)
    )
    result = ColorDetectionResult(
        observations=observations,
        color_count=1,
        color_matrix=base_matrix,
        mean_confidence=1.0,
        diagnostics=ColorDetectionDiagnostics(
            rows=rows,
            columns=columns,
            cluster_distance_threshold=18.0,
            sample_pixel_counts=(100,) * (rows * columns),
            within_cell_spreads=(1.0,) * (rows * columns),
            cluster_centers_lab=((120.0, 130.0, 140.0),),
            minimum_intercluster_distance=None,
        ),
    )
    board = Board(result)
    for row, row_values in enumerate(values):
        for column, value in enumerate(row_values):
            board.cells[row][column] = value
    return board


def _values(board: Board) -> tuple[tuple[str, ...], ...]:
    """Capture a test-only value view for no-mutation assertions."""

    return tuple(tuple(row) for row in board.cells)


def _simple_board() -> Board:
    """Return a small board suitable for behavior-only fake rule tests."""

    return _board_from_values((("C0", "C1"), ("C2", "C3")))


def test_default_rule_order_contains_all_seven_cats_rules() -> None:
    """Keep the explicit Cats priority independent from import or registry order."""

    assert tuple(type(rule).__name__ for rule in DEFAULT_CATS_RULES) == (
        "SingleRemainingColorCellRule",
        "SingleRemainingLineCellRule",
        "MonochromaticLineColorExclusionRule",
        "ColorSubsetConfinedToLinesRule",
        "AdjacentColorPairExclusionRule",
        "ColorConfinedToLineRule",
        "ImpossibleCatCandidateRule",
    )


def test_single_remaining_line_precedes_monochromatic_exclusion() -> None:
    """Place a forced line cat before making the weaker color exclusion."""

    rule_names = tuple(type(rule).__name__ for rule in DEFAULT_CATS_RULES)

    assert rule_names.index("SingleRemainingLineCellRule") < rule_names.index(
        "MonochromaticLineColorExclusionRule"
    )


def test_color_subset_rule_has_required_priority() -> None:
    """Place the general subset deduction between line and local pair logic."""

    rule_names = tuple(type(rule).__name__ for rule in DEFAULT_CATS_RULES)
    subset_index = rule_names.index("ColorSubsetConfinedToLinesRule")

    assert rule_names.index("MonochromaticLineColorExclusionRule") < subset_index
    assert subset_index < rule_names.index("AdjacentColorPairExclusionRule")
    assert subset_index < rule_names.index("ColorConfinedToLineRule")


def test_impossible_candidate_rule_is_last_and_after_cheaper_rules() -> None:
    """Keep one-step lookahead behind every direct deterministic deduction."""

    rule_names = tuple(type(rule).__name__ for rule in DEFAULT_CATS_RULES)
    lookahead_index = rule_names.index("ImpossibleCatCandidateRule")

    assert lookahead_index == len(rule_names) - 1
    assert rule_names.index("ColorSubsetConfinedToLinesRule") < lookahead_index
    assert rule_names.index("AdjacentColorPairExclusionRule") < lookahead_index
    assert rule_names.index("ColorConfinedToLineRule") < lookahead_index


def test_last_lookahead_success_restarts_from_first_rule() -> None:
    """Restart at the highest priority after the seventh rule blocks one cell."""

    calls: list[str] = []
    rule_names = (
        "SingleRemainingColorCellRule",
        "SingleRemainingLineCellRule",
        "MonochromaticLineColorExclusionRule",
        "ColorSubsetConfinedToLinesRule",
        "AdjacentColorPairExclusionRule",
        "ColorConfinedToLineRule",
    )
    rules = (
        *(_ScriptedRule(name, (False, False), calls) for name in rule_names),
        _ScriptedRule(
            "ImpossibleCatCandidateRule",
            (True, False),
            calls,
        ),
    )

    successful_applications = apply_cats_rules_until_stalled(
        _simple_board(),
        rules=rules,
    )

    assert successful_applications == 1
    assert calls[:7] == [*rule_names, "ImpossibleCatCandidateRule"]
    assert calls[7] == "SingleRemainingColorCellRule"


def test_all_false_rules_return_zero() -> None:
    """Count no successful application after one completely unchanged pass."""

    calls: list[str] = []
    rules = tuple(
        _ScriptedRule(name, (False,), calls) for name in ("first", "second", "third")
    )

    assert apply_cats_rules_until_stalled(_simple_board(), rules=rules) == 0


def test_all_false_rules_leave_board_unchanged() -> None:
    """Do not mutate Board when every rule reports no deduction."""

    board = _simple_board()
    expected = _values(board)
    calls: list[str] = []
    rules = tuple(
        _ScriptedRule(name, (False,), calls) for name in ("first", "second", "third")
    )

    apply_cats_rules_until_stalled(board, rules=rules)

    assert _values(board) == expected


def test_empty_rule_tuple_returns_zero_immediately() -> None:
    """Treat an explicitly empty ordered rule set as an already stable pass."""

    assert apply_cats_rules_until_stalled(_simple_board(), rules=()) == 0


def test_first_true_skips_later_rules_in_same_pass() -> None:
    """Break the current for-loop immediately after the first rule succeeds."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (True, False), calls),
        _ScriptedRule("second", (False,), calls),
        _ScriptedRule("third", (False,), calls),
    )

    apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls[:2] == ["first", "first"]


def test_first_true_restarts_from_first_rule() -> None:
    """Begin the next while iteration at rule index zero after first succeeds."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (True, False), calls),
        _ScriptedRule("second", (False,), calls),
        _ScriptedRule("third", (False,), calls),
    )

    apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls == ["first", "first", "second", "third"]


def test_second_true_skips_third_rule_in_same_pass() -> None:
    """Do not invoke the third rule until evaluation has restarted at the first."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (False, False), calls),
        _ScriptedRule("second", (True, False), calls),
        _ScriptedRule("third", (False,), calls),
    )

    apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls[:3] == ["first", "second", "first"]


def test_second_true_restarts_next_pass_from_first_rule() -> None:
    """Re-evaluate higher-priority logic after a second-rule mutation."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (False, False), calls),
        _ScriptedRule("second", (True, False), calls),
        _ScriptedRule("third", (False,), calls),
    )

    apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls == ["first", "second", "first", "second", "third"]


def test_third_true_restarts_next_pass_from_first_rule() -> None:
    """Restart from rule zero even when the final rule produced the mutation."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (False, False), calls),
        _ScriptedRule("second", (False, False), calls),
        _ScriptedRule("third", (True, False), calls),
    )

    apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls == ["first", "second", "third", "first", "second", "third"]


def test_loop_stops_only_after_complete_false_pass() -> None:
    """Require every ordered rule to report False before returning."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (True, False), calls),
        _ScriptedRule("second", (False,), calls),
        _ScriptedRule("third", (False,), calls),
    )

    successful_applications = apply_cats_rules_until_stalled(
        _simple_board(),
        rules=rules,
    )

    assert successful_applications == 1
    assert calls[-3:] == ["first", "second", "third"]


def test_result_counts_applications_not_changed_cells() -> None:
    """Count one successful rule even when that rule finalizes three cells."""

    board = _simple_board()
    calls: list[str] = []

    def block_three_cells(target: Board) -> None:
        """Perform the fake rule's three real mutations for count semantics."""

        target.set_blocked(0, 0)
        target.set_blocked(0, 1)
        target.set_blocked(1, 0)

    rules = (_ScriptedRule("multi", (True, False), calls, on_true=block_three_cells),)

    successful_applications = apply_cats_rules_until_stalled(board, rules=rules)

    assert successful_applications == 1
    assert sum(value == "X" for row in board.cells for value in row) == 3


def test_multiple_successful_applications_return_exact_count() -> None:
    """Accumulate one count for each successful call across restarted passes."""

    calls: list[str] = []
    rules = (
        _ScriptedRule("first", (True, False, False, False), calls),
        _ScriptedRule("second", (True, False, False), calls),
        _ScriptedRule("third", (True, False), calls),
    )

    assert apply_cats_rules_until_stalled(_simple_board(), rules=rules) == 3


def test_board_state_error_propagates_to_caller() -> None:
    """Expose contradictions without translating or suppressing their type."""

    calls: list[str] = []

    with pytest.raises(BoardStateError, match="synthetic Cats rule contradiction"):
        apply_cats_rules_until_stalled(
            _simple_board(),
            rules=(_FailingRule("failing", calls),),
        )


def test_rules_after_board_state_error_are_not_called() -> None:
    """Abort evaluation immediately instead of continuing after contradiction."""

    calls: list[str] = []
    rules = (
        _FailingRule("failing", calls),
        _ScriptedRule("later", (False,), calls),
    )

    with pytest.raises(BoardStateError):
        apply_cats_rules_until_stalled(_simple_board(), rules=rules)

    assert calls == ["failing"]


def test_real_rules_restart_singleton_after_adjacent_pair_exclusion() -> None:
    """Let the C0 pair create a C1 singleton consumed on the restarted pass."""

    board = _board_from_values(
        (
            ("C2", "C0", "C3", "C4"),
            ("C1", "C0", "C5", "C6"),
            ("C2", "C7", "C1", "C3"),
            ("C4", "C7", "C5", "C6"),
        )
    )

    successful_applications = apply_cats_rules_until_stalled(board)

    assert successful_applications >= 2
    assert board.get(2, 2) == "K"
    assert tuple(
        board.get(row, column) for row, column in ((0, 0), (0, 2), (1, 0), (1, 2))
    ) == (
        "X",
        "X",
        "X",
        "X",
    )


def test_real_rules_place_single_remaining_line_cell_before_weaker_rules() -> None:
    """Let the second rule immediately place the sole possibility in row zero."""

    board = _board_from_values(
        (
            ("X", "X", "C0", "X"),
            ("C1", "C2", "C3", "C0"),
            ("C1", "C2", "C3", "C4"),
            ("C4", "C5", "C5", "C0"),
        )
    )

    successful_applications = apply_cats_rules_until_stalled(board)

    assert successful_applications >= 1
    assert board.get(0, 2) == "K"
    assert tuple(board.get(row, column) for row, column in ((1, 3), (3, 3))) == (
        "X",
        "X",
    )
    assert tuple(board.get(row, 2) for row in range(1, 4)) == ("X", "X", "X")


def test_real_rules_restart_singleton_after_color_subset_exclusion() -> None:
    """Let the C0/C1 column subset create a C4 singleton after loop restart."""

    board = _board_from_values(
        (
            ("C0", "C0", "C2", "C3"),
            ("C0", "C4", "C2", "C3"),
            ("C1", "C1", "C4", "C5"),
            ("C1", "C5", "C6", "C6"),
        )
    )

    successful_applications = apply_cats_rules_until_stalled(board)

    assert successful_applications >= 2
    assert board.get(1, 1) == "X"
    assert board.get(3, 1) == "X"
    assert board.get(2, 2) == "K"
