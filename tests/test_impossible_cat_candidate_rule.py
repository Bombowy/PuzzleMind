"""Deterministic tests for one-step impossible Cats candidate lookahead."""

import inspect
import re

import pytest

from logicforge.core import Board, BoardStateError
from logicforge.plugins.cats import ImpossibleCatCandidateRule
from logicforge.plugins.cats import impossible_cat_candidate_rule as rule_module
from logicforge.plugins.cats.board_actions import block_cell as real_block_cell
from logicforge.plugins.cats.rule_loop import DEFAULT_CATS_RULES
from logicforge.vision.color_detector import (
    ColorDetectionDiagnostics,
    ColorDetectionResult,
    ColorObservation,
)


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
    """Capture a test-only immutable view for no-mutation assertions."""

    return tuple(tuple(row) for row in board.cells)


def _safe_latin_board() -> Board:
    """Return a balanced board with no immediate failed candidate."""

    return _board_from_values(
        (
            ("C0", "C1", "C2", "C3", "C4"),
            ("C1", "C2", "C3", "C4", "C0"),
            ("C2", "C3", "C4", "C0", "C1"),
            ("C3", "C4", "C0", "C1", "C2"),
            ("C4", "C0", "C1", "C2", "C3"),
        )
    )


def _color_contradiction_board() -> Board:
    """Return the documented board where (0, 0) removes every C1 candidate."""

    return _board_from_values(
        (
            ("C0", "C1", "C1", "C2"),
            ("C3", "C1", "C4", "C2"),
            ("C0", "C3", "C4", "C5"),
            ("C5", "C6", "C6", "C2"),
        )
    )


def _row_contradiction_values() -> tuple[tuple[str, ...], ...]:
    """Return a board where hypothetical (0, 0) empties row one."""

    return (
        ("C0", "C1", "C2", "C3", "C4"),
        ("C5", "C6", "X", "X", "X"),
        ("C7", "C7", "C1", "C2", "C5"),
        ("C8", "C8", "C3", "C4", "C6"),
        ("C0", "C9", "C9", "C8", "C7"),
    )


def _transpose(
    values: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    """Transpose a rectangular test matrix without changing logical values."""

    return tuple(
        tuple(row[column] for row in values) for column in range(len(values[0]))
    )


def test_candidate_eliminating_another_color_is_blocked() -> None:
    """Reject (0, 0) because its direct plan removes every current C1."""

    board = _color_contradiction_board()

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_hypothetical_color_analysis_mutates_only_proven_target() -> None:
    """Leave every non-target cell unchanged during and after failed lookahead."""

    board = _color_contradiction_board()
    expected = _values(board)

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"
    assert all(
        board.get(row, column) == expected[row][column]
        for row, values in enumerate(expected)
        for column in range(len(values))
        if (row, column) != (0, 0)
    )


def test_apply_returns_true_after_real_candidate_exclusion() -> None:
    """Report the one actual C<n> to X mutation performed by the rule."""

    assert ImpossibleCatCandidateRule().apply(_color_contradiction_board()) is True


def test_one_apply_blocks_at_most_one_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop immediately after the first row-major impossible candidate."""

    board = _safe_latin_board()

    def two_candidates_are_impossible(
        target_board: Board,
        row: int,
        column: int,
        analysis: object,
    ) -> bool:
        """Mark two coordinates impossible without mutating the supplied board."""

        del target_board, analysis
        return (row, column) in {(0, 1), (1, 0)}

    monkeypatch.setattr(
        rule_module,
        "_candidate_causes_contradiction",
        two_candidates_are_impossible,
    )

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(1, 0) == "C1"
    assert sum(value == "X" for row in board.cells for value in row) == 1


def test_target_color_is_not_checked_as_eliminated() -> None:
    """Treat the hypothetical cat as resolving its own singleton color."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2", "C3"),
            ("C1", "C2", "C3", "C4"),
            ("C2", "C3", "C4", "C1"),
            ("C3", "C4", "C1", "C2"),
        )
    )
    analysis = rule_module._analyze_board(board)
    rule_module._validate_existing_cat_state(analysis)
    expected = _values(board)

    assert not rule_module._candidate_causes_contradiction(board, 0, 0, analysis)
    assert _values(board) == expected


def test_candidate_emptying_another_row_is_blocked() -> None:
    """Reject (0, 0) when row one would retain neither K nor C<n>."""

    board = _board_from_values(_row_contradiction_values())

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_candidate_emptying_another_column_is_blocked() -> None:
    """Apply the same viability check to columns on the transposed board."""

    board = _board_from_values(_transpose(_row_contradiction_values()))

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_candidate_conflicting_with_cat_in_same_row_is_blocked() -> None:
    """Reject a candidate whose row exclusion plan contains an existing K."""

    board = _safe_latin_board()
    board.cells[0][4] = "K"

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_candidate_conflicting_with_cat_in_same_column_is_blocked() -> None:
    """Reject a candidate whose column exclusion plan contains an existing K."""

    board = _safe_latin_board()
    board.cells[4][0] = "K"

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_candidate_touching_cat_diagonally_is_blocked() -> None:
    """Reject a candidate whose eight-neighbor plan contains a diagonal K."""

    board = _safe_latin_board()
    board.cells[1][1] = "K"

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 0) == "X"


def test_cat_outside_hypothetical_plan_does_not_reject_candidate() -> None:
    """Do not treat a distant, non-conflicting K as evidence against the target."""

    board = _safe_latin_board()
    board.cells[2][2] = "K"
    analysis = rule_module._analyze_board(board)
    rule_module._validate_existing_cat_state(analysis)
    expected = _values(board)

    assert not rule_module._candidate_causes_contradiction(board, 0, 0, analysis)
    assert _values(board) == expected


def test_existing_x_are_not_possibilities_or_new_mutations() -> None:
    """Ignore existing X in row viability and preserve them as terminal state."""

    board = _board_from_values(_row_contradiction_values())

    ImpossibleCatCandidateRule().apply(board)

    assert tuple(board.get(1, column) for column in range(2, 5)) == ("X", "X", "X")
    assert sum(value == "X" for row in board.cells for value in row) == 4


def test_safe_latin_board_returns_false() -> None:
    """Make no deduction when every immediate hypothetical state remains viable."""

    assert ImpossibleCatCandidateRule().apply(_safe_latin_board()) is False


def test_false_result_leaves_board_unchanged() -> None:
    """Preserve the complete matrix when no immediate contradiction is found."""

    board = _safe_latin_board()
    expected = _values(board)

    assert ImpossibleCatCandidateRule().apply(board) is False
    assert _values(board) == expected


def test_only_current_unknown_cells_are_analyzed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never ask hypothetical planning to evaluate existing K or X cells."""

    board = _board_from_values(
        (
            ("K", "X", "C0"),
            ("X", "C1", "C2"),
            ("C3", "C4", "C5"),
        )
    )
    examined: list[tuple[int, int]] = []

    def empty_hypothetical_plan(
        target_board: Board,
        row: int,
        column: int,
    ) -> tuple[tuple[int, int], ...]:
        """Record planner targets while keeping every candidate viable."""

        assert target_board is board
        examined.append((row, column))
        return ()

    monkeypatch.setattr(
        rule_module,
        "collect_cat_exclusion_coordinates",
        empty_hypothetical_plan,
    )

    assert ImpossibleCatCandidateRule().apply(board) is False
    assert examined == [
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_candidates_are_examined_in_row_major_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use coordinate order rather than color identifier order."""

    board = _safe_latin_board()
    examined: list[tuple[int, int]] = []

    def second_coordinate_is_impossible(
        target_board: Board,
        row: int,
        column: int,
        analysis: object,
    ) -> bool:
        """Record deterministic order and stop on coordinate (0, 1)."""

        del target_board, analysis
        examined.append((row, column))
        return (row, column) == (0, 1)

    monkeypatch.setattr(
        rule_module,
        "_candidate_causes_contradiction",
        second_coordinate_is_impossible,
    )

    ImpossibleCatCandidateRule().apply(board)

    assert examined == [(0, 0), (0, 1)]


def test_first_of_two_impossible_candidates_is_blocked_row_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the later impossible candidate for another apply call."""

    board = _safe_latin_board()

    def selected_candidates_are_impossible(
        target_board: Board,
        row: int,
        column: int,
        analysis: object,
    ) -> bool:
        """Expose two independent impossible coordinates."""

        del target_board, analysis
        return (row, column) in {(0, 1), (1, 0)}

    monkeypatch.setattr(
        rule_module,
        "_candidate_causes_contradiction",
        selected_candidates_are_impossible,
    )

    ImpossibleCatCandidateRule().apply(board)

    assert board.get(0, 1) == "X"
    assert board.get(1, 0) == "C1"


def test_second_apply_can_block_next_impossible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild analysis and remove the next row-major failed candidate later."""

    board = _safe_latin_board()

    def selected_candidates_are_impossible(
        target_board: Board,
        row: int,
        column: int,
        analysis: object,
    ) -> bool:
        """Expose the same two candidates on each fresh board scan."""

        del target_board, analysis
        return (row, column) in {(0, 1), (1, 0)}

    monkeypatch.setattr(
        rule_module,
        "_candidate_causes_contradiction",
        selected_candidates_are_impossible,
    )

    assert ImpossibleCatCandidateRule().apply(board) is True
    assert ImpossibleCatCandidateRule().apply(board) is True
    assert board.get(0, 1) == "X"
    assert board.get(1, 0) == "X"


def test_invalid_value_raises_board_state_error() -> None:
    """Reject unsupported state during the initial read-only scan."""

    board = _safe_latin_board()
    board.cells[2][3] = "INVALID"

    with pytest.raises(BoardStateError):
        ImpossibleCatCandidateRule().apply(board)


def test_invalid_value_error_contains_value_and_coordinates() -> None:
    """Provide actionable diagnostics for corrupted board state."""

    board = _safe_latin_board()
    board.cells[2][3] = "INVALID"

    with pytest.raises(BoardStateError, match=r"INVALID.*\(2, 3\)"):
        ImpossibleCatCandidateRule().apply(board)


def test_invalid_value_error_does_not_mutate_board() -> None:
    """Fail before testing or blocking any candidate."""

    board = _safe_latin_board()
    board.cells[2][3] = "INVALID"
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ImpossibleCatCandidateRule().apply(board)

    assert _values(board) == expected


def test_preexisting_empty_row_without_cat_raises() -> None:
    """Reject a row that already has neither K nor C<n>."""

    board = _safe_latin_board()
    board.cells[1] = ["X"] * 5

    with pytest.raises(BoardStateError, match="Row 1"):
        ImpossibleCatCandidateRule().apply(board)


def test_preexisting_empty_column_without_cat_raises() -> None:
    """Reject a column that already has neither K nor C<n>."""

    board = _safe_latin_board()
    for row in range(5):
        board.cells[row][2] = "X"

    with pytest.raises(BoardStateError, match="Column 2"):
        ImpossibleCatCandidateRule().apply(board)


def test_preexisting_contradiction_does_not_block_random_candidate() -> None:
    """Report invalid input instead of using it to justify an arbitrary X."""

    board = _safe_latin_board()
    board.cells[1] = ["X"] * 5
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ImpossibleCatCandidateRule().apply(board)

    assert _values(board) == expected


def test_two_cats_in_same_row_raise() -> None:
    """Validate the existing one-cat-per-row invariant before lookahead."""

    board = _safe_latin_board()
    board.cells[0][0] = "K"
    board.cells[0][4] = "K"

    with pytest.raises(BoardStateError, match="Row 0"):
        ImpossibleCatCandidateRule().apply(board)


def test_two_cats_in_same_column_raise() -> None:
    """Validate the existing one-cat-per-column invariant before lookahead."""

    board = _safe_latin_board()
    board.cells[0][0] = "K"
    board.cells[4][0] = "K"

    with pytest.raises(BoardStateError, match="Column 0"):
        ImpossibleCatCandidateRule().apply(board)


def test_touching_existing_cats_raise() -> None:
    """Reject orthogonally or diagonally adjacent confirmed cats."""

    board = _safe_latin_board()
    board.cells[0][0] = "K"
    board.cells[1][1] = "K"

    with pytest.raises(BoardStateError, match="touch"):
        ImpossibleCatCandidateRule().apply(board)


def test_existing_state_error_leaves_board_unchanged() -> None:
    """Keep all cells intact when pre-lookahead validation fails."""

    board = _safe_latin_board()
    board.cells[0][0] = "K"
    board.cells[1][1] = "K"
    expected = _values(board)

    with pytest.raises(BoardStateError):
        ImpossibleCatCandidateRule().apply(board)

    assert _values(board) == expected


def test_rule_supports_rectangular_three_by_five_board() -> None:
    """Use dynamic dimensions when a same-row K disproves the first candidate."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2", "C3", "K"),
            ("C1", "C2", "C3", "C4", "C0"),
            ("C2", "C3", "C4", "C0", "C1"),
        )
    )

    assert ImpossibleCatCandidateRule().apply(board) is True
    assert board.get(0, 0) == "X"


def test_rule_supports_rectangular_five_by_three_board() -> None:
    """Use dynamic dimensions when a same-column K disproves the first candidate."""

    board = _board_from_values(
        (
            ("C0", "C1", "C2"),
            ("C1", "C2", "C0"),
            ("C2", "C0", "C1"),
            ("C0", "C1", "C2"),
            ("K", "C2", "C0"),
        )
    )

    assert ImpossibleCatCandidateRule().apply(board) is True
    assert board.get(0, 0) == "X"


def test_rule_does_not_copy_board() -> None:
    """Keep lookahead coordinate-based without Board cloning or snapshots."""

    source = inspect.getsource(rule_module)

    assert "copy(" not in source
    assert "deepcopy" not in source


def test_real_mutation_uses_only_block_cell(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delegate the sole proven exclusion to the shared Cats action."""

    board = _color_contradiction_board()
    calls: list[tuple[int, int]] = []

    def recording_block_cell(target: Board, row: int, column: int) -> bool:
        """Record and forward the single authorized real mutation."""

        calls.append((row, column))
        return real_block_cell(target, row, column)

    monkeypatch.setattr(rule_module, "block_cell", recording_block_cell)

    assert ImpossibleCatCandidateRule().apply(board) is True
    assert calls == [(0, 0)]


def test_rule_does_not_call_place_cat() -> None:
    """Keep failed-candidate handling separate from real cat placement."""

    source = inspect.getsource(rule_module)

    assert "place_cat(" not in source


def test_rule_does_not_call_board_set_cat() -> None:
    """Never create a temporary or real K during one-step analysis."""

    source = inspect.getsource(rule_module)

    assert ".set_cat(" not in source


def test_rule_does_not_call_board_set_blocked() -> None:
    """Route the one real X through block_cell instead of Board directly."""

    source = inspect.getsource(rule_module)

    assert ".set_blocked(" not in source


def test_rule_does_not_assign_board_cells() -> None:
    """Read the sole mutable matrix without assigning to it directly."""

    source = inspect.getsource(rule_module)

    assert re.search(r"board\.cells\[[^\n]*\]\s*=", source) is None


def test_hypothetical_analysis_leaves_no_temporary_cat_or_x() -> None:
    """Observe only the final proven X and no simulation residue anywhere else."""

    board = _color_contradiction_board()
    initial = _values(board)

    ImpossibleCatCandidateRule().apply(board)

    assert all(value != "K" for row in board.cells for value in row)
    assert sum(value == "X" for row in board.cells for value in row) == 1
    assert all(
        board.get(row, column) == initial[row][column]
        for row, values in enumerate(initial)
        for column in range(len(values))
        if (row, column) != (0, 0)
    )


def test_one_step_lookahead_runs_after_all_six_cheaper_rules() -> None:
    """Prove the supplied 5x5 case is reached only by failed-candidate logic."""

    board = _board_from_values(
        (
            ("C0", "C3", "C1", "C2", "C3"),
            ("C3", "C1", "C2", "C3", "C4"),
            ("C1", "C2", "C3", "C4", "C2"),
            ("C2", "C3", "C4", "C2", "C3"),
            ("C3", "C4", "C2", "C3", "C0"),
        )
    )

    assert all(rule.apply(board) is False for rule in DEFAULT_CATS_RULES[:-1])
    assert ImpossibleCatCandidateRule().apply(board) is True
    assert board.get(0, 0) == "X"
