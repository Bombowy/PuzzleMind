"""Deterministic exact constraint-search fallback for stalled Cats boards."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from logicforge.core.board import Board
from logicforge.plugins.cats.board_actions import block_cell, place_cat

type CellCoordinates = tuple[int, int]
type OriginalColorMatrix = tuple[tuple[str, ...], ...]


class CatsExactSearchStatus(StrEnum):
    """Classify a bounded proof of Cats solution cardinality."""

    UNIQUE = "UNIQUE"
    UNSAT = "UNSAT"
    AMBIGUOUS = "AMBIGUOUS"
    LIMIT_REACHED = "LIMIT_REACHED"


@dataclass(frozen=True, slots=True)
class CatsExactSearchResult:
    """Expose deterministic solution, effort, propagation, and MRV diagnostics."""

    status: CatsExactSearchStatus
    solution: tuple[CellCoordinates, ...] | None
    solutions_found: int
    search_nodes: int
    propagation_steps: int
    branch_groups: tuple[str, ...] = ()
    propagation_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.solutions_found <= 2:
            raise ValueError("solutions_found must be within 0..2.")
        if self.search_nodes < 0 or self.propagation_steps < 0:
            raise ValueError("Search effort counts must be non-negative.")
        if self.status is CatsExactSearchStatus.UNIQUE:
            if self.solution is None or self.solutions_found != 1:
                raise ValueError("UNIQUE requires exactly one retained solution.")
        elif self.solution is not None:
            raise ValueError("Only UNIQUE may expose a solution.")
        if self.status is CatsExactSearchStatus.UNSAT and self.solutions_found != 0:
            raise ValueError("UNSAT cannot contain a discovered solution.")
        if self.status is CatsExactSearchStatus.AMBIGUOUS and self.solutions_found != 2:
            raise ValueError("AMBIGUOUS requires two distinct solutions.")


class CatsExactSearchError(RuntimeError):
    """Report malformed programmer input or invalid unique-result application."""


@dataclass(frozen=True, slots=True)
class _SearchContext:
    row_count: int
    column_count: int
    color_ids: tuple[str, ...]
    original_colors: OriginalColorMatrix


@dataclass(frozen=True, slots=True)
class _SearchState:
    candidates: frozenset[CellCoordinates]
    selected_cats: frozenset[CellCoordinates]
    used_rows: frozenset[int]
    used_columns: frozenset[int]
    used_colors: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ConstraintGroup:
    group_type: str
    identifier: str | int
    candidates: tuple[CellCoordinates, ...]

    @property
    def label(self) -> str:
        return f"{self.group_type}:{self.identifier}"


@dataclass(slots=True)
class _SearchProgress:
    maximum_search_nodes: int
    search_nodes: int = 0
    propagation_steps: int = 0
    limit_reached: bool = False
    solutions: list[tuple[CellCoordinates, ...]] | None = None
    branch_groups: list[str] | None = None
    propagation_groups: list[str] | None = None

    def __post_init__(self) -> None:
        self.solutions = []
        self.branch_groups = []
        self.propagation_groups = []


def solve_cats_exact(
    board: Board,
    original_color_matrix: OriginalColorMatrix,
    *,
    maximum_solutions: int = 2,
    maximum_search_nodes: int = 250_000,
) -> CatsExactSearchResult:
    """Prove zero, one, or multiple Cats solutions without mutating ``board``."""

    if not isinstance(maximum_solutions, int) or isinstance(maximum_solutions, bool):
        raise CatsExactSearchError("maximum_solutions must be an integer.")
    if maximum_solutions != 2:
        raise CatsExactSearchError(
            "maximum_solutions must be exactly two to prove uniqueness without "
            "exploring unnecessary solutions."
        )
    if not isinstance(maximum_search_nodes, int) or isinstance(
        maximum_search_nodes, bool
    ):
        raise CatsExactSearchError("maximum_search_nodes must be an integer.")
    if maximum_search_nodes < 1:
        raise CatsExactSearchError("maximum_search_nodes must be positive.")

    context, fixed_cats, unresolved = _validate_input(
        board,
        original_color_matrix,
    )
    fixed_contradiction = _fixed_cat_contradiction(
        fixed_cats,
        context.original_colors,
    )
    if fixed_contradiction:
        return CatsExactSearchResult(
            status=CatsExactSearchStatus.UNSAT,
            solution=None,
            solutions_found=0,
            search_nodes=0,
            propagation_steps=0,
        )

    used_rows = frozenset(row for row, _ in fixed_cats)
    used_columns = frozenset(column for _, column in fixed_cats)
    used_colors = frozenset(
        context.original_colors[row][column] for row, column in fixed_cats
    )
    candidates = frozenset(
        coordinate
        for coordinate in unresolved
        if _coordinate_available(
            coordinate,
            context.original_colors,
            used_rows,
            used_columns,
            used_colors,
            fixed_cats,
        )
    )
    state = _SearchState(
        candidates=candidates,
        selected_cats=frozenset(fixed_cats),
        used_rows=used_rows,
        used_columns=used_columns,
        used_colors=used_colors,
    )
    progress = _SearchProgress(maximum_search_nodes=maximum_search_nodes)
    _search(context, state, progress)
    solutions = progress.solutions or []
    branch_groups = progress.branch_groups or []
    propagation_groups = progress.propagation_groups or []
    if len(solutions) >= 2:
        return CatsExactSearchResult(
            status=CatsExactSearchStatus.AMBIGUOUS,
            solution=None,
            solutions_found=2,
            search_nodes=progress.search_nodes,
            propagation_steps=progress.propagation_steps,
            branch_groups=tuple(branch_groups),
            propagation_groups=tuple(propagation_groups),
        )
    if progress.limit_reached:
        return CatsExactSearchResult(
            status=CatsExactSearchStatus.LIMIT_REACHED,
            solution=None,
            solutions_found=len(solutions),
            search_nodes=progress.search_nodes,
            propagation_steps=progress.propagation_steps,
            branch_groups=tuple(branch_groups),
            propagation_groups=tuple(propagation_groups),
        )
    if not solutions:
        return CatsExactSearchResult(
            status=CatsExactSearchStatus.UNSAT,
            solution=None,
            solutions_found=0,
            search_nodes=progress.search_nodes,
            propagation_steps=progress.propagation_steps,
            branch_groups=tuple(branch_groups),
            propagation_groups=tuple(propagation_groups),
        )
    return CatsExactSearchResult(
        status=CatsExactSearchStatus.UNIQUE,
        solution=solutions[0],
        solutions_found=1,
        search_nodes=progress.search_nodes,
        propagation_steps=progress.propagation_steps,
        branch_groups=tuple(branch_groups),
        propagation_groups=tuple(propagation_groups),
    )


def apply_unique_cats_exact_solution(
    board: Board,
    result: CatsExactSearchResult,
    *,
    original_color_matrix: OriginalColorMatrix | None = None,
) -> None:
    """Apply one proven unique solution through existing atomic Cats actions."""

    if result.status is not CatsExactSearchStatus.UNIQUE or result.solution is None:
        raise CatsExactSearchError("Only a UNIQUE exact-search result can be applied.")
    row_count = len(board.cells)
    column_count = len(board.cells[0]) if board.cells else 0
    if (
        not row_count
        or not column_count
        or any(len(row) != column_count for row in board.cells)
    ):
        raise CatsExactSearchError("Cats Board must be non-empty and rectangular.")
    solution = result.solution
    solution_set = frozenset(solution)
    if len(solution) != len(solution_set):
        raise CatsExactSearchError("Exact solution contains duplicate coordinates.")
    if not len(solution) == row_count == column_count:
        raise CatsExactSearchError(
            "Exact solution count must equal Board rows and columns."
        )
    for row, column in solution:
        if not 0 <= row < row_count or not 0 <= column < column_count:
            raise CatsExactSearchError(
                f"Exact solution coordinate ({row}, {column}) is outside Board."
            )
        if board.is_blocked(row, column):
            raise CatsExactSearchError(
                f"Exact solution coordinate ({row}, {column}) is blocked."
            )
        if not board.is_cat(row, column) and not board.is_unknown(row, column):
            raise CatsExactSearchError(
                f"Exact solution coordinate ({row}, {column}) is not available."
            )
    if {row for row, _ in solution} != set(range(row_count)):
        raise CatsExactSearchError("Exact solution must contain one cat per row.")
    if {column for _, column in solution} != set(range(column_count)):
        raise CatsExactSearchError("Exact solution must contain one cat per column.")
    if any(
        _coordinates_touch(first, second) for first, second in combinations(solution, 2)
    ):
        raise CatsExactSearchError("Exact solution cats cannot touch.")
    if original_color_matrix is not None:
        context, _, _ = _validate_input(board, original_color_matrix)
        solution_colors = {
            context.original_colors[row][column] for row, column in solution
        }
        if len(solution_colors) != row_count:
            raise CatsExactSearchError(
                "Exact solution must contain one cat per original color."
            )
    current_cats = tuple(
        (row, column)
        for row, values in enumerate(board.cells)
        for column in range(len(values))
        if board.is_cat(row, column)
    )
    if any(coordinate not in solution_set for coordinate in current_cats):
        raise CatsExactSearchError("Exact solution omits an existing fixed cat.")
    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            if (row, column) in solution_set:
                continue
            if not board.is_unknown(row, column) and not board.is_blocked(row, column):
                raise CatsExactSearchError(
                    f"Non-solution coordinate ({row}, {column}) has invalid state."
                )

    for row, column in solution:
        if board.is_cat(row, column):
            continue
        place_cat(board, row, column)
    for row, values in enumerate(board.cells):
        for column in range(len(values)):
            if (row, column) in solution_set or board.is_blocked(row, column):
                continue
            if not board.is_unknown(row, column):
                raise CatsExactSearchError(
                    f"Non-solution coordinate ({row}, {column}) has invalid state."
                )
            block_cell(board, row, column)


def _validate_input(
    board: Board,
    original_color_matrix: OriginalColorMatrix,
) -> tuple[_SearchContext, tuple[CellCoordinates, ...], tuple[CellCoordinates, ...]]:
    """Validate rectangular shapes, values, and current/original correspondence."""

    if not board.cells or not board.cells[0]:
        raise CatsExactSearchError("Cats Board must be non-empty.")
    row_count = len(board.cells)
    column_count = len(board.cells[0])
    if any(len(row) != column_count for row in board.cells):
        raise CatsExactSearchError("Cats Board matrix must be rectangular.")
    if len(original_color_matrix) != row_count or any(
        len(row) != column_count for row in original_color_matrix
    ):
        raise CatsExactSearchError(
            "Original color matrix shape must exactly match Board."
        )
    if row_count != column_count:
        raise CatsExactSearchError("Cats exact search requires a square Board.")

    fixed_cats: list[CellCoordinates] = []
    unresolved: list[CellCoordinates] = []
    original_ids: set[str] = set()
    for row, values in enumerate(board.cells):
        for column, current in enumerate(values):
            original = original_color_matrix[row][column]
            if not _is_color_id(original):
                raise CatsExactSearchError(
                    f"Invalid original color {original!r} at ({row}, {column})."
                )
            original_ids.add(original)
            if current == "K":
                fixed_cats.append((row, column))
            elif current == "X":
                continue
            elif _is_color_id(current):
                if current != original:
                    raise CatsExactSearchError(
                        f"Unresolved value {current} at ({row}, {column}) does not "
                        f"match original color {original}."
                    )
                unresolved.append((row, column))
            else:
                raise CatsExactSearchError(
                    f"Invalid current Board value {current!r} at ({row}, {column})."
                )
    color_ids = tuple(sorted(original_ids, key=_color_sort_key))
    expected_ids = tuple(f"C{index}" for index in range(len(color_ids)))
    if color_ids != expected_ids:
        raise CatsExactSearchError("Original colors must be contiguous C0..Cn.")
    if len(color_ids) != row_count:
        raise CatsExactSearchError(
            "Cats rows, columns, and original color count must be equal."
        )
    return (
        _SearchContext(
            row_count=row_count,
            column_count=column_count,
            color_ids=color_ids,
            original_colors=original_color_matrix,
        ),
        tuple(fixed_cats),
        tuple(unresolved),
    )


def _fixed_cat_contradiction(
    fixed_cats: tuple[CellCoordinates, ...],
    original_colors: OriginalColorMatrix,
) -> bool:
    rows = tuple(row for row, _ in fixed_cats)
    columns = tuple(column for _, column in fixed_cats)
    colors = tuple(original_colors[row][column] for row, column in fixed_cats)
    if len(rows) != len(set(rows)):
        return True
    if len(columns) != len(set(columns)):
        return True
    if len(colors) != len(set(colors)):
        return True
    return any(
        _coordinates_touch(first, second)
        for first, second in combinations(fixed_cats, 2)
    )


def _search(
    context: _SearchContext,
    state: _SearchState,
    progress: _SearchProgress,
) -> None:
    """Depth-first deterministic traversal that continues through solution two."""

    solutions = progress.solutions
    branch_groups = progress.branch_groups
    if solutions is None or branch_groups is None or len(solutions) >= 2:
        return
    if progress.search_nodes >= progress.maximum_search_nodes:
        progress.limit_reached = True
        return
    progress.search_nodes += 1
    propagated = _propagate(context, state, progress)
    if propagated is None:
        return
    groups = _constraint_groups(context, propagated)
    if not groups:
        if _is_complete_solution(context, propagated):
            solution = tuple(sorted(propagated.selected_cats))
            if solution not in solutions:
                solutions.append(solution)
        return

    branchable = tuple(group for group in groups if len(group.candidates) > 1)
    if not branchable:
        return
    selected_group = min(branchable, key=_group_sort_key)
    branch_groups.append(selected_group.label)
    for coordinate in selected_group.candidates:
        if len(solutions) >= 2 or progress.limit_reached:
            return
        assigned = _assign(context, propagated, coordinate)
        if assigned is not None:
            _search(context, assigned, progress)


def _propagate(
    context: _SearchContext,
    state: _SearchState,
    progress: _SearchProgress,
) -> _SearchState | None:
    """Apply deterministic color, row, and column singleton constraints."""

    current = state
    while True:
        groups = _constraint_groups(context, current)
        if any(not group.candidates for group in groups):
            return None
        singleton_groups = tuple(
            group for group in groups if len(group.candidates) == 1
        )
        if not singleton_groups:
            return current
        selected_group = min(singleton_groups, key=_group_sort_key)
        assigned = _assign(context, current, selected_group.candidates[0])
        if assigned is None:
            return None
        progress.propagation_steps += 1
        if progress.propagation_groups is not None:
            progress.propagation_groups.append(selected_group.label)
        current = assigned


def _constraint_groups(
    context: _SearchContext,
    state: _SearchState,
) -> tuple[_ConstraintGroup, ...]:
    """Build every currently unsatisfied group with row-major candidates."""

    ordered_candidates = tuple(sorted(state.candidates))
    groups: list[_ConstraintGroup] = []
    for color_id in context.color_ids:
        if color_id not in state.used_colors:
            groups.append(
                _ConstraintGroup(
                    "color",
                    color_id,
                    tuple(
                        coordinate
                        for coordinate in ordered_candidates
                        if context.original_colors[coordinate[0]][coordinate[1]]
                        == color_id
                    ),
                )
            )
    for row in range(context.row_count):
        if row not in state.used_rows:
            groups.append(
                _ConstraintGroup(
                    "row",
                    row,
                    tuple(
                        coordinate
                        for coordinate in ordered_candidates
                        if coordinate[0] == row
                    ),
                )
            )
    for column in range(context.column_count):
        if column not in state.used_columns:
            groups.append(
                _ConstraintGroup(
                    "column",
                    column,
                    tuple(
                        coordinate
                        for coordinate in ordered_candidates
                        if coordinate[1] == column
                    ),
                )
            )
    return tuple(groups)


def _assign(
    context: _SearchContext,
    state: _SearchState,
    coordinate: CellCoordinates,
) -> _SearchState | None:
    """Return one new lightweight state for a hypothetical cat assignment."""

    if coordinate not in state.candidates:
        return None
    row, column = coordinate
    color_id = context.original_colors[row][column]
    if (
        row in state.used_rows
        or column in state.used_columns
        or color_id in state.used_colors
        or any(_coordinates_touch(coordinate, cat) for cat in state.selected_cats)
    ):
        return None
    selected = state.selected_cats | {coordinate}
    candidates = frozenset(
        candidate
        for candidate in state.candidates
        if candidate != coordinate
        and candidate[0] != row
        and candidate[1] != column
        and context.original_colors[candidate[0]][candidate[1]] != color_id
        and not _coordinates_touch(candidate, coordinate)
    )
    return _SearchState(
        candidates=candidates,
        selected_cats=selected,
        used_rows=state.used_rows | {row},
        used_columns=state.used_columns | {column},
        used_colors=state.used_colors | {color_id},
    )


def _is_complete_solution(context: _SearchContext, state: _SearchState) -> bool:
    """Require exact row, column, color, count, and non-touching completion."""

    expected_count = context.row_count
    if len(state.selected_cats) != expected_count:
        return False
    if state.used_rows != frozenset(range(context.row_count)):
        return False
    if state.used_columns != frozenset(range(context.column_count)):
        return False
    if state.used_colors != frozenset(context.color_ids):
        return False
    ordered = tuple(sorted(state.selected_cats))
    return not any(
        _coordinates_touch(first, second) for first, second in combinations(ordered, 2)
    )


def _coordinate_available(
    coordinate: CellCoordinates,
    original_colors: OriginalColorMatrix,
    used_rows: frozenset[int],
    used_columns: frozenset[int],
    used_colors: frozenset[str],
    fixed_cats: tuple[CellCoordinates, ...],
) -> bool:
    row, column = coordinate
    return (
        row not in used_rows
        and column not in used_columns
        and original_colors[row][column] not in used_colors
        and not any(_coordinates_touch(coordinate, cat) for cat in fixed_cats)
    )


def _group_sort_key(
    group: _ConstraintGroup,
) -> tuple[int, int, int, tuple[CellCoordinates, ...]]:
    type_order = {"color": 0, "row": 1, "column": 2}
    numeric_identifier = (
        _color_sort_key(group.identifier)
        if isinstance(group.identifier, str)
        else group.identifier
    )
    return (
        len(group.candidates),
        type_order[group.group_type],
        numeric_identifier,
        group.candidates,
    )


def _color_sort_key(color_id: str) -> int:
    return int(color_id[1:])


def _is_color_id(value: object) -> bool:
    return isinstance(value, str) and value.startswith("C") and value[1:].isdigit()


def _coordinates_touch(
    first: CellCoordinates,
    second: CellCoordinates,
) -> bool:
    return (
        first != second
        and max(
            abs(first[0] - second[0]),
            abs(first[1] - second[1]),
        )
        <= 1
    )
