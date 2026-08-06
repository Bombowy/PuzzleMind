"""Domain enumerations shared by board, rule, and solver abstractions."""

from enum import StrEnum, auto


class CellState(StrEnum):
    """Describe the lifecycle state of a cell without encoding puzzle-specific rules.

    TODO: Extend this vocabulary only when a state is proven to be shared by
    multiple puzzle plugins; plugin-only states belong inside their own packages.
    """

    UNKNOWN = auto()
    GIVEN = auto()
    SOLVED = auto()
    CONFLICT = auto()


class CandidateStatus(StrEnum):
    """Describe whether a candidate is still viable in an immutable board snapshot.

    TODO: Add provenance identifiers when the deduction engine can associate a
    candidate transition with the exact rule application that caused it.
    """

    POSSIBLE = auto()
    CONFIRMED = auto()
    ELIMINATED = auto()


class RegionKind(StrEnum):
    """Classify generic region topology while leaving game semantics to plugins.

    TODO: Introduce plugin-defined region metadata instead of growing this enum
    with concepts that are meaningful to only one puzzle family.
    """

    ROW = auto()
    COLUMN = auto()
    AREA = auto()
    CUSTOM = auto()


class DeductionKind(StrEnum):
    """Classify high-level deduction effects for explanations and diagnostics.

    TODO: Refine these categories after the first rule engine establishes the
    stable, puzzle-independent deduction vocabulary.
    """

    ASSIGNMENT = auto()
    ELIMINATION = auto()
    CONTRADICTION = auto()
