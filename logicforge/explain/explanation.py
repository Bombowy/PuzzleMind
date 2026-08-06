"""Puzzle-neutral explanation records derived from solver provenance."""

from dataclasses import dataclass

from logicforge.core.coordinates import Coordinates
from logicforge.core.enums import DeductionKind
from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class Explanation:
    """Describe why a deduction is valid in a presentation-neutral form.

    The record deliberately stores semantic content rather than formatted text so
    CLI, GUI, and machine-readable formatters can share the same evidence.

    TODO: Add structured premises, conclusion links, and localization keys during
    v0.6 after real Cats deductions define the minimum useful evidence model.
    """

    rule_id: str
    kind: DeductionKind
    summary: str
    details: str
    coordinates: tuple[Coordinates, ...] = ()
    evidence: Metadata = ()
