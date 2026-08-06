"""Puzzle-neutral candidate value records."""

from dataclasses import dataclass

from logicforge.core.enums import CandidateStatus
from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class Candidate:
    """Represent one possible value and its status at a point in deduction time.

    The opaque metadata map is intentionally reserved for adapters and plugins;
    generic solver code must not inspect plugin-specific keys.

    TODO: Add typed provenance after rule outcomes expose stable identifiers and
    explanation links rather than relying on an unstructured metadata mapping.
    """

    value: str
    status: CandidateStatus = CandidateStatus.POSSIBLE
    metadata: Metadata = ()
