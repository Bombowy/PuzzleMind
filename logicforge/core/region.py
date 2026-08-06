"""Logical groupings of cells used by puzzle rules."""

from dataclasses import dataclass

from logicforge.core.coordinates import Coordinates
from logicforge.core.enums import RegionKind
from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class Region:
    """Group coordinates that share a puzzle-defined relationship.

    Core code stores membership and identity only. Plugins are responsible for
    interpreting custom metadata and attaching actual constraints through rules.

    TODO: Add topology validation once parsers can report overlapping, incomplete,
    or malformed region detections through a common diagnostics API.
    """

    identifier: str
    kind: RegionKind
    coordinates: tuple[Coordinates, ...]
    metadata: Metadata = ()
