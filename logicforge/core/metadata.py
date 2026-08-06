"""Shared immutable shape for optional extension metadata."""

# Metadata is an ordered tuple rather than a mutable mapping so frozen snapshots
# cannot be changed indirectly through a caller-owned dictionary. TODO: Replace
# opaque values with subsystem-specific typed records as public contracts mature.
type Metadata = tuple[tuple[str, object], ...]
