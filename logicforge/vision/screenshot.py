"""Transport model for screenshots accepted by vision adapters."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class Screenshot:
    """Carry screenshot metadata and backend-owned image payload across a boundary.

    The payload remains opaque to keep NumPy, Pillow, and OpenCV out of the domain
    dependency graph. Concrete vision adapters validate and narrow its type.

    TODO: Define a stable pixel-buffer protocol in v0.2 after benchmarking the
    required color fidelity, ownership semantics, and conversion overhead.
    """

    width: int
    height: int
    payload: object
    captured_at: datetime | None = None
    source: Path | None = None
    metadata: Metadata = ()
