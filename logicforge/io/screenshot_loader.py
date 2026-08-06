"""Input port for loading screenshots from persistent storage."""

from abc import ABC, abstractmethod
from pathlib import Path

from logicforge.vision.screenshot import Screenshot


class ScreenshotLoader(ABC):
    """Load external image files into the backend-neutral screenshot contract."""

    @abstractmethod
    def load(self, path: Path) -> Screenshot:
        """Decode and validate one screenshot file.

        TODO: Implement format detection, size limits, safe decoding, metadata
        normalization, and actionable input errors in the v0.2 milestone.
        """

        raise NotImplementedError
