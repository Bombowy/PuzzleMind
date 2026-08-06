"""Output port for persisting renderer artifacts."""

from abc import ABC, abstractmethod
from pathlib import Path

from logicforge.visualization.board_renderer import RenderArtifact


class ImageExporter(ABC):
    """Persist supported visual artifacts without coupling renderers to storage."""

    @abstractmethod
    def export(self, artifact: RenderArtifact, destination: Path) -> Path:
        """Write an artifact atomically and return its resolved output path.

        TODO: Implement media-type validation, safe path handling, atomic writes,
        and overwrite policy after concrete render formats are introduced.
        """

        raise NotImplementedError
