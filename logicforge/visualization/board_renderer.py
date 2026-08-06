"""Presentation boundary for board visualization."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.core.board import Board
from logicforge.core.metadata import Metadata


@dataclass(frozen=True, slots=True)
class RenderArtifact:
    """Carry a renderer-owned output payload plus portable media metadata.

    TODO: Define typed raster, vector, and text variants after consumers establish
    ownership, streaming, and serialization requirements.
    """

    media_type: str
    payload: object
    metadata: Metadata = ()


class BoardRenderer(ABC):
    """Render the current mutable board without introducing UI domain coupling."""

    @abstractmethod
    def render(self, board: Board) -> RenderArtifact:
        """Create a presentation artifact from the board's current matrix.

        TODO: Implement accessible text and image renderers after v0.3 defines the
        stable board representation and plugin symbol contract.
        """

        raise NotImplementedError
