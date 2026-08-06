"""High-level parsing port that translates screenshots into domain boards."""

from abc import ABC, abstractmethod

from logicforge.core.board import Board
from logicforge.vision.screenshot import Screenshot


class PuzzleParser(ABC):
    """Define the plugin-facing boundary between vision data and the core domain.

    Parsers orchestrate detector ports and interpret observations for one puzzle
    type. They must not perform deduction or mutate external application state.
    """

    @abstractmethod
    def parse(self, screenshot: Screenshot) -> Board:
        """Translate one screenshot into the single mutable domain board.

        TODO: Implement puzzle-specific observation mapping, validation, and rich
        parse diagnostics in v0.2 without introducing solver behavior.
        """

        raise NotImplementedError
