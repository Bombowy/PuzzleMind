"""Reserved generic parser boundary outside the active Cats analysis pipeline."""

from logicforge.core.board import Board
from logicforge.vision.parser import PuzzleParser
from logicforge.vision.screenshot import Screenshot


class CatsParser(PuzzleParser):
    """Reserve the generic ``PuzzleParser`` adapter for a future unified API.

    Production Cats analysis is composed through backend-neutral detector ports in
    ``logicforge.application.cats.analysis`` and does not use this scaffold.
    """

    def parse(self, screenshot: Screenshot) -> Board:
        """Reject use until the generic parser API is connected intentionally."""

        raise NotImplementedError
