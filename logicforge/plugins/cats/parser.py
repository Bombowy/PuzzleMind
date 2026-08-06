"""Cats-specific screenshot parsing boundary; no parsing logic exists yet."""

from logicforge.core.board import Board
from logicforge.vision.parser import PuzzleParser
from logicforge.vision.screenshot import Screenshot


class CatsParser(PuzzleParser):
    """Translate Cats screenshots into generic board snapshots in a future release.

    TODO: Compose calibrated board, grid, symbol, and color detectors in v0.5 and
    map their evidence into validated Cats regions without performing deductions.
    """

    def parse(self, screenshot: Screenshot) -> Board:
        """Parse a Cats screenshot through future injected vision dependencies.

        TODO: Implement Cats-specific observation mapping, confidence thresholds,
        region reconstruction, and actionable ambiguity diagnostics in v0.5.
        """

        raise NotImplementedError
