"""Formatting port for human- and machine-readable explanations."""

from abc import ABC, abstractmethod

from logicforge.explain.explanation import Explanation


class ExplanationFormatter(ABC):
    """Convert semantic explanations into a delivery-channel representation."""

    @abstractmethod
    def format(self, explanation: Explanation) -> str:
        """Render one explanation without changing its semantic content.

        TODO: Implement plain-text, Markdown, and structured JSON formatters in
        v0.6 with escaping, localization, and deterministic snapshot tests.
        """

        raise NotImplementedError
