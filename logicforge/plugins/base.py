"""Stable extension contract implemented by every puzzle plugin."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from logicforge.rules.base_rule import BaseRule
from logicforge.vision.parser import PuzzleParser


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Describe plugin identity and compatibility without importing its internals.

    TODO: Add semantic compatibility ranges and optional capabilities when the
    packaging and third-party discovery strategy is finalized.
    """

    identifier: str
    display_name: str
    version: str
    description: str


class PuzzlePlugin(ABC):
    """Expose all puzzle-specific components through one replaceable extension.

    Plugins may interpret screenshots and define rules, but cannot bypass domain
    contracts or directly invoke solver, automation, or persistence adapters.
    """

    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return immutable plugin identity and compatibility information.

        TODO: Implement metadata in concrete plugins and validate identifiers and
        versions when entry-point discovery is introduced.
        """

        raise NotImplementedError

    @abstractmethod
    def create_parser(self) -> PuzzleParser:
        """Create the parser composition owned by this plugin.

        TODO: Inject configured vision ports through a future plugin context rather
        than constructing infrastructure dependencies inside plugin code.
        """

        raise NotImplementedError

    @abstractmethod
    def create_rules(self) -> tuple[BaseRule, ...]:
        """Create a deterministically ordered, stateless rule collection.

        TODO: Implement plugin rule catalogs after rule ordering and compatibility
        validation are specified by the v0.4 rule engine.
        """

        raise NotImplementedError
