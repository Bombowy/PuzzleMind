"""Discovery boundary for built-in and third-party puzzle plugins."""

from abc import ABC, abstractmethod

from logicforge.plugins.base import PuzzlePlugin


class PluginRegistry(ABC):
    """Resolve plugin implementations without exposing packaging mechanisms.

    TODO: Implement Python entry-point discovery, duplicate handling, compatibility
    checks, deterministic ordering, and safe load failures before third-party use.
    """

    @abstractmethod
    def available(self) -> tuple[PuzzlePlugin, ...]:
        """Return every compatible plugin in deterministic identifier order.

        TODO: Implement cached discovery and structured diagnostics once the plugin
        compatibility model is finalized.
        """

        raise NotImplementedError

    @abstractmethod
    def get(self, identifier: str) -> PuzzlePlugin:
        """Resolve one plugin or raise a future domain-specific lookup error.

        TODO: Implement normalization, unknown-plugin diagnostics, and version
        selection together with entry-point discovery.
        """

        raise NotImplementedError
