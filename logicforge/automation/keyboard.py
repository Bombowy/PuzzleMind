"""Port for future keyboard-based gameplay automation."""

from abc import ABC, abstractmethod


class KeyboardController(ABC):
    """Abstract keyboard side effects from puzzle-solving and plugin code.

    TODO: Add an explicit permission and focus-guard contract before any adapter is
    permitted to emit operating-system events in the v0.7 milestone.
    """

    @abstractmethod
    def press(self, key: str) -> None:
        """Press and release one normalized key identifier.

        TODO: Implement platform-specific key mapping, validation, dry-run tracing,
        and fail-safe cancellation as part of automatic gameplay support.
        """

        raise NotImplementedError

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Emit validated text through the configured keyboard adapter.

        TODO: Implement explicit character support, pacing, focus checks, and
        redacted audit logs before accepting user-provided text.
        """

        raise NotImplementedError
