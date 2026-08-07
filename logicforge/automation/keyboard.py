"""Reserved port for keyboard-based gameplay automation."""

from abc import ABC, abstractmethod


class KeyboardController(ABC):
    """Abstract keyboard side effects from puzzle-solving and plugin code.

    A concrete adapter must define explicit permission and focus guards before it
    is permitted to emit operating-system events.
    """

    @abstractmethod
    def press(self, key: str) -> None:
        """Press and release one normalized key identifier.

        Concrete adapters own key mapping, validation, tracing, and cancellation.
        """

        raise NotImplementedError

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Emit validated text through the configured keyboard adapter.

        Concrete adapters own character support, pacing, focus checks, and audit
        logging.
        """

        raise NotImplementedError
