"""Port for dependency-injected pointer-based gameplay automation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum, auto


class MouseButton(StrEnum):
    """Identify portable pointer buttons supported by automation adapters.

    TODO: Extend only when a supported game interaction requires another portable
    button; adapter-specific controls must remain outside this shared enum.
    """

    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()


@dataclass(frozen=True, slots=True)
class ScreenPoint:
    """Identify an absolute physical-pixel position on the virtual desktop."""

    x: int
    y: int


class MouseController(ABC):
    """Abstract operating-system pointer actions behind an auditable boundary.

    Automation remains outside solving use cases so tests can use safe fakes and
    production adapters can enforce focus, bounds, and emergency-stop policies.
    """

    @abstractmethod
    def click(self, point: ScreenPoint, button: MouseButton = MouseButton.LEFT) -> None:
        """Perform one adapter-validated click at an absolute screen point."""

        raise NotImplementedError
