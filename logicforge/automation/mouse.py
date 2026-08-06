"""Port for future pointer-based gameplay automation."""

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
    """Identify an absolute screen position in physical pixel coordinates.

    TODO: Add display identity and scaling metadata before v0.7 so multi-monitor
    automation can validate coordinate systems and DPI transformations safely.
    """

    x: int
    y: int


class MouseController(ABC):
    """Abstract operating-system pointer actions behind an auditable boundary.

    Automation remains outside solving use cases so tests can use safe fakes and
    production adapters can enforce focus, bounds, and emergency-stop policies.
    """

    @abstractmethod
    def click(self, point: ScreenPoint, button: MouseButton = MouseButton.LEFT) -> None:
        """Perform one validated click at an absolute screen point.

        TODO: Implement platform adapters, dry-run support, focus verification,
        rate limiting, and fail-safe abort behavior for the v0.7 milestone.
        """

        raise NotImplementedError
