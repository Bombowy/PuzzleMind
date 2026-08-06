"""Application contracts for capturing one explicitly located desktop window."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class WindowCaptureError(RuntimeError):
    """Represent an expected failure while locating or capturing a target window.

    A dedicated application error keeps scripts independent from pywin32, MSS, and
    operating-system exception types while retaining the original cause chain.
    """


class WindowNotFoundError(WindowCaptureError):
    """Indicate that no visible window satisfies the configured title policy."""


class WindowUnavailableError(WindowCaptureError):
    """Indicate that a located window cannot currently produce a valid capture."""


@dataclass(frozen=True, slots=True)
class WindowBounds:
    """Describe one window rectangle in virtual-desktop pixel coordinates.

    Width and height are validated at the boundary so an invalid native rectangle
    can never be mistaken for a request to capture a monitor or the full desktop.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject empty or inverted rectangles before they reach a capture adapter."""

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Window width and height must both be positive.")

    def as_bbox(self) -> tuple[int, int, int, int]:
        """Return the MSS-compatible ``left, top, right, bottom`` capture box."""

        return (self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """Carry the resolved native window title and its validated screen rectangle."""

    title: str
    bounds: WindowBounds


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Report the persisted screenshot and observable capture measurements.

    The result contains no CV interpretation; it records only window identity,
    output location, captured pixel dimensions, and elapsed capture/save time.
    """

    window: WindowInfo
    output_path: Path
    resolution_width: int
    resolution_height: int
    elapsed_seconds: float


class WindowLocator(ABC):
    """Define a replaceable port for resolving one eligible desktop window."""

    @abstractmethod
    def locate(self) -> WindowInfo:
        """Return the selected visible window or raise a typed capture error."""

        raise NotImplementedError


class WindowCapturer(ABC):
    """Define a port that persists pixels from one explicit window rectangle."""

    @abstractmethod
    def capture(self, window: WindowInfo, destination: Path) -> CaptureResult:
        """Capture only ``window.bounds`` and save the result at ``destination``."""

        raise NotImplementedError


class WindowCaptureService:
    """Orchestrate window lookup and rectangle capture without infrastructure APIs.

    This application service is intentionally small: detector, parser, solver, and
    automation responsibilities are outside this milestone and cannot enter here.
    """

    def __init__(self, locator: WindowLocator, capturer: WindowCapturer) -> None:
        """Receive window infrastructure through narrow dependency-inverted ports."""

        self._locator = locator
        self._capturer = capturer

    def capture(self, destination: Path) -> CaptureResult:
        """Locate the configured window and persist exactly its current rectangle."""

        window = self._locator.locate()
        return self._capturer.capture(window, destination)
