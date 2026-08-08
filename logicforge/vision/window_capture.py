"""Application contracts for capturing one explicitly located desktop window."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from logicforge.vision.screenshot import Screenshot

DEFAULT_DEBUG_IMAGE_PATH = Path("artifacts/vision/bluestacks_capture.png")
type DebugImageSaver = Callable[[Screenshot, Path], Path]


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


class WindowLocator(ABC):
    """Define a replaceable port for resolving one eligible desktop window."""

    @abstractmethod
    def locate(self) -> WindowInfo:
        """Return the selected visible window or raise a typed capture error."""

        raise NotImplementedError


class WindowCapturer(ABC):
    """Define a port that captures one explicit window rectangle into memory."""

    @abstractmethod
    def capture(self, window: WindowInfo) -> Screenshot:
        """Capture only ``window.bounds`` and return an in-memory BGR snapshot."""

        raise NotImplementedError


class WindowCaptureService:
    """Orchestrate window lookup and rectangle capture without infrastructure APIs.

    This service deliberately owns no detection, solving, or automation policy.
    """

    def __init__(
        self,
        locator: WindowLocator,
        capturer: WindowCapturer,
        *,
        debug_image_saver: DebugImageSaver | None = None,
        debug_image_path: Path = DEFAULT_DEBUG_IMAGE_PATH,
    ) -> None:
        """Receive capture and optional debug persistence through injected ports."""

        self._locator = locator
        self._capturer = capturer
        self._debug_image_saver = debug_image_saver
        self._debug_image_path = debug_image_path

    def locate_window(self) -> WindowInfo:
        """Expose one resolved window for callers that need capture diagnostics."""

        return self._locator.locate()

    def capture_window(self, window: WindowInfo, *, debug: bool = False) -> Screenshot:
        """Capture a resolved window and optionally persist a separate debug image."""

        screenshot = self._capturer.capture(window)
        if debug:
            if self._debug_image_saver is None:
                raise WindowCaptureError(
                    "Debug capture requested without a configured debug image saver."
                )
            self._debug_image_saver(screenshot, self._debug_image_path)
        return screenshot

    def capture(self, *, debug: bool = False) -> Screenshot:
        """Locate and capture the configured window, returning pixels in memory."""

        return self.capture_window(self.locate_window(), debug=debug)
