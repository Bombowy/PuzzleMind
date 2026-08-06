"""Win32 adapter for emitting one explicit pointer click at a desktop point."""

from typing import Final, Protocol, cast

from logicforge.automation.mouse import MouseButton, MouseController, ScreenPoint

MOUSEEVENTF_LEFTDOWN: Final = 0x0002
MOUSEEVENTF_LEFTUP: Final = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final = 0x0008
MOUSEEVENTF_RIGHTUP: Final = 0x0010
MOUSEEVENTF_MIDDLEDOWN: Final = 0x0020
MOUSEEVENTF_MIDDLEUP: Final = 0x0040

_BUTTON_FLAGS: Final[dict[MouseButton, tuple[int, int]]] = {
    MouseButton.LEFT: (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    MouseButton.RIGHT: (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    MouseButton.MIDDLE: (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}


class MouseAutomationError(RuntimeError):
    """Report a native failure while positioning or clicking the pointer."""


class MouseApi(Protocol):
    """Describe only the pywin32 calls required by the mouse adapter."""

    def SetCursorPos(self, point: tuple[int, int]) -> None:
        """Move the native pointer to one virtual-desktop coordinate."""

        ...

    def mouse_event(
        self,
        flags: int,
        dx: int,
        dy: int,
        data: int,
        extra_info: int,
    ) -> None:
        """Emit one low-level mouse transition using Win32 flags."""

        ...


def _load_mouse_api() -> MouseApi:
    """Import pywin32 only when the first native click is actually requested."""

    import win32api

    return cast(MouseApi, win32api)


class Win32MouseController(MouseController):
    """Implement one portable click as cursor positioning plus down/up events."""

    __slots__ = ("_mouse_api",)

    def __init__(self, mouse_api: MouseApi | None = None) -> None:
        """Accept an optional native boundary for safe deterministic tests."""

        self._mouse_api = mouse_api

    def click(
        self,
        point: ScreenPoint,
        button: MouseButton = MouseButton.LEFT,
    ) -> None:
        """Move to ``point`` and emit exactly one matching down/up pair."""

        try:
            mouse_api = self._get_mouse_api()
            down_flag, up_flag = _BUTTON_FLAGS[button]
            mouse_api.SetCursorPos((point.x, point.y))
            mouse_api.mouse_event(down_flag, 0, 0, 0, 0)
            mouse_api.mouse_event(up_flag, 0, 0, 0, 0)
        except Exception as error:
            raise MouseAutomationError(
                f"Failed to click {button.value} at ({point.x}, {point.y})."
            ) from error

    def _get_mouse_api(self) -> MouseApi:
        """Resolve and retain the native API only on the first actual click."""

        if self._mouse_api is None:
            self._mouse_api = _load_mouse_api()
        return self._mouse_api
