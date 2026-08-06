"""pywin32 adapter for locating a window owned by a BlueStacks executable."""

import ntpath
from collections.abc import Callable
from typing import Final, Protocol, cast

from logicforge.vision.window_capture import (
    WindowBounds,
    WindowInfo,
    WindowLocator,
    WindowNotFoundError,
    WindowUnavailableError,
)

EXACT_TITLE: Final = "BlueStacks App Player"
TITLE_FRAGMENT: Final = "BlueStacks"
BLUESTACKS_EXECUTABLE_NAMES: Final = frozenset(
    {
        "hd-player.exe",
        "bluestacks.exe",
        "bluestacksappplayer.exe",
    }
)

type WindowEnumerationCallback = Callable[[int, object], bool]
type ProcessNameResolver = Callable[[int], str | None]


class WindowApi(Protocol):
    """Describe the narrow win32gui surface required by the locator."""

    def FindWindow(self, class_name: object, window_name: str) -> int:
        """Return the exact-title top-level window handle or zero."""

        ...

    def IsWindowVisible(self, handle: int) -> bool:
        """Return whether a top-level window has the visible style."""

        ...

    def GetWindowText(self, handle: int) -> str:
        """Return the current top-level window title."""

        ...

    def EnumWindows(
        self,
        callback: WindowEnumerationCallback,
        extra: object,
    ) -> None:
        """Enumerate top-level windows in native order."""

        ...

    def IsIconic(self, handle: int) -> bool:
        """Return whether a window is minimized."""

        ...

    def GetWindowRect(self, handle: int) -> tuple[int, int, int, int]:
        """Return a window rectangle in virtual-desktop coordinates."""

        ...


def _load_window_api() -> WindowApi:
    """Load win32gui lazily so non-Windows package inspection remains possible."""

    import win32gui

    return cast(WindowApi, win32gui)


def _resolve_process_executable_name(window_handle: int) -> str | None:
    """Return the owning process executable name or reject unreadable process data.

    The process handle is always closed. Any failure to obtain the PID, open the
    process, or read its main module path produces ``None`` so fallback selection
    fails closed instead of trusting an unverified window title.
    """

    import win32api
    import win32con
    import win32process

    process_handle: object | None = None
    try:
        _, process_id = win32process.GetWindowThreadProcessId(window_handle)
        if process_id <= 0:
            return None

        required_access = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
        process_handle = win32api.OpenProcess(required_access, False, process_id)
        executable_path = win32process.GetModuleFileNameEx(process_handle, 0)
        executable_name = ntpath.basename(executable_path).strip()
        return executable_name or None
    except Exception:
        return None
    finally:
        if process_handle is not None:
            try:
                win32api.CloseHandle(cast(int, process_handle))
            except Exception:
                pass


class Win32BlueStacksWindowLocator(WindowLocator):
    """Resolve BlueStacks through verified Win32 title and process ownership.

    Exact visible title matching retains priority. Fallback candidates must be
    visible, contain ``BlueStacks`` in their title, and belong to an explicitly
    supported BlueStacks executable. Unreadable process information is rejected.
    """

    def __init__(
        self,
        *,
        window_api: WindowApi | None = None,
        process_name_resolver: ProcessNameResolver | None = None,
    ) -> None:
        """Accept injectable native boundaries for deterministic unit testing."""

        self._window_api = window_api or _load_window_api()
        self._process_name_resolver = (
            process_name_resolver or _resolve_process_executable_name
        )

    def locate(self) -> WindowInfo:
        """Locate BlueStacks and translate its Win32 rectangle to a typed value."""

        exact_handle = self._window_api.FindWindow(None, EXACT_TITLE)
        if exact_handle and self._window_api.IsWindowVisible(exact_handle):
            return self._describe_window(exact_handle)

        matching_handles: list[int] = []

        def collect_verified_bluestacks_window(
            handle: int,
            _: object,
        ) -> bool:
            """Collect candidates that pass visibility, title, and process checks."""

            if not self._window_api.IsWindowVisible(handle):
                return True

            title = self._window_api.GetWindowText(handle)
            if TITLE_FRAGMENT.casefold() not in title.casefold():
                return True

            executable_name = self._process_name_resolver(handle)
            if executable_name is None:
                return True
            if executable_name.casefold() not in BLUESTACKS_EXECUTABLE_NAMES:
                return True

            matching_handles.append(handle)
            return True

        self._window_api.EnumWindows(collect_verified_bluestacks_window, None)
        if not matching_handles:
            raise WindowNotFoundError(
                f'No visible BlueStacks-owned window titled "{EXACT_TITLE}" or '
                f'containing "{TITLE_FRAGMENT}" was found.'
            )

        return self._describe_window(matching_handles[0])

    def _describe_window(self, handle: int) -> WindowInfo:
        """Read and validate title, minimized state, and screen-space dimensions."""

        if self._window_api.IsIconic(handle):
            raise WindowUnavailableError(
                "The BlueStacks window is minimized; restore it before capture."
            )

        title = self._window_api.GetWindowText(handle)
        left, top, right, bottom = self._window_api.GetWindowRect(handle)
        try:
            bounds = WindowBounds(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
            )
        except ValueError as error:
            raise WindowUnavailableError(
                f'BlueStacks returned an invalid window rectangle: "{title}".'
            ) from error

        return WindowInfo(title=title, bounds=bounds)
