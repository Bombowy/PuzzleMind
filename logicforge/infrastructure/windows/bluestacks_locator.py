"""pywin32 adapter for locating the visible BlueStacks App Player window."""

from typing import Final

from logicforge.vision.window_capture import (
    WindowBounds,
    WindowInfo,
    WindowLocator,
    WindowNotFoundError,
    WindowUnavailableError,
)

EXACT_TITLE: Final = "BlueStacks App Player"
TITLE_FRAGMENT: Final = "BlueStacks"


class Win32BlueStacksWindowLocator(WindowLocator):
    """Resolve BlueStacks through Win32 top-level window APIs.

    Exact visible title matching has priority. If it fails, the adapter selects the
    first visible top-level window whose title contains ``BlueStacks``, preserving
    the fallback behavior required by this milestone.
    """

    def locate(self) -> WindowInfo:
        """Locate BlueStacks and translate its Win32 rectangle to a typed value."""

        import win32gui

        exact_handle = win32gui.FindWindow(None, EXACT_TITLE)
        if exact_handle and win32gui.IsWindowVisible(exact_handle):
            return self._describe_window(exact_handle)

        matching_handles: list[int] = []

        def collect_visible_bluestacks_window(
            handle: int,
            _: object,
        ) -> bool:
            """Collect visible partial-title matches in native enumeration order."""

            if not win32gui.IsWindowVisible(handle):
                return True

            title = win32gui.GetWindowText(handle)
            if TITLE_FRAGMENT.casefold() in title.casefold():
                matching_handles.append(handle)
            return True

        win32gui.EnumWindows(collect_visible_bluestacks_window, None)
        if not matching_handles:
            raise WindowNotFoundError(
                f'No visible window titled "{EXACT_TITLE}" or containing '
                f'"{TITLE_FRAGMENT}" was found.'
            )

        return self._describe_window(matching_handles[0])

    @staticmethod
    def _describe_window(handle: int) -> WindowInfo:
        """Read and validate title, minimized state, and screen-space dimensions."""

        import win32gui

        if win32gui.IsIconic(handle):
            raise WindowUnavailableError(
                "The BlueStacks window is minimized; restore it before capture."
            )

        title = win32gui.GetWindowText(handle)
        left, top, right, bottom = win32gui.GetWindowRect(handle)
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
