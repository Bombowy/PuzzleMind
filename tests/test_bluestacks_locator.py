"""Tests for process-verified BlueStacks fallback window selection."""

from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from logicforge.infrastructure.windows.bluestacks_locator import (
    EXACT_TITLE,
    Win32BlueStacksWindowLocator,
    WindowApi,
    WindowEnumerationCallback,
)
from logicforge.vision.window_capture import WindowNotFoundError


@dataclass(frozen=True, slots=True)
class FakeWindow:
    """Describe one deterministic top-level window exposed by the fake Win32 API."""

    title: str
    visible: bool = True
    minimized: bool = False
    rectangle: tuple[int, int, int, int] = (10, 20, 810, 620)


class FakeWindowApi(WindowApi):
    """Provide the locator's minimal win32gui contract without desktop access."""

    def __init__(
        self,
        windows: Mapping[int, FakeWindow],
        *,
        exact_handle: int = 0,
    ) -> None:
        """Store deterministic windows and the exact-title lookup result."""

        self._windows = dict(windows)
        self._exact_handle = exact_handle

    def FindWindow(self, class_name: object, window_name: str) -> int:
        """Return the configured exact handle only for the required title."""

        assert class_name is None
        assert window_name == EXACT_TITLE
        return self._exact_handle

    def IsWindowVisible(self, handle: int) -> bool:
        """Return the fake window visibility flag."""

        return self._windows[handle].visible

    def GetWindowText(self, handle: int) -> str:
        """Return the fake window title."""

        return self._windows[handle].title

    def EnumWindows(
        self,
        callback: WindowEnumerationCallback,
        extra: object,
    ) -> None:
        """Visit fake handles in insertion order to emulate native enumeration."""

        for handle in self._windows:
            if not callback(handle, extra):
                break

    def IsIconic(self, handle: int) -> bool:
        """Return the fake minimized state."""

        return self._windows[handle].minimized

    def GetWindowRect(self, handle: int) -> tuple[int, int, int, int]:
        """Return the fake virtual-desktop rectangle."""

        return self._windows[handle].rectangle


class FakeProcessNameResolver:
    """Map fake window handles to executable names or simulated read failures."""

    def __init__(self, executable_names: Mapping[int, str | None]) -> None:
        """Store process results and initialize an observable call list."""

        self._executable_names = dict(executable_names)
        self.requested_handles: list[int] = []

    def __call__(self, handle: int) -> str | None:
        """Return the configured process name and record ownership verification."""

        self.requested_handles.append(handle)
        return self._executable_names.get(handle)


def create_locator(
    windows: Mapping[int, FakeWindow],
    executable_names: Mapping[int, str | None],
    *,
    exact_handle: int = 0,
) -> tuple[Win32BlueStacksWindowLocator, FakeProcessNameResolver]:
    """Compose a locator with deterministic window and process native boundaries."""

    process_resolver = FakeProcessNameResolver(executable_names)
    locator = Win32BlueStacksWindowLocator(
        window_api=FakeWindowApi(windows, exact_handle=exact_handle),
        process_name_resolver=process_resolver,
    )
    return locator, process_resolver


def test_exact_valid_bluestacks_window_keeps_priority() -> None:
    """Preserve exact visible title selection without entering fallback discovery."""

    locator, process_resolver = create_locator(
        {1: FakeWindow(title=EXACT_TITLE)},
        {1: "HD-Player.exe"},
        exact_handle=1,
    )

    window = locator.locate()

    assert window.title == EXACT_TITLE
    assert window.bounds.width == 800
    assert process_resolver.requested_handles == []


def test_valid_fallback_owned_by_hd_player_is_selected_case_insensitively() -> None:
    """Accept a partial-title fallback only when HD-Player owns the window."""

    locator, process_resolver = create_locator(
        {2: FakeWindow(title="BlueStacks 5 - Instance")},
        {2: "HD-PLAYER.EXE"},
    )

    window = locator.locate()

    assert window.title == "BlueStacks 5 - Instance"
    assert process_resolver.requested_handles == [2]


def test_pycharm_title_containing_bluestacks_capture_is_rejected() -> None:
    """Never trust an editor title that merely displays a BlueStacks-named file."""

    locator, process_resolver = create_locator(
        {3: FakeWindow(title="bluestacks_capture.png - PuzzleMind - PyCharm")},
        {3: "pycharm64.exe"},
    )

    with pytest.raises(WindowNotFoundError):
        locator.locate()

    assert process_resolver.requested_handles == [3]


def test_unrelated_process_with_bluestacks_in_title_is_rejected() -> None:
    """Reject browsers, explorers, terminals, and other unrelated title matches."""

    locator, process_resolver = create_locator(
        {
            4: FakeWindow(title="BlueStacks help - Chrome"),
            5: FakeWindow(title="BlueStacks folder"),
            6: FakeWindow(title="BlueStacks logs - Terminal"),
        },
        {4: "chrome.exe", 5: "explorer.exe", 6: "WindowsTerminal.exe"},
    )

    with pytest.raises(WindowNotFoundError):
        locator.locate()

    assert process_resolver.requested_handles == [4, 5, 6]


def test_no_valid_candidate_raises_window_not_found() -> None:
    """Fail closed when candidates are hidden, unrelated, or unreadable."""

    locator, process_resolver = create_locator(
        {
            7: FakeWindow(title="BlueStacks hidden", visible=False),
            8: FakeWindow(title="Unrelated visible window"),
            9: FakeWindow(title="BlueStacks unreadable process"),
        },
        {7: "HD-Player.exe", 8: "HD-Player.exe", 9: None},
    )

    with pytest.raises(WindowNotFoundError):
        locator.locate()

    assert process_resolver.requested_handles == [9]
