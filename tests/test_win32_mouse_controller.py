"""Unit tests for the Win32 mouse adapter without moving the real pointer."""

from inspect import getsource

import pytest

from logicforge.automation.mouse import MouseButton, ScreenPoint
from logicforge.infrastructure.windows import win32_mouse_controller as mouse_module
from logicforge.infrastructure.windows.win32_mouse_controller import (
    MouseAutomationError,
    Win32MouseController,
)


class _FakeMouseApi:
    """Record the narrow native call surface used by the production adapter."""

    def __init__(
        self,
        *,
        fail_position: bool = False,
        fail_event: bool = False,
    ) -> None:
        """Configure deterministic failures without loading or calling pywin32."""

        self.fail_position = fail_position
        self.fail_event = fail_event
        self.calls: list[tuple[str, object]] = []

    def SetCursorPos(self, point: tuple[int, int]) -> None:
        """Record cursor placement or raise the configured native-like failure."""

        self.calls.append(("position", point))
        if self.fail_position:
            raise OSError("synthetic SetCursorPos failure")

    def mouse_event(
        self,
        flags: int,
        dx: int,
        dy: int,
        data: int,
        extra_info: int,
    ) -> None:
        """Record one transition flag or raise the configured native-like failure."""

        del dx, dy, data, extra_info
        self.calls.append(("event", flags))
        if self.fail_event:
            raise OSError("synthetic mouse_event failure")


def _event_flags(mouse_api: _FakeMouseApi) -> list[int]:
    """Extract emitted event flags while retaining their recorded order."""

    flags: list[int] = []
    for name, value in mouse_api.calls:
        if name != "event":
            continue
        if not isinstance(value, int):
            raise AssertionError("recorded event flag must be an integer")
        flags.append(value)
    return flags


def test_left_click_positions_cursor_at_exact_point() -> None:
    """Pass the absolute ScreenPoint directly to SetCursorPos."""

    mouse_api = _FakeMouseApi()

    Win32MouseController(mouse_api).click(ScreenPoint(321, 654))

    assert mouse_api.calls[0] == ("position", (321, 654))


def test_left_click_emits_down_before_up() -> None:
    """Emit exactly one ordered left-button transition pair."""

    mouse_api = _FakeMouseApi()

    Win32MouseController(mouse_api).click(ScreenPoint(10, 20))

    assert _event_flags(mouse_api) == [
        mouse_module.MOUSEEVENTF_LEFTDOWN,
        mouse_module.MOUSEEVENTF_LEFTUP,
    ]


def test_one_click_emits_exactly_one_down_up_pair() -> None:
    """Keep double-click orchestration outside this single-click adapter."""

    mouse_api = _FakeMouseApi()

    Win32MouseController(mouse_api).click(ScreenPoint(10, 20))

    assert len(_event_flags(mouse_api)) == 2


def test_negative_virtual_desktop_coordinates_are_forwarded() -> None:
    """Support monitors left of or above the primary display."""

    mouse_api = _FakeMouseApi()

    Win32MouseController(mouse_api).click(ScreenPoint(-1200, -50))

    assert mouse_api.calls[0] == ("position", (-1200, -50))


@pytest.mark.parametrize(
    ("button", "expected_flags"),
    [
        (
            MouseButton.RIGHT,
            (
                mouse_module.MOUSEEVENTF_RIGHTDOWN,
                mouse_module.MOUSEEVENTF_RIGHTUP,
            ),
        ),
        (
            MouseButton.MIDDLE,
            (
                mouse_module.MOUSEEVENTF_MIDDLEDOWN,
                mouse_module.MOUSEEVENTF_MIDDLEUP,
            ),
        ),
    ],
)
def test_non_left_buttons_use_matching_win32_flags(
    button: MouseButton,
    expected_flags: tuple[int, int],
) -> None:
    """Map every portable button supported by the existing port."""

    mouse_api = _FakeMouseApi()

    Win32MouseController(mouse_api).click(ScreenPoint(10, 20), button)

    assert _event_flags(mouse_api) == list(expected_flags)


def test_native_api_is_loaded_only_on_first_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructing the adapter alone must not import or resolve pywin32."""

    mouse_api = _FakeMouseApi()
    load_calls: list[None] = []

    def load_mouse_api() -> _FakeMouseApi:
        load_calls.append(None)
        return mouse_api

    monkeypatch.setattr(mouse_module, "_load_mouse_api", load_mouse_api)
    controller = Win32MouseController()

    assert load_calls == []
    controller.click(ScreenPoint(10, 20))
    controller.click(ScreenPoint(30, 40))
    assert load_calls == [None]


def test_injected_api_never_attempts_native_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep deterministic tests entirely behind the injected fake boundary."""

    def reject_native_load() -> _FakeMouseApi:
        raise AssertionError("native API must not be loaded")

    monkeypatch.setattr(mouse_module, "_load_mouse_api", reject_native_load)

    Win32MouseController(_FakeMouseApi()).click(ScreenPoint(10, 20))


def test_native_load_failure_becomes_typed_automation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate an unavailable pywin32 boundary into the public adapter error."""

    def fail_native_load() -> _FakeMouseApi:
        raise ImportError("synthetic win32api import failure")

    monkeypatch.setattr(mouse_module, "_load_mouse_api", fail_native_load)

    with pytest.raises(MouseAutomationError, match=r"left at \(10, 20\)"):
        Win32MouseController().click(ScreenPoint(10, 20))


def test_set_cursor_failure_becomes_typed_automation_error() -> None:
    """Expose a failed pointer move instead of pretending the click succeeded."""

    controller = Win32MouseController(_FakeMouseApi(fail_position=True))

    with pytest.raises(MouseAutomationError, match=r"left at \(10, 20\)"):
        controller.click(ScreenPoint(10, 20))


def test_mouse_event_failure_becomes_typed_automation_error() -> None:
    """Translate a failed native transition through the adapter boundary."""

    controller = Win32MouseController(_FakeMouseApi(fail_event=True))

    with pytest.raises(MouseAutomationError, match=r"right at \(-10, 25\)"):
        controller.click(ScreenPoint(-10, 25), MouseButton.RIGHT)


def test_error_message_contains_coordinates_and_button() -> None:
    """Provide actionable point and button diagnostics for native failures."""

    controller = Win32MouseController(_FakeMouseApi(fail_position=True))

    with pytest.raises(MouseAutomationError) as error_info:
        controller.click(ScreenPoint(55, -77), MouseButton.MIDDLE)

    message = str(error_info.value)
    assert "middle" in message
    assert "(55, -77)" in message


def test_adapter_uses_no_external_mouse_or_subprocess_library() -> None:
    """Keep native execution limited to the requested lazy pywin32 boundary."""

    source = getsource(mouse_module)

    assert "pyautogui" not in source
    assert "pynput" not in source
    assert "subprocess" not in source
