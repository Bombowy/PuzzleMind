"""Windows-only adapters used by the BlueStacks capture composition."""

from logicforge.infrastructure.windows.bluestacks_locator import (
    Win32BlueStacksWindowLocator,
)
from logicforge.infrastructure.windows.mss_window_capture import MssWindowCapturer
from logicforge.infrastructure.windows.win32_mouse_controller import (
    MouseAutomationError,
    Win32MouseController,
)

__all__ = [
    "MouseAutomationError",
    "MssWindowCapturer",
    "Win32BlueStacksWindowLocator",
    "Win32MouseController",
]
