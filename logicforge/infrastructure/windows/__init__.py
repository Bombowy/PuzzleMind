"""Windows-only adapters used by the BlueStacks capture composition."""

from logicforge.infrastructure.windows.bluestacks_locator import (
    Win32BlueStacksWindowLocator,
)
from logicforge.infrastructure.windows.mss_window_capture import MssWindowCapturer

__all__ = ["MssWindowCapturer", "Win32BlueStacksWindowLocator"]
