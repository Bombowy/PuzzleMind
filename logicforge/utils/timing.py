"""Clock and duration-recording ports for deterministic observability."""

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Protocol


class Clock(Protocol):
    """Supply monotonic timestamps through an injectable, testable dependency.

    TODO: Add a production monotonic-clock adapter and deterministic test clock
    when engine timing and cancellation are implemented.
    """

    def now(self) -> float:
        """Return an implementation-defined monotonic timestamp in seconds.

        TODO: Implement this method in runtime and test adapters; callers must use
        timestamp differences and never interpret the value as wall-clock time.
        """

        ...


class TimingSink(ABC):
    """Receive named operation durations without selecting a metrics backend."""

    @abstractmethod
    def record(self, operation: str, duration: timedelta) -> None:
        """Record one completed duration measurement.

        TODO: Implement log and metrics adapters with bounded labels after stable
        operation names are defined by actual orchestration code.
        """

        raise NotImplementedError
