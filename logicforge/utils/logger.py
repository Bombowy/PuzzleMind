"""Logging composition boundary."""

from abc import ABC, abstractmethod
from logging import Logger

from logicforge.config.settings import LoggingSettings


class LoggerFactory(ABC):
    """Create configured loggers without leaking global setup into domain modules.

    TODO: Provide a structured standard-library adapter with correlation fields,
    redaction, and idempotent configuration once application startup is defined.
    """

    @abstractmethod
    def create(self, name: str, settings: LoggingSettings) -> Logger:
        """Return a logger configured for one named application component.

        TODO: Implement adapter selection, handler ownership, and repeat-call
        semantics together with the future composition root.
        """

        raise NotImplementedError
