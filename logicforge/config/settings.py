"""Dependency-free settings records for framework composition."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VisionSettings:
    """Hold tunable, puzzle-neutral parameters for future vision adapters.

    TODO: Add detector thresholds and calibration profiles in v0.2 after concrete
    vision backends establish units, valid ranges, and serialization requirements.
    """

    debug_artifacts_directory: Path = Path("artifacts/vision")


@dataclass(frozen=True, slots=True)
class SolverSettings:
    """Hold safety limits for future deterministic solver orchestration.

    TODO: Add iteration and timeout limits in v0.4 together with explicit failure
    results, cancellation behavior, and reproducibility guarantees.
    """

    explanations_enabled: bool = True


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    """Describe logging policy without configuring process-global state.

    TODO: Add structured output, redaction, and correlation-id options when the
    application composition layer introduces an operational logging adapter.
    """

    level: str = "INFO"


@dataclass(frozen=True, slots=True)
class LogicForgeSettings:
    """Aggregate immutable settings passed through the composition root.

    Nested settings keep subsystems independent and make future configuration
    sources replaceable without making domain code aware of environment variables.

    TODO: Add an external settings loader after precedence rules for files,
    environment variables, and command-line options are documented and tested.
    """

    vision: VisionSettings = field(default_factory=VisionSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
