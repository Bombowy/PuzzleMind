"""Puzzle-specific contracts for classifying visible Cats application screens."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from logicforge.vision.screenshot import Screenshot


def _validate_score(name: str, value: float) -> None:
    """Reject non-finite confidence values outside the inclusive unit interval."""

    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within 0.0 and 1.0.")


class CatsScreenState(StrEnum):
    """Identify the puzzle and transition screens relevant to future automation."""

    BOARD = "board"
    RANKING = "ranking"
    LEVEL_COMPLETE = "level_complete"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CatsScreenPoint:
    """Describe one safe future action point in screenshot pixel coordinates."""

    x: int
    y: int

    def __post_init__(self) -> None:
        """Keep plugin-level coordinates inside the non-negative image plane."""

        if self.x < 0 or self.y < 0:
            raise ValueError("Cats screen point coordinates must be non-negative.")


@dataclass(frozen=True, slots=True)
class CatsScreenRect:
    """Describe one half-open rectangle in screenshot pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject negative positions and empty or inverted geometry."""

        if self.x < 0 or self.y < 0:
            raise ValueError("Cats screen rectangle coordinates must be non-negative.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Cats screen rectangle dimensions must be positive.")

    @property
    def center_x(self) -> int:
        """Return the deterministic integer center within the half-open rectangle."""

        return self.x + self.width // 2

    @property
    def center_y(self) -> int:
        """Return the deterministic integer center within the half-open rectangle."""

        return self.y + self.height // 2


@dataclass(frozen=True, slots=True)
class CatsScreenStateDiagnostics:
    """Carry primitive classification evidence without OpenCV-specific objects."""

    game_viewport_candidate: CatsScreenRect | None
    game_viewport_score: float
    level_button_candidate: CatsScreenRect | None
    level_button_score: float
    ranking_card_candidates: tuple[CatsScreenRect, ...]
    ranking_score: float
    board_candidate: CatsScreenRect | None
    board_confidence: float | None
    grid_confidence: float | None
    detected_rows: int | None
    detected_columns: int | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate every public score and optional grid dimension consistently."""

        _validate_score("game_viewport_score", self.game_viewport_score)
        _validate_score("level_button_score", self.level_button_score)
        _validate_score("ranking_score", self.ranking_score)
        if self.board_confidence is not None:
            _validate_score("board_confidence", self.board_confidence)
        if self.grid_confidence is not None:
            _validate_score("grid_confidence", self.grid_confidence)
        if (self.detected_rows is None) != (self.detected_columns is None):
            raise ValueError("Detected rows and columns must be provided together.")
        if self.detected_rows is not None and self.detected_rows <= 0:
            raise ValueError("Detected rows must be positive when provided.")
        if self.detected_columns is not None and self.detected_columns <= 0:
            raise ValueError("Detected columns must be positive when provided.")


@dataclass(frozen=True, slots=True)
class CatsScreenStateDetection:
    """Return one classified Cats screen and an optional screenshot-space action."""

    state: CatsScreenState
    confidence: float
    action_point: CatsScreenPoint | None
    diagnostics: CatsScreenStateDiagnostics

    def __post_init__(self) -> None:
        """Enforce action availability and confidence semantics for every state."""

        _validate_score("confidence", self.confidence)
        if (
            self.state
            in (
                CatsScreenState.LEVEL_COMPLETE,
                CatsScreenState.RANKING,
            )
            and self.action_point is None
        ):
            raise ValueError(f"{self.state.name} detection requires an action point.")
        if self.state in (CatsScreenState.BOARD, CatsScreenState.UNKNOWN) and (
            self.action_point is not None
        ):
            raise ValueError(
                f"{self.state.name} detection cannot have an action point."
            )
        if self.state is CatsScreenState.UNKNOWN and self.confidence != 0.0:
            raise ValueError("UNKNOWN detection confidence must be exactly 0.0.")


class CatsScreenStateDetector(ABC):
    """Define a plugin-facing port for one-shot Cats screen classification."""

    @abstractmethod
    def detect(self, screenshot: Screenshot) -> CatsScreenStateDetection:
        """Classify one immutable screenshot without mutating or interacting."""

        raise NotImplementedError
