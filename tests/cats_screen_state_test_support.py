"""Shared fixtures for Cats screen-state detector tests."""

from dataclasses import fields, is_dataclass

import cv2
import numpy as np

from logicforge.plugins.cats import (
    CatsScreenState,
    CatsScreenStateDiagnostics,
)
from logicforge.vision.board_detector import (
    BoardDetection,
    BoardDetectionDiagnostics,
    BoardDetectionError,
    BoardDetector,
)
from logicforge.vision.grid_detector import (
    CellBounds,
    GridDetection,
    GridDetectionDiagnostics,
    GridDetectionError,
    GridDetector,
)
from logicforge.vision.screenshot import Screenshot
from synthetic_cats_screen_states import (
    synthetic_bluestacks_window,
    synthetic_unknown_screen,
)
from synthetic_vision import screenshot_from_image

ORANGE = (0, 145, 255)


def _empty_diagnostics() -> CatsScreenStateDiagnostics:
    """Build valid empty diagnostics for focused public-contract tests."""

    return CatsScreenStateDiagnostics(
        game_viewport_candidate=None,
        game_viewport_score=0.0,
        level_button_candidate=None,
        level_button_score=0.0,
        ranking_card_candidates=(),
        ranking_score=0.0,
        board_candidate=None,
        board_confidence=None,
        grid_confidence=None,
        detected_rows=None,
        detected_columns=None,
        rejection_reasons=(),
    )


def _orange_shape_screenshot(
    rectangle: tuple[int, int, int, int],
    *,
    width: int = 800,
    height: int = 1000,
) -> Screenshot:
    """Draw one controlled orange rectangle on an otherwise unknown screen."""

    image = synthetic_unknown_screen(width=width, height=height).image.copy()
    x, y, shape_width, shape_height = rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + shape_width, y + shape_height),
        ORANGE,
        cv2.FILLED,
    )
    return screenshot_from_image(image)


def _board_error() -> BoardDetectionError:
    """Build a typed board rejection for injected fail-closed tests."""

    return BoardDetectionError(
        "synthetic board rejection",
        BoardDetectionDiagnostics(
            contour_count=0,
            candidates=(),
            selected_candidate=None,
            competitive_candidate_count=0,
        ),
    )


def _grid_error() -> GridDetectionError:
    """Build a typed grid rejection for injected fail-closed tests."""

    return GridDetectionError(
        "synthetic grid rejection",
        GridDetectionDiagnostics(
            board_x=100,
            board_y=100,
            board_width=300,
            board_height=300,
            normalized_horizontal_positions=(),
            normalized_vertical_positions=(),
            horizontal_lines=(),
            vertical_lines=(),
            estimated_rows=0,
            estimated_columns=0,
            horizontal_spacing_coefficient_of_variation=1.0,
            vertical_spacing_coefficient_of_variation=1.0,
            horizontal_coverage=0.0,
            vertical_coverage=0.0,
            grid_evidence_score=0.0,
            rejection_reasons=("synthetic rejection",),
        ),
    )


def _fake_grid() -> GridDetection:
    """Return a complete 3x3 public geometry model for injection tests."""

    horizontal = (100, 200, 300, 400)
    vertical = (100, 200, 300, 400)
    cells = tuple(
        CellBounds(
            row=row,
            column=column,
            x=vertical[column],
            y=horizontal[row],
            width=100,
            height=100,
            center_x=vertical[column] + 50,
            center_y=horizontal[row] + 50,
        )
        for row in range(3)
        for column in range(3)
    )
    return GridDetection(
        horizontal_lines=horizontal,
        vertical_lines=vertical,
        rows=3,
        columns=3,
        cells=cells,
        confidence=0.82,
    )


class _FakeBoardDetector(BoardDetector):
    """Return one injected board result or typed rejection."""

    def __init__(self, *, fail: bool = False) -> None:
        """Configure success or fail-closed behavior and call counting."""

        self.fail = fail
        self.calls = 0

    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Record one invocation and return deterministic screenshot geometry."""

        del screenshot
        self.calls += 1
        if self.fail:
            raise _board_error()
        return BoardDetection(x=100, y=100, width=300, height=300, confidence=0.88)


class _FakeGridDetector(GridDetector):
    """Return one injected public grid or typed rejection."""

    def __init__(self, *, fail: bool = False) -> None:
        """Configure success or fail-closed behavior and call counting."""

        self.fail = fail
        self.calls = 0

    def detect(
        self,
        screenshot: Screenshot,
        board: BoardDetection,
    ) -> GridDetection:
        """Record one invocation after accepting public screenshot and board data."""

        del screenshot, board
        self.calls += 1
        if self.fail:
            raise _grid_error()
        return _fake_grid()


class _OpenCvFailingBoardDetector(BoardDetector):
    """Simulate a backend processing failure distinct from ordinary rejection."""

    def detect(self, screenshot: Screenshot) -> BoardDetection:
        """Raise the native OpenCV exception expected at the adapter boundary."""

        del screenshot
        raise cv2.error("synthetic OpenCV failure")


def _assert_no_backend_objects(value: object) -> None:
    """Recursively reject NumPy matrices from public diagnostic structures."""

    assert not isinstance(value, np.ndarray)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_backend_objects(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_no_backend_objects(item)


def _bluestacks_window(
    state: CatsScreenState,
    *,
    screenshot_width: int = 916,
    viewport_x: int = 321,
    left_ad: bool = True,
    right_toolbar: bool = True,
    card_count: int = 3,
    aligned: bool = True,
) -> Screenshot:
    """Create one live-proportioned full BlueStacks window."""

    return synthetic_bluestacks_window(
        screenshot_width=screenshot_width,
        screenshot_height=1032,
        viewport_x=viewport_x,
        viewport_y=33,
        viewport_width=562,
        viewport_height=999,
        state=state,
        left_ad=left_ad,
        right_toolbar=right_toolbar,
        card_count=card_count,
        aligned=aligned,
    )
