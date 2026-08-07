"""Deterministic tests for Cats transition and board-state classification."""

from dataclasses import fields, is_dataclass
from inspect import getsource

import cv2
import numpy as np
import pytest

from logicforge.infrastructure import (
    opencv_cats_screen_state_detector as detector_module,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionError,
    CatsScreenStateDetectionSettings,
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_tile_grid_detector import (
    OpenCvCatsTileGridDetector,
)
from logicforge.plugins.cats import (
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
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
    synthetic_board_screen,
    synthetic_level_complete_screen,
    synthetic_ranking_screen,
    synthetic_unknown_screen,
)
from synthetic_cats_tile_grids import synthetic_cats_tile_grid
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


@pytest.mark.parametrize(
    ("point", "rectangle"),
    (
        ((-1, 0), (0, 0, 1, 1)),
        ((0, -1), (0, 0, 1, 1)),
        ((0, 0), (-1, 0, 1, 1)),
        ((0, 0), (0, -1, 1, 1)),
        ((0, 0), (0, 0, 0, 1)),
        ((0, 0), (0, 0, 1, 0)),
    ),
)
def test_public_geometry_rejects_invalid_screenshot_coordinates(
    point: tuple[int, int],
    rectangle: tuple[int, int, int, int],
) -> None:
    """Reject negative positions and empty rectangles at the public boundary."""

    if min(point) < 0:
        with pytest.raises(ValueError):
            CatsScreenPoint(*point)
    if min(rectangle[:2]) < 0 or min(rectangle[2:]) <= 0:
        with pytest.raises(ValueError):
            CatsScreenRect(*rectangle)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf")])
def test_public_diagnostics_reject_invalid_scores(score: float) -> None:
    """Require finite unit-interval evidence in plugin-facing diagnostics."""

    with pytest.raises(ValueError):
        CatsScreenStateDiagnostics(
            game_viewport_candidate=None,
            game_viewport_score=0.0,
            level_button_candidate=None,
            level_button_score=score,
            ranking_card_candidates=(),
            ranking_score=0.0,
            board_candidate=None,
            board_confidence=None,
            grid_confidence=None,
            detected_rows=None,
            detected_columns=None,
            rejection_reasons=(),
        )

    with pytest.raises(ValueError):
        CatsScreenStateDiagnostics(
            game_viewport_candidate=None,
            game_viewport_score=score,
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


def test_public_detection_enforces_action_point_semantics() -> None:
    """Require actions only for transition states and never for BOARD/UNKNOWN."""

    diagnostics = _empty_diagnostics()
    point = CatsScreenPoint(10, 20)

    with pytest.raises(ValueError):
        CatsScreenStateDetection(
            CatsScreenState.LEVEL_COMPLETE,
            0.8,
            None,
            diagnostics,
        )
    with pytest.raises(ValueError):
        CatsScreenStateDetection(CatsScreenState.RANKING, 0.8, None, diagnostics)
    with pytest.raises(ValueError):
        CatsScreenStateDetection(CatsScreenState.BOARD, 0.8, point, diagnostics)
    with pytest.raises(ValueError):
        CatsScreenStateDetection(CatsScreenState.UNKNOWN, 0.0, point, diagnostics)
    with pytest.raises(ValueError):
        CatsScreenStateDetection(CatsScreenState.UNKNOWN, 0.1, None, diagnostics)


def test_settings_reject_non_finite_aspect_ratio() -> None:
    """Keep every scale-independent geometry threshold finite."""

    with pytest.raises(ValueError):
        CatsScreenStateDetectionSettings(level_button_maximum_aspect_ratio=float("inf"))


def test_warm_cta_setting_names_preserve_live_calibrated_values() -> None:
    """Rename red/orange evidence without changing any calibrated threshold."""

    settings = CatsScreenStateDetectionSettings()

    assert settings.level_warm_cta_hue_minimum == 0
    assert settings.level_warm_cta_hue_maximum == 28
    assert settings.level_warm_cta_red_wrap_hue_minimum == 170
    assert settings.level_warm_cta_saturation_minimum == 145
    assert settings.level_warm_cta_value_minimum == 120
    assert settings.level_button_minimum_warm_fill_ratio == 0.48


@pytest.mark.parametrize(
    "overrides",
    (
        {"ranking_maximum_width_variation": float("nan")},
        {"ranking_maximum_gap_coefficient_of_variation": float("inf")},
        {"level_morphology_kernel_ratio": 0.0},
        {"ranking_morphology_horizontal_kernel_ratio": 0.0},
        {"ranking_maximum_edge_alignment_ratio": 0.0},
        {"ranking_maximum_gap_coefficient_of_variation": 0.0},
    ),
)
def test_settings_reject_invalid_relative_tolerances(
    overrides: dict[str, float],
) -> None:
    """Reject non-finite tolerances and zero values used as score divisors."""

    with pytest.raises(ValueError):
        CatsScreenStateDetectionSettings(**overrides)  # type: ignore[arg-type]


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


def test_large_lower_orange_button_is_level_complete() -> None:
    """Recognize the mandatory transition through color and geometry alone."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())

    assert result.state is CatsScreenState.LEVEL_COMPLETE


def test_level_complete_action_is_exact_button_center() -> None:
    """Expose the center of the detected orange rectangle as screenshot action."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())
    button = result.diagnostics.level_button_candidate

    assert button is not None
    assert result.action_point is not None
    assert (result.action_point.x, result.action_point.y) == (
        button.center_x,
        button.center_y,
    )


@pytest.mark.parametrize(
    "rectangle",
    (
        (380, 850, 40, 40),
        (130, 100, 540, 85),
        (100, 860, 600, 16),
    ),
)
def test_small_top_or_very_thin_orange_shapes_are_not_level_complete(
    rectangle: tuple[int, int, int, int],
) -> None:
    """Reject icons, upper banners, and thin strips despite saturated orange."""

    result = OpenCvCatsScreenStateDetector().detect(_orange_shape_screenshot(rectangle))

    assert result.state is not CatsScreenState.LEVEL_COMPLETE


def test_orange_confetti_is_not_level_complete() -> None:
    """Require one large coherent lower component instead of many small accents."""

    image = synthetic_unknown_screen().image.copy()
    for index in range(18):
        x = 35 + (index * 97) % 720
        y = 620 + (index * 53) % 300
        cv2.circle(image, (x, y), 5, ORANGE, cv2.FILLED)

    result = OpenCvCatsScreenStateDetector().detect(screenshot_from_image(image))

    assert result.state is not CatsScreenState.LEVEL_COMPLETE


def test_valid_level_button_is_not_shadowed_by_higher_scoring_invalid_shape() -> None:
    """Select the best accepted component before retaining rejected diagnostics."""

    image = synthetic_unknown_screen().image.copy()
    cv2.rectangle(image, (180, 760), (620, 830), ORANGE, cv2.FILLED)
    cv2.rectangle(image, (50, 850), (750, 940), ORANGE, cv2.FILLED)

    result = OpenCvCatsScreenStateDetector().detect(screenshot_from_image(image))

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.diagnostics.level_button_candidate is not None
    assert result.diagnostics.level_button_candidate.width < 600


@pytest.mark.parametrize("card_count", [2, 3])
def test_two_or_three_aligned_bright_cards_are_ranking(card_count: int) -> None:
    """Accept a vertical stack without requiring exactly three cards or text."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=card_count)
    )

    assert result.state is CatsScreenState.RANKING
    assert len(result.diagnostics.ranking_card_candidates) == card_count


def test_one_bright_card_is_not_ranking() -> None:
    """Reject a single modal card as insufficient stack evidence."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=1)
    )

    assert result.state is not CatsScreenState.RANKING
    assert "only one viewport-relative ranking card was accepted" in (
        result.diagnostics.rejection_reasons
    )


def test_unaligned_bright_rectangles_are_not_ranking() -> None:
    """Reject multiple cards that do not share a sufficiently aligned stack."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_ranking_screen(card_count=3, aligned=False)
    )

    assert result.state is not CatsScreenState.RANKING


def test_ranking_action_is_below_stack_and_inside_screenshot() -> None:
    """Derive a safe lower action from stack union rather than fixed pixels."""

    screenshot = synthetic_ranking_screen()
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    action = result.action_point
    cards = result.diagnostics.ranking_card_candidates

    assert action is not None
    assert action.y > max(card.y + card.height for card in cards)
    assert 0 <= action.x < screenshot.width
    assert 0 <= action.y < screenshot.height


def test_ranking_action_scales_with_resolution() -> None:
    """Keep action ratios stable when no absolute screenshot size is assumed."""

    small = synthetic_ranking_screen(width=480, height=720)
    large = synthetic_ranking_screen(width=960, height=1440)
    small_action = OpenCvCatsScreenStateDetector().detect(small).action_point
    large_action = OpenCvCatsScreenStateDetector().detect(large).action_point

    assert small_action is not None
    assert large_action is not None
    assert small_action.x / small.width == pytest.approx(
        large_action.x / large.width,
        abs=0.01,
    )
    assert small_action.y / small.height == pytest.approx(
        large_action.y / large.height,
        abs=0.01,
    )


def test_regular_grid_screen_is_board_without_action() -> None:
    """Delegate ordinary board recognition to the existing detector pair."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_board_screen())

    assert result.state is CatsScreenState.BOARD
    assert result.action_point is None
    assert result.diagnostics.board_candidate is not None
    assert result.diagnostics.detected_rows == 6
    assert result.diagnostics.detected_columns == 6


def test_cats_tile_lattice_is_board_without_reliable_outer_contour() -> None:
    """Use tile-grid-first geometry for a stable Cats-like board by default."""

    screenshot = synthetic_cats_tile_grid(
        rows=9,
        columns=9,
        pastel_outer_column=True,
    ).screenshot

    result = OpenCvCatsScreenStateDetector().detect(screenshot)

    assert result.state is CatsScreenState.BOARD
    assert result.diagnostics.detected_rows == 9
    assert result.diagnostics.detected_columns == 9


def test_successful_tile_grid_does_not_invoke_generic_board_fallback() -> None:
    """Never let contour fallback replace a complete primary Cats lattice."""

    board_detector = _FakeBoardDetector(fail=True)
    grid_detector = _FakeGridDetector(fail=True)
    screenshot = synthetic_cats_tile_grid(rows=9, columns=9).screenshot
    detector = OpenCvCatsScreenStateDetector(
        tile_grid_detector=OpenCvCatsTileGridDetector(),
        board_detector=board_detector,
        grid_detector=grid_detector,
    )

    result = detector.detect(screenshot)

    assert result.state is CatsScreenState.BOARD
    assert board_detector.calls == 0
    assert grid_detector.calls == 0


def test_empty_unknown_screen_returns_unknown_without_action() -> None:
    """Return a normal zero-confidence result when no known evidence exists."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert result.confidence == 0.0
    assert result.action_point is None


def test_ranking_visible_over_board_has_priority_over_board() -> None:
    """Classify the modal stack before the partially visible background board."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_ranking_screen())

    assert result.state is CatsScreenState.RANKING


def test_level_complete_visible_over_board_has_priority() -> None:
    """Classify the orange transition before any background grid evidence."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_level_complete_screen())

    assert result.state is CatsScreenState.LEVEL_COMPLETE


def test_level_complete_has_priority_over_ranking() -> None:
    """Prefer the characteristic lower button when both overlays are present."""

    result = OpenCvCatsScreenStateDetector().detect(
        synthetic_level_complete_screen(include_ranking_cards=True)
    )

    assert result.state is CatsScreenState.LEVEL_COMPLETE


def test_overlay_states_do_not_invoke_injected_board_detectors() -> None:
    """Prove both transition checks run before delegated BOARD analysis."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    detector = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    )

    assert detector.detect(synthetic_ranking_screen()).state is CatsScreenState.RANKING
    assert (
        detector.detect(synthetic_level_complete_screen()).state
        is CatsScreenState.LEVEL_COMPLETE
    )
    assert board_detector.calls == 0
    assert grid_detector.calls == 0


def test_all_confidences_and_scores_are_in_unit_interval() -> None:
    """Keep transition and board evidence safe for public diagnostics."""

    detector = OpenCvCatsScreenStateDetector()
    results = (
        detector.detect(synthetic_board_screen()),
        detector.detect(synthetic_ranking_screen()),
        detector.detect(synthetic_level_complete_screen()),
        detector.detect(synthetic_unknown_screen()),
    )

    for result in results:
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.diagnostics.level_button_score <= 1.0
        assert 0.0 <= result.diagnostics.ranking_score <= 1.0


def test_repeated_analysis_is_exactly_deterministic() -> None:
    """Return equal immutable diagnostics independently of contour ordering."""

    screenshot = synthetic_ranking_screen(include_bubble=True)
    detector = OpenCvCatsScreenStateDetector()

    results = tuple(detector.detect(screenshot) for _ in range(3))

    assert results[0] == results[1] == results[2]


def _assert_no_backend_objects(value: object) -> None:
    """Recursively reject NumPy matrices from public diagnostic structures."""

    assert not isinstance(value, np.ndarray)
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _assert_no_backend_objects(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_no_backend_objects(item)


def test_diagnostics_contain_no_opencv_or_numpy_objects() -> None:
    """Expose only frozen primitive plugin models across the adapter boundary."""

    result = OpenCvCatsScreenStateDetector().detect(synthetic_ranking_screen())

    _assert_no_backend_objects(result.diagnostics)


def test_detection_does_not_mutate_screenshot() -> None:
    """Treat immutable source pixels as read-only throughout all analyses."""

    screenshot = synthetic_level_complete_screen(include_ranking_cards=True)
    expected = screenshot.image.copy()

    OpenCvCatsScreenStateDetector().detect(screenshot)

    assert np.array_equal(screenshot.image, expected)


@pytest.mark.parametrize(
    ("width", "height", "factory", "expected_state"),
    (
        (400, 720, synthetic_level_complete_screen, CatsScreenState.LEVEL_COMPLETE),
        (810, 1440, synthetic_ranking_screen, CatsScreenState.RANKING),
        (700, 700, synthetic_board_screen, CatsScreenState.BOARD),
        (500, 900, synthetic_board_screen, CatsScreenState.BOARD),
    ),
)
def test_detector_supports_multiple_resolutions_and_portrait_ratios(
    width: int,
    height: int,
    factory: object,
    expected_state: CatsScreenState,
) -> None:
    """Use relative thresholds across square and BlueStacks-like portrait frames."""

    screenshot_factory = factory
    assert callable(screenshot_factory)
    screenshot = screenshot_factory(width=width, height=height)

    assert OpenCvCatsScreenStateDetector().detect(screenshot).state is expected_state


def test_injected_board_and_grid_detectors_produce_board() -> None:
    """Compose the public ports without requiring their concrete OpenCV adapters."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    result = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.BOARD
    assert result.confidence == 0.82
    assert board_detector.calls == 1
    assert grid_detector.calls == 1


def test_board_detector_rejection_becomes_unknown_not_exception() -> None:
    """Treat ordinary absence of a board as a classification outcome."""

    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(fail=True),
        grid_detector=_FakeGridDetector(),
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert "board detector rejected the screenshot" in (
        result.diagnostics.rejection_reasons
    )


def test_grid_detector_rejection_becomes_unknown_not_exception() -> None:
    """Retain the board candidate but fail closed when public grid evidence fails."""

    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(),
        grid_detector=_FakeGridDetector(fail=True),
    ).detect(synthetic_unknown_screen())

    assert result.state is CatsScreenState.UNKNOWN
    assert result.diagnostics.board_candidate is not None
    assert "grid detector rejected the board" in result.diagnostics.rejection_reasons


def test_opencv_board_processing_failure_raises_typed_detection_error() -> None:
    """Distinguish an unusable backend from an ordinary UNKNOWN screen."""

    detector = OpenCvCatsScreenStateDetector(
        board_detector=_OpenCvFailingBoardDetector(),
        grid_detector=_FakeGridDetector(),
    )

    with pytest.raises(CatsScreenStateDetectionError):
        detector.detect(synthetic_unknown_screen())


def test_detector_source_uses_no_ocr_template_or_fixed_resolution() -> None:
    """Keep classification geometric, color-based, and scale-relative."""

    source = getsource(detector_module).casefold()

    for forbidden in (
        "pytesseract",
        "easyocr",
        "matchtemplate",
        "template matching",
        "800x",
        "1000x",
    ):
        assert forbidden not in source


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


@pytest.mark.parametrize(
    ("screenshot_width", "viewport_x", "left_ad"),
    ((916, 321, True), (1920, 679, False)),
)
def test_detects_game_viewport_in_wide_bluestacks_windows(
    screenshot_width: int,
    viewport_x: int,
    left_ad: bool,
) -> None:
    """Locate the game strip for both observed live window proportions."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(
            CatsScreenState.LEVEL_COMPLETE,
            screenshot_width=screenshot_width,
            viewport_x=viewport_x,
            left_ad=left_ad,
        )
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x == pytest.approx(viewport_x, abs=25)
    assert viewport.y == pytest.approx(33, abs=3)
    assert viewport.width == pytest.approx(562, abs=30)
    assert viewport.height == pytest.approx(999, abs=3)
    assert viewport.width / viewport.height == pytest.approx(9 / 16, abs=0.035)
    assert 0.0 <= result.diagnostics.game_viewport_score <= 1.0


def test_viewport_uses_full_screenshot_coordinates_and_excludes_side_ui() -> None:
    """Keep the ad left of and toolbar right of the selected viewport."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.LEVEL_COMPLETE)
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x >= 321 - 25
    assert viewport.x + viewport.width <= 321 + 562 + 25
    assert viewport.y > 0


def test_viewport_detection_is_deterministic_and_does_not_mutate_pixels() -> None:
    """Return identical global geometry while preserving the immutable capture."""

    screenshot = _bluestacks_window(CatsScreenState.RANKING)
    expected = screenshot.image.copy()
    detector = OpenCvCatsScreenStateDetector()

    results = tuple(detector.detect(screenshot) for _ in range(3))

    assert results[0] == results[1] == results[2]
    assert np.array_equal(screenshot.image, expected)


def test_narrow_full_content_viewport_uses_fallback() -> None:
    """Accept whole portrait content without two better side boundaries."""

    screenshot = synthetic_level_complete_screen(width=560, height=1000)
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert viewport is not None
    assert viewport.x == 0
    assert viewport.width == screenshot.width


def test_viewport_at_left_edge_is_supported() -> None:
    """Allow one screenshot edge to be a legitimate viewport boundary."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        viewport_x=0,
        left_ad=False,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert viewport is not None
    assert viewport.x == 0


def test_advertisement_and_toolbar_do_not_win_viewport_selection() -> None:
    """Reject complex ad content and a narrow toolbar as game viewports."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        screenshot_width=1920,
        viewport_x=679,
        left_ad=True,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert viewport.x == pytest.approx(679, abs=25)
    assert viewport.width > 500
    assert viewport.x + viewport.width < 1900


def test_no_sensible_viewport_returns_none_and_specific_reason() -> None:
    """Do not run overlay geometry against a landscape full screenshot."""

    screenshot = synthetic_unknown_screen(width=1200, height=500)
    result = OpenCvCatsScreenStateDetector(
        board_detector=_FakeBoardDetector(fail=True),
        grid_detector=_FakeGridDetector(),
    ).detect(screenshot)

    assert result.diagnostics.game_viewport_candidate is None
    assert "no reliable Cats game viewport was found" in (
        result.diagnostics.rejection_reasons
    )


def test_missing_viewport_does_not_block_full_screenshot_board_attempt() -> None:
    """Keep BOARD independent from transition-screen viewport availability."""

    board_detector = _FakeBoardDetector()
    grid_detector = _FakeGridDetector()
    result = OpenCvCatsScreenStateDetector(
        board_detector=board_detector,
        grid_detector=grid_detector,
    ).detect(synthetic_unknown_screen(width=1200, height=500))

    assert result.state is CatsScreenState.BOARD
    assert result.diagnostics.game_viewport_candidate is None
    assert board_detector.calls == 1
    assert grid_detector.calls == 1


@pytest.mark.parametrize(
    "overrides",
    (
        {"viewport_minimum_height_ratio": 0.0},
        {"viewport_boundary_probe_ratio": 0.0},
        {"viewport_boundary_smoothing_ratio": 0.0},
        {"viewport_minimum_score": float("nan")},
        {"viewport_minimum_aspect_ratio": 0.57},
        {"viewport_preferred_aspect_ratio": 0.47},
        {"viewport_maximum_aspect_ratio": float("inf")},
        {"viewport_content_top_maximum_ratio": 0.11},
    ),
)
def test_viewport_settings_are_fully_validated(overrides: dict[str, float]) -> None:
    """Reject non-finite, non-positive, and unordered viewport settings."""

    with pytest.raises(ValueError):
        CatsScreenStateDetectionSettings(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("screenshot_width", "viewport_x", "left_ad"),
    ((916, 321, True), (1920, 679, False)),
)
def test_live_proportioned_level_button_is_viewport_relative(
    screenshot_width: int,
    viewport_x: int,
    left_ad: bool,
) -> None:
    """Recognize a 70%-viewport button below 45% of the full window width."""

    screenshot = _bluestacks_window(
        CatsScreenState.LEVEL_COMPLETE,
        screenshot_width=screenshot_width,
        viewport_x=viewport_x,
        left_ad=left_ad,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate
    button = result.diagnostics.level_button_candidate

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.confidence >= 0.60
    assert viewport is not None
    assert button is not None
    assert button.width / screenshot.width < 0.45
    assert button.width / viewport.width > 0.65
    assert result.action_point == CatsScreenPoint(button.center_x, button.center_y)


def test_orange_advertisement_element_cannot_trigger_level_complete() -> None:
    """Ignore a large orange ad component lying outside the selected game crop."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING)
    )
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.RANKING
    assert viewport is not None
    assert result.diagnostics.level_button_candidate is None


@pytest.mark.parametrize("card_count", [2, 3])
def test_full_window_ranking_cards_are_viewport_relative(card_count: int) -> None:
    """Recognize interrupted neutral, cream, and peach cards inside the game."""

    screenshot = _bluestacks_window(
        CatsScreenState.RANKING,
        card_count=card_count,
    )
    result = OpenCvCatsScreenStateDetector().detect(screenshot)
    viewport = result.diagnostics.game_viewport_candidate

    assert result.state is CatsScreenState.RANKING
    assert viewport is not None
    assert len(result.diagnostics.ranking_card_candidates) == card_count
    assert result.diagnostics.ranking_score >= 0.64
    assert all(
        card.width / viewport.width > 0.74
        for card in result.diagnostics.ranking_card_candidates
    )
    assert result.action_point is not None
    assert result.action_point.y > max(
        card.y + card.height for card in result.diagnostics.ranking_card_candidates
    )
    assert viewport.x <= result.action_point.x < viewport.x + viewport.width


def test_one_or_unaligned_full_window_card_stack_is_not_ranking() -> None:
    """Require at least two aligned viewport-relative cards."""

    one = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING, card_count=1)
    )
    unaligned = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.RANKING, aligned=False)
    )

    assert one.state is not CatsScreenState.RANKING
    assert unaligned.state is not CatsScreenState.RANKING


def test_bright_ad_rectangles_do_not_trigger_ranking() -> None:
    """Do not count the synthetic ad's light rectangles as ranking cards."""

    result = OpenCvCatsScreenStateDetector().detect(
        _bluestacks_window(CatsScreenState.LEVEL_COMPLETE)
    )

    assert result.state is CatsScreenState.LEVEL_COMPLETE
    assert result.diagnostics.ranking_card_candidates == ()


def test_detector_contains_no_live_resolution_or_coordinate_constants() -> None:
    """Keep production geometry proportional and independent from regression sizes."""

    source = getsource(detector_module)

    assert "916" not in source
    assert "1920" not in source
    assert "y=32" not in source
