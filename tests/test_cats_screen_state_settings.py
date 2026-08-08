"""Cats screen-state settings tests."""

import pytest

from cats_screen_state_test_support import (
    _empty_diagnostics,
)
from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    CatsScreenStateDetectionSettings,
)
from logicforge.plugins.cats import (
    CatsScreenPoint,
    CatsScreenRect,
    CatsScreenState,
    CatsScreenStateDetection,
    CatsScreenStateDiagnostics,
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
