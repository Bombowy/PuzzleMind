"""Tests for explicit Cats screen-state debug rendering and persistence."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from logicforge.infrastructure.opencv_cats_screen_state_detector import (
    OpenCvCatsScreenStateDetector,
)
from logicforge.infrastructure.opencv_cats_screen_state_renderer import (
    CatsScreenStateDebugRenderError,
    OpenCvCatsScreenStateDebugRenderer,
)
from logicforge.plugins.cats import CatsScreenState
from logicforge.vision.screenshot import Screenshot
from synthetic_cats_screen_states import (
    synthetic_bluestacks_window,
    synthetic_board_screen,
    synthetic_level_complete_screen,
    synthetic_ranking_screen,
    synthetic_unknown_screen,
)
from synthetic_vision import screenshot_from_image


def test_render_returns_new_writable_matrix() -> None:
    """Return an independently writable BGR overlay at source resolution."""

    screenshot = synthetic_level_complete_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)

    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    assert overlay is not screenshot.image
    assert overlay.shape == screenshot.image.shape
    assert overlay.dtype == np.uint8
    assert overlay.flags.writeable


def test_render_does_not_mutate_screenshot() -> None:
    """Preserve the immutable screenshot while drawing every diagnostic layer."""

    screenshot = synthetic_ranking_screen()
    expected = screenshot.image.copy()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)

    OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    assert np.array_equal(screenshot.image, expected)


def test_level_complete_draws_candidate_and_action_point() -> None:
    """Annotate both warm-CTA geometry and its exact detected action."""

    screenshot = synthetic_level_complete_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)
    button = detection.diagnostics.level_button_candidate
    action = detection.action_point

    assert button is not None
    assert action is not None
    assert not np.array_equal(
        overlay[
            button.y : button.y + button.height, button.x : button.x + button.width
        ],
        screenshot.image[
            button.y : button.y + button.height,
            button.x : button.x + button.width,
        ],
    )
    assert not np.array_equal(
        overlay[action.y, action.x], screenshot.image[action.y, action.x]
    )


def test_ranking_draws_every_card() -> None:
    """Annotate every rectangle in the selected best vertical stack."""

    screenshot = synthetic_ranking_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    for card in detection.diagnostics.ranking_card_candidates:
        assert not np.array_equal(
            overlay[card.y, card.x], screenshot.image[card.y, card.x]
        )


def test_board_draws_detected_board_rectangle() -> None:
    """Visualize the existing delegated board candidate for BOARD state."""

    screenshot = synthetic_board_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)
    board = detection.diagnostics.board_candidate

    assert board is not None
    assert not np.array_equal(
        overlay[board.y, board.x], screenshot.image[board.y, board.x]
    )


def test_unknown_still_draws_state_label() -> None:
    """Produce useful state and score text even without geometric candidates."""

    screenshot = synthetic_unknown_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    assert not np.array_equal(overlay, screenshot.image)


def test_debug_false_creates_no_file_or_directory(tmp_path: Path) -> None:
    """Keep normal disabled rendering entirely free of filesystem side effects."""

    screenshot = synthetic_board_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    output_path = tmp_path / "missing" / "cats_screen_state.png"

    result = OpenCvCatsScreenStateDebugRenderer().save_debug_overlay(
        screenshot,
        detection,
        output_path,
        debug=False,
    )

    assert result is None
    assert not output_path.exists()
    assert not output_path.parent.exists()


def test_debug_true_saves_readable_png(tmp_path: Path) -> None:
    """Encode one explicit overlay that OpenCV can read at source resolution."""

    screenshot = synthetic_ranking_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    output_path = tmp_path / "vision" / "cats_screen_state.png"

    saved = OpenCvCatsScreenStateDebugRenderer().save_debug_overlay(
        screenshot,
        detection,
        output_path,
        debug=True,
    )
    loaded = cv2.imread(str(output_path), cv2.IMREAD_COLOR)

    assert saved == output_path.resolve()
    assert loaded is not None
    assert loaded.shape == screenshot.image.shape


def test_failed_encoding_raises_typed_render_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fail visibly when OpenCV cannot persist the requested diagnostic PNG."""

    screenshot = synthetic_unknown_screen()
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    monkeypatch.setattr(cv2, "imwrite", lambda path, image: False)

    with pytest.raises(CatsScreenStateDebugRenderError, match="could not encode"):
        OpenCvCatsScreenStateDebugRenderer().save_debug_overlay(
            screenshot,
            detection,
            tmp_path / "cats_screen_state.png",
            debug=True,
        )


def _full_window(
    state: CatsScreenState,
    *,
    aligned: bool = True,
) -> Screenshot:
    """Create a 916x1032 window with global viewport geometry."""

    return synthetic_bluestacks_window(
        screenshot_width=916,
        screenshot_height=1032,
        viewport_x=321,
        viewport_y=33,
        viewport_width=562,
        viewport_height=999,
        state=state,
        aligned=aligned,
    )


def test_viewport_is_drawn_in_full_window_coordinates() -> None:
    """Annotate the selected game boundary independently from state geometry."""

    screenshot = _full_window(CatsScreenState.LEVEL_COMPLETE)
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)
    viewport = detection.diagnostics.game_viewport_candidate

    assert viewport is not None
    assert not np.array_equal(
        overlay[viewport.y, viewport.x],
        screenshot.image[viewport.y, viewport.x],
    )


def test_unknown_still_draws_detected_viewport_and_unaccepted_cards() -> None:
    """Retain useful viewport/card evidence when final stack alignment fails."""

    screenshot = _full_window(CatsScreenState.RANKING, aligned=False)
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)
    viewport = detection.diagnostics.game_viewport_candidate
    cards = detection.diagnostics.ranking_card_candidates

    assert detection.state is CatsScreenState.UNKNOWN
    assert viewport is not None
    assert cards
    assert not np.array_equal(
        overlay[viewport.y, viewport.x],
        screenshot.image[viewport.y, viewport.x],
    )
    for card in cards:
        assert not np.array_equal(
            overlay[card.y, card.x],
            screenshot.image[card.y, card.x],
        )


def test_rejected_level_candidate_is_drawn() -> None:
    """Draw the best orange component even when local geometry rejects it."""

    base = _full_window(CatsScreenState.RANKING, aligned=False)
    image = base.image.copy()
    cv2.rectangle(image, (450, 830), (540, 865), (0, 145, 255), cv2.FILLED)
    screenshot = screenshot_from_image(image)
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    candidate = detection.diagnostics.level_button_candidate
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    assert detection.state is CatsScreenState.UNKNOWN
    assert candidate is not None
    assert not np.array_equal(
        overlay[candidate.y, candidate.x], screenshot.image[candidate.y, candidate.x]
    )


def test_global_action_point_is_drawn_without_retranslation() -> None:
    """Use the already-global transition action coordinate on the full overlay."""

    screenshot = _full_window(CatsScreenState.RANKING)
    detection = OpenCvCatsScreenStateDetector().detect(screenshot)
    action = detection.action_point
    overlay = OpenCvCatsScreenStateDebugRenderer().render(screenshot, detection)

    assert action is not None
    assert action.x > 321
    assert not np.array_equal(
        overlay[action.y, action.x],
        screenshot.image[action.y, action.x],
    )
