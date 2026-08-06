"""Focused tests for maximal-envelope board debug visualization."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from logicforge.infrastructure.opencv_board_detection_renderer import (
    OpenCvBoardDetectionDebugRenderer,
)
from logicforge.infrastructure.opencv_board_detector import OpenCvBoardDetector
from logicforge.vision.board_detector import BoardDetectionAnalysis
from logicforge.vision.screenshot import Screenshot
from synthetic_vision import truncated_outer_grid_envelope_screenshot


def _refined_case() -> tuple[Screenshot, BoardDetectionAnalysis]:
    """Return one accepted right-side refinement and its immutable screenshot."""

    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
    )
    return screenshot, OpenCvBoardDetector().analyze(screenshot)


def test_refined_overlay_draws_seed_final_envelope_and_added_band() -> None:
    """Use separate visible colors for all three maximal-envelope geometry layers."""

    screenshot, analysis = _refined_case()
    original = screenshot.image.copy()

    overlay = OpenCvBoardDetectionDebugRenderer().render(screenshot, analysis)

    seed_pixels = np.all(overlay == (255, 180, 40), axis=2)
    final_pixels = np.all(overlay == (40, 220, 40), axis=2)
    band_pixels = np.all(overlay == (40, 180, 255), axis=2)
    assert np.any(seed_pixels)
    assert np.any(final_pixels)
    assert np.any(band_pixels)
    assert np.array_equal(screenshot.image, original)


def test_refinement_metrics_are_written_to_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Render direction, dimensions, continuation, support, and refinement score."""

    screenshot, analysis = _refined_case()
    labels: list[str] = []
    original_put_text = cv2.putText

    def record_text(*args: Any, **kwargs: Any) -> Any:
        labels.append(str(args[1]))
        return original_put_text(*args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record_text)

    OpenCvBoardDetectionDebugRenderer().render(screenshot, analysis)

    assert any("seed=9x8, refined=9x9" in label for label in labels)
    assert any("direction=right" in label and "added=" in label for label in labels)
    assert any("continuation=" in label and "supported=" in label for label in labels)
    assert any("refinement=" in label for label in labels)


def test_non_refined_overlay_preserves_existing_selected_board_layer() -> None:
    """Keep the established green board and grid visualization without refinement."""

    screenshot, _ = truncated_outer_grid_envelope_screenshot(
        rows=9,
        columns=9,
        clipped_side="right",
        low_contrast_outer_band=False,
    )
    analysis = OpenCvBoardDetector().analyze(screenshot)

    overlay = OpenCvBoardDetectionDebugRenderer().render(screenshot, analysis)

    assert analysis.diagnostics.selected_refinement is None
    assert np.any(np.all(overlay == (40, 220, 40), axis=2))
    assert not np.any(np.all(overlay == (40, 180, 255), axis=2))


def test_refined_debug_false_writes_nothing(tmp_path: Path) -> None:
    """Retain explicit opt-in persistence for the new overlay layers."""

    screenshot, analysis = _refined_case()
    destination = tmp_path / "missing" / "board.png"

    saved = OpenCvBoardDetectionDebugRenderer().save_debug_overlay(
        screenshot,
        analysis,
        destination,
        debug=False,
    )

    assert saved is None
    assert not destination.exists()
    assert not destination.parent.exists()
