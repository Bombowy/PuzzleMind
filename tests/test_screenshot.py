"""Tests for the immutable in-memory BGR screenshot model."""

from datetime import UTC, datetime

import numpy as np
import pytest

from logicforge.vision.screenshot import Screenshot


def test_screenshot_owns_a_contiguous_read_only_bgr_copy() -> None:
    """Protect detector input from caller mutation and non-contiguous array views."""

    source = np.arange(60, dtype=np.uint8).reshape((4, 5, 3))[:, ::-1, :]
    screenshot = Screenshot(
        image=source,
        width=5,
        height=4,
        timestamp=datetime.now(UTC),
    )
    original_first_pixel = screenshot.image[0, 0].copy()

    source[0, 0] = 0

    assert screenshot.image.shape == (4, 5, 3)
    assert screenshot.image.dtype == np.uint8
    assert screenshot.image.flags.c_contiguous
    assert not screenshot.image.flags.writeable
    assert np.array_equal(screenshot.image[0, 0], original_first_pixel)
    with pytest.raises(ValueError, match="read-only"):
        screenshot.image[0, 0, 0] = 0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        screenshot.image.setflags(write=True)


def test_screenshot_rejects_non_bgr_shape() -> None:
    """Reject grayscale or mismatched data before detector consumption."""

    with pytest.raises(ValueError, match="exactly"):
        Screenshot(
            image=np.zeros((4, 5), dtype=np.uint8),
            width=5,
            height=4,
            timestamp=datetime.now(UTC),
        )


def test_screenshot_rejects_non_uint8_pixels() -> None:
    """Keep the public pixel contract precise and compatible with OpenCV BGR."""

    with pytest.raises(TypeError, match="uint8"):
        Screenshot(
            image=np.zeros((4, 5, 3), dtype=np.float32),
            width=5,
            height=4,
            timestamp=datetime.now(UTC),
        )


def test_screenshot_rejects_a_naive_timestamp() -> None:
    """Require timezone-aware capture time for portable ordering and diagnostics."""

    with pytest.raises(ValueError, match="timezone"):
        Screenshot(
            image=np.zeros((4, 5, 3), dtype=np.uint8),
            width=5,
            height=4,
            timestamp=datetime.now(),
        )
