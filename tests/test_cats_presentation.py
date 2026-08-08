"""Cats solve presentation tests."""

import pytest

from logicforge.application import cats as cats_app


def test_format_matrix_supports_immutable_tuple_matrix() -> None:
    """Format the immutable color-result representation without conversion."""

    assert cats_app.format_matrix((("C0", "C1"), ("C2", "C3"))) == ("C0 C1\nC2 C3")


def test_format_matrix_supports_mutable_list_matrix() -> None:
    """Format the Board's mutable nested-list representation directly."""

    assert cats_app.format_matrix([["K", "X"], ["C0", "C1"]]) == (" K  X\nC0 C1")


def test_format_matrix_aligns_c2_with_c10() -> None:
    """Right-align short values to the widest logical identifier."""

    assert cats_app.format_matrix((("C2", "C10"), ("X", "K"))) == (" C2 C10\n  X   K")


def test_print_click_plan_outputs_target_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the number of planned new-cat targets."""

    target = cats_app.CatClickTarget(1, 2, 300, 250, 700, 550)

    cats_app.print_cat_click_plan((target,))

    assert "Planned cat click targets: 1" in capsys.readouterr().out


def test_print_click_plan_outputs_logical_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print zero-based row and column for inspection."""

    target = cats_app.CatClickTarget(1, 2, 300, 250, 700, 550)

    cats_app.print_cat_click_plan((target,))

    assert "row=1, column=2" in capsys.readouterr().out


def test_print_click_plan_outputs_screenshot_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the exact CellBounds center in screenshot space."""

    target = cats_app.CatClickTarget(1, 2, 300, 250, 700, 550)

    cats_app.print_cat_click_plan((target,))

    assert "screenshot=(300, 250)" in capsys.readouterr().out


def test_print_click_plan_outputs_desktop_coordinates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print the offset absolute virtual-desktop position."""

    target = cats_app.CatClickTarget(1, 2, 300, 250, 700, 550)

    cats_app.print_cat_click_plan((target,))

    assert "desktop=(700, 550)" in capsys.readouterr().out


def test_print_empty_click_plan_outputs_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print only a zero-count summary when no K exists."""

    cats_app.print_cat_click_plan(())

    assert capsys.readouterr().out.strip() == "Planned cat click targets: 0"
