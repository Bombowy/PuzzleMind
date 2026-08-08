"""Shared normalized scoring primitives for Cats screen-state features."""


def clamp_unit(value: float) -> float:
    """Clamp a normalized component into the inclusive unit interval."""

    return max(0.0, min(1.0, value))


def triangular_score(
    value: float,
    minimum: float,
    preferred: float,
    maximum: float,
) -> float:
    """Score a value linearly around one preferred point inside hard bounds."""

    if value < minimum or value > maximum:
        return 0.0
    if value <= preferred:
        return clamp_unit((value - minimum) / max(preferred - minimum, 1e-9))
    return clamp_unit((maximum - value) / max(maximum - preferred, 1e-9))


def threshold_score(value: float, minimum: float) -> float:
    """Normalize evidence from its hard minimum toward the ideal value one."""

    return clamp_unit((value - minimum) / max(1.0 - minimum, 1e-9))
