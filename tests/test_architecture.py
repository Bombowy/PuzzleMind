"""Smoke tests for the v0.1 architectural contracts."""

from importlib import import_module
from inspect import isabstract
from pkgutil import walk_packages

import logicforge
from logicforge.automation.keyboard import KeyboardController
from logicforge.automation.mouse import MouseController
from logicforge.plugins.base import PuzzlePlugin
from logicforge.rules.rule_engine import RuleEngine
from logicforge.solver.deduction_solver import DeductionSolver
from logicforge.vision.parser import PuzzleParser


def test_every_package_module_imports_without_optional_infrastructure() -> None:
    """Ensure public contracts do not accidentally require future heavy adapters.

    This protects the dependency-inversion boundary while Computer Vision and
    desktop-automation libraries remain intentionally absent from dependencies.
    """

    discovered_modules = walk_packages(logicforge.__path__, prefix="logicforge.")

    for module in discovered_modules:
        import_module(module.name)


def test_primary_boundary_types_remain_abstract() -> None:
    """Ensure v0.1 ports cannot be mistaken for working concrete implementations.

    Concrete behavior belongs to future milestone adapters and services; keeping
    these types abstract makes accidental production use fail during composition.
    """

    boundary_types = (
        KeyboardController,
        MouseController,
        PuzzlePlugin,
        RuleEngine,
        DeductionSolver,
        PuzzleParser,
    )

    assert all(isabstract(boundary_type) for boundary_type in boundary_types)


def test_package_version_matches_architecture_milestone() -> None:
    """Keep runtime package metadata aligned with the documented v0.1 milestone."""

    assert logicforge.__version__ == "0.1.0"
