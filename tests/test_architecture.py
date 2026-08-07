"""Executable dependency-boundary checks for the LogicForge package."""

import ast
from importlib import import_module
from inspect import isabstract
from pathlib import Path
from pkgutil import walk_packages

import logicforge
from logicforge.automation.keyboard import KeyboardController
from logicforge.automation.mouse import MouseController
from logicforge.plugins.base import PuzzlePlugin
from logicforge.rules.rule_engine import RuleEngine
from logicforge.solver.deduction_solver import DeductionSolver
from logicforge.vision.parser import PuzzleParser

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _python_files(relative_directory: str) -> tuple[Path, ...]:
    """Return production Python files in deterministic path order."""

    return tuple(sorted((PROJECT_ROOT / relative_directory).rglob("*.py")))


def _imports(path: Path) -> tuple[str, ...]:
    """Read absolute import module names without executing the source file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    return tuple(imports)


def test_every_package_module_imports_without_optional_infrastructure() -> None:
    """Ensure every packaged module remains importable in the development runtime."""

    discovered_modules = walk_packages(logicforge.__path__, prefix="logicforge.")

    for module in discovered_modules:
        import_module(module.name)


def test_primary_boundary_types_remain_abstract() -> None:
    """Keep ports distinct from the concrete adapters composed by CLI scripts."""

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


def test_production_scripts_do_not_import_other_scripts() -> None:
    """Keep both Cats commands as independent thin composition roots."""

    script_paths = (
        PROJECT_ROOT / "scripts" / "solve_bluestacks_cats.py",
        PROJECT_ROOT / "scripts" / "autoplay_bluestacks_cats.py",
    )
    assert all(
        not module.startswith("scripts")
        for path in script_paths
        for module in _imports(path)
    )


def test_cats_application_policy_has_no_infrastructure_imports() -> None:
    """Require concrete OpenCV and Win32 adapters only at composition roots."""

    assert all(
        not module.startswith("logicforge.infrastructure")
        for path in _python_files("logicforge/application/cats")
        for module in _imports(path)
    )


def test_cats_plugin_does_not_depend_on_application_layer() -> None:
    """Keep reusable Cats rules and exact search below orchestration policy."""

    assert all(
        not module.startswith("logicforge.application")
        for path in _python_files("logicforge/plugins/cats")
        for module in _imports(path)
    )


def test_core_does_not_depend_on_outer_or_plugin_layers() -> None:
    """Protect the innermost logical model from outward dependencies."""

    forbidden = (
        "logicforge.application",
        "logicforge.infrastructure",
        "logicforge.plugins",
    )
    assert all(
        not module.startswith(forbidden)
        for path in _python_files("logicforge/core")
        for module in _imports(path)
    )


def test_cv_detectors_do_not_import_solve_or_autoplay_policy() -> None:
    """Prevent detector adapters from acquiring application orchestration."""

    detector_paths = tuple(
        path
        for path in _python_files("logicforge/infrastructure")
        if path.name.startswith("opencv_") and path.name.endswith("_detector.py")
    )
    assert all(
        not module.startswith("logicforge.application.cats")
        for path in detector_paths
        for module in _imports(path)
    )


def test_exact_search_remains_free_of_cv_win32_and_application_policy() -> None:
    """Keep exact search as deterministic backend-neutral Cats logic."""

    imports = _imports(
        PROJECT_ROOT / "logicforge" / "plugins" / "cats" / "exact_search.py"
    )
    forbidden = ("cv2", "numpy", "win32", "logicforge.application")
    assert all(not module.startswith(forbidden) for module in imports)
