# Development guide

## Toolchain

LogicForge targets Python 3.13 and uses uv for Python acquisition, dependency
resolution, locking, environments, command execution, and package builds.

```bash
uv sync --locked --all-groups
uv run black --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Run `uv lock` only when intentionally changing project or development dependencies.
Review lockfile changes together with `pyproject.toml`.

## Definition of done

A change is complete when it respects package dependency rules, includes strict
types and responsibility-focused documentation, adds appropriate deterministic
tests, updates user and architecture documentation, records notable changes, and
passes the same commands as CI.

## Adding a module

Place a module in the package that owns its policy. Define the smallest useful
contract before choosing a concrete library. Keep image backend types and external
exceptions behind adapters. Avoid utility modules that combine unrelated behavior.

## Logging and diagnostics

Domain objects do not log. Future application services may receive a logger through
an outer-layer factory. Expected failures will use typed results or exceptions;
logs are operational evidence, not an API or error-handling mechanism.
