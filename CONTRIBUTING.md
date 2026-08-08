# Contributing to LogicForge

LogicForge values deterministic behavior, explicit safety checks, small
abstractions derived from real requirements, and regression-focused tests.

## Before starting

For substantial work, open an issue describing the concrete problem, affected
boundary, and test strategy. Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing
public contracts. Report security concerns through [SECURITY.md](SECURITY.md), not
through a public issue.

## Development setup

Install Python 3.13 and `uv`, then run:

```bash
git clone https://github.com/Bombowy/PuzzleMind.git
cd PuzzleMind
uv sync --locked --all-groups
```

Create a focused branch from `main`. Do not commit `.venv`, generated artifacts,
private screenshots, repository exports, local reports, or tool caches.

## Architecture rules

- Keep `logicforge.core` free from Cats, application, and infrastructure concerns.
- Keep puzzle-specific mutations, rules, constraints, and exact search in the
  owning plugin.
- Keep application orchestration outside plugin and infrastructure modules.
- Depend on active backend-neutral ports, not concrete OpenCV or Win32 adapters.
- Add an abstraction only when a real adapter or consumer requires it.
- Preserve immutable captured evidence and one authoritative mutable logical board.
- Keep rules deterministic and stateless, with no I/O or external side effects.
- Mutate the board only through owning domain/plugin actions.
- Validate every multi-cell mutation plan before its first write.
- Treat rule order, exact-search traversal, thresholds, and automation defaults as
  observable behavior requiring explicit regression coverage.

Public API or dependency-boundary changes require updated architecture documentation
and a changelog entry. Cross-cutting decisions may be recorded under
`docs/decisions/` when the rationale cannot be expressed clearly near the code.

## Code standards

Production code targets Python 3.13 and uses complete type annotations. Public
classes and functions need responsibility-focused docstrings. Avoid milestone
fiction, speculative TODOs, and placeholder interfaces without real consumers.

Run the complete local gate before opening a pull request:

```bash
uv run black .
uv run black --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Do not weaken a quality rule or delete behavioral coverage to make a change pass.
Document a narrow suppression beside the code only when a tool limitation requires
one.

## Tests

Tests must be deterministic and should use synthetic images or injected fakes.
They must not depend on network access, a live emulator, desktop focus, real mouse
events, wall-clock timing, or execution order.

Behavior changes require focused positive and negative cases. Safety-sensitive
paths need explicit zero-click tests. Vision changes need scale/geometry variants;
solver changes need contradiction, ambiguity, determinism, and non-mutation cases.

## Commits and pull requests

Use short imperative commit subjects. A pull request should explain:

- the concrete problem and chosen design;
- boundaries and observable behavior affected;
- tests and manual validation performed;
- compatibility or migration impact;
- follow-up work intentionally left out.

Keep pull requests focused, let CI finish, and resolve review conversations before
merge.
