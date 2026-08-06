# Contributing to LogicForge

Thank you for helping build LogicForge. The project values correctness,
explainability, small interfaces, and maintainable puzzle extensions.

## Before starting

For substantial work, open an issue describing the problem, proposed boundary, and
test strategy. Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing a public
contract. Security reports must follow [SECURITY.md](SECURITY.md), not public issues.

## Development setup

Install Python 3.13 and uv, then run:

```bash
git clone https://github.com/Bombowy/PuzzleMind.git
cd PuzzleMind
uv sync --locked --all-groups
```

Create a focused branch from `main`. Do not commit `.venv`, generated artifacts,
screenshots containing private data, or tool caches.

## Architecture rules

- Keep the `core` package free from infrastructure and plugin dependencies.
- Put puzzle-specific parsing and rules inside the owning plugin.
- Prefer immutable input and result records over shared mutable state.
- Depend on narrow abstract ports, never concrete CV or automation libraries.
- Keep rules deterministic, stateless, side-effect free, and independently tested.
- Preserve provenance for every future board transition.
- Add new abstractions only when a real use case needs them.

Public API changes require updated architecture documentation and a changelog entry.
Cross-cutting decisions should include an ADR in `docs/decisions/`.

## Code standards

All production code must use Python 3.13 type hints. Public classes and functions
need responsibility-focused docstrings. TODO comments must state the deferred work,
the milestone or prerequisite, and relevant correctness or safety constraints.

Run the complete local gate before opening a pull request:

```bash
uv run black .
uv run ruff check .
uv run mypy
uv run pytest
uv build
```

Do not weaken a quality rule merely to make a change pass. Document and justify a
narrow suppression beside the affected code when a tool limitation requires one.

## Tests

Place unit tests under `tests/` using the same conceptual structure as the package.
Future vision and plugin behavior must use versioned, legally shareable fixtures.
Tests must not depend on network access, a live game, desktop focus, or execution
order. Automation adapters require safe fakes and dry-run tests before real events.

## Commits and pull requests

Use short imperative commit subjects. A pull request should explain:

- the problem and chosen design;
- architectural boundaries affected;
- tests and manual validation performed;
- compatibility or migration impact;
- follow-up work intentionally left out.

Keep pull requests focused. CI must pass and review conversations must be resolved
before merge.
