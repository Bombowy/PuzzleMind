# Roadmap

The roadmap is capability-based. Version dates will be assigned only when the
preceding milestone meets its acceptance criteria.

## v0.1 — Project architecture

- Establish package boundaries and dependency rules.
- Define typed entities, value objects, ports, and plugin contracts.
- Configure uv, Ruff, Black, MyPy, Pytest, packaging, and CI.
- Publish contributor, architecture, governance, and roadmap documentation.

## v0.2 — Screenshot parser

- Define typed parse results, diagnostics, and confidence semantics.
- Implement screenshot loading plus public board and grid detector adapters.
- Add puzzle-neutral deterministic cell-color classification.
- Add anonymized screenshot fixtures and debug overlays.
- Document supported formats, coordinate systems, and failure modes.

## v0.3 — Board representation

- Finalize board invariants and validated construction.
- Add efficient immutable coordinate and region indexes.
- Define uncertain and incomplete observation semantics.
- Add serialization and property-based domain tests.

## v0.4 — Rule Engine

- Implement deterministic rule scheduling and evaluation.
- Define typed state-transition commands.
- Implement atomic propagation and conflict handling.
- Add traceability, limits, cancellation, and replay tests.

## v0.5 — Cats Puzzle Solver

- Implement the Cats screenshot parser.
- Model Cats regions, symbols, and constraints.
- Implement small, independently tested Cats rules.
- Validate end-to-end solving against a versioned fixture corpus.

## v0.6 — Explainable deductions

- Add structured premises and conclusions to every transition.
- Implement plain-text, Markdown, and JSON formatters.
- Add replayable explanation chains and localization foundations.

## v0.7 — Automatic gameplay

- Implement dry-run mouse and keyboard adapters.
- Add display scaling, focus, bounds, and rate-limit guards.
- Require explicit authorization and an emergency stop.
- Validate actions against screenshots after each interaction.

## v1.0 — Stable Cats plugin

- Stabilize the public plugin and solver APIs.
- Publish compatibility and deprecation policies.
- Meet documented performance, correctness, and explanation targets.
- Ship a production-ready Cats plugin and user guide.

## Future

- Sudoku
- Nonogram
- Kakuro
- Hashi
- Nurikabe
