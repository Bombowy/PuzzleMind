# Changelog

All notable changes to LogicForge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Semantic Versioning will
govern compatibility after the public API stabilizes.

## [Unreleased]

### Added

- End-to-end Cats workflow for a live BlueStacks window: immutable capture,
  screen-state classification, tile-lattice geometry, LAB colors, existing-cat
  evidence, logical solving, complete validation, click planning, and explicit
  opt-in Win32 automation.
- Classical OpenCV Cats tile-lattice detection with independently supported row
  and column runs, scale-relative fitting, occupied missing-slot support, and
  deterministic diagnostics/rendering.
- Four-corner LAB background classification with robust sampling, corner consensus,
  deterministic complete-link clustering, and central-sprite resistance.
- CellBounds-local existing-cat detection using scale-relative foreground and
  connected-component evidence plus fail-closed Cats invariant validation.
- Seven ordered deterministic Cats deduction rules over one mutable `Board`, with
  atomic plugin-owned `place_cat()` and `block_cell()` actions.
- Deterministic exact constraint-search fallback with row/column/color singleton
  propagation, MRV branching, fixed existing cats, uniqueness proof, and bounded
  `UNIQUE`/`UNSAT`/`AMBIGUOUS`/`LIMIT_REACHED` outcomes.
- Full solution and click-plan validation across all cats, while excluding already
  present cats from newly executed targets.
- Dependency-injected autoplay state machine for `BOARD`, `RANKING`,
  `LEVEL_COMPLETE`, and `UNKNOWN`, including bounded transient BOARD retries,
  overlay retries, stale-board protection, moved-window guards, and summaries.
- Diagnostic commands and explicit debug overlays for capture, generic board/grid,
  Cats lattice, colors, existing cats, and screen state.
- Synthetic regression coverage for multi-size lattices, occupied slots, color
  sampling, cat/X separation, exact-search outcomes, transition states, retries,
  coordinate mapping, and zero-click safety paths.

### Changed

- Made Cats tile-grid-first analysis the primary geometry path while retaining the
  generic contour-first board/grid adapters as diagnostics and a typed fallback.
- Reduced solve and autoplay scripts to CLI parsing, concrete dependency composition,
  presentation, exit-code mapping, and `main()`; reusable policy now lives under
  `logicforge.application.cats`.
- Replaced arbitrary solve-status strings with the typed `CatsSolveStatus` enum
  without changing existing CLI values or meanings.
- Shared exact four-corner sampling geometry between color detection and rendering;
  removed the unused historical central-sample setting.
- Renamed transition evidence from orange-only terminology to warm red/orange CTA
  terminology without changing calibrated values or scoring.
- Removed speculative generic solver, rule-engine, plugin-registry, parser,
  explainability, I/O, visualization, utility, keyboard, and unused core-model
  scaffolding that had no runtime consumer.
- Rewrote public documentation around the current working product, real dependency
  boundaries, safety model, executable workflows, and capability roadmap.

### Fixed

- Preserved a full supported Cats lattice when an existing cat hides one normal
  tile component instead of selecting a smaller perfect inner grid.
- Prevented central sprites, X marks, highlights, and short transition animations
  from corrupting color or existing-cat interpretation.
- Prevented the solver from clicking the first valid completion without proving
  uniqueness, and retained zero-click behavior for contradictions and unresolved
  outcomes.
- Retried transient board/grid/color/existing-cat/geometry failures with newly
  captured frames for a bounded window without treating failures as progress.
- Rejected stale board analysis, moved-window coordinates, invalid Cats geometry,
  duplicate/extraneous click plans, and incomplete final logical states before
  mouse input.

## [0.1.0] - 2026-08-06

### Added

- Initial Clean Architecture package structure and dependency boundaries.
- Typed domain records and placeholder interfaces for the originally planned
  subsystems.
- Initial Cats namespace without puzzle-solving behavior.
- `uv` project metadata and lockfile workflow.
- Ruff, Black, strict MyPy, Pytest, coverage, and GitHub Actions configuration.
- Initial architecture, roadmap, contribution, governance, and development docs.

[Unreleased]: https://github.com/Bombowy/PuzzleMind/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bombowy/PuzzleMind/releases/tag/v0.1.0
