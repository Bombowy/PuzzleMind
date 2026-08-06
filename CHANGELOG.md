# Changelog

All notable changes to LogicForge will be documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases will
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) after the public
API reaches stability.

## [Unreleased]

### Planned

- Screenshot parsing contracts and diagnostics for v0.2.

### Added

- Windows-only BlueStacks App Player lookup through pywin32.
- Typed capture ports, application service, debugging script, and milestone report.
- Immutable in-memory BGR `Screenshot` backed by a read-only NumPy array.
- Optional OpenCV debug export to `artifacts/vision/bluestacks_capture.png`.
- Classical OpenCV puzzle-board detection behind the existing `BoardDetector` port.
- Typed scale-relative board-detection settings, candidate diagnostics, and
  `BoardDetectionError` failure reporting.
- Explicit board overlay rendering to `artifacts/vision/board_detection.png`.
- Synthetic detector and renderer tests that require no live desktop application.
- Primitive internal-grid diagnostics with normalized separator positions,
  estimated dimensions, spacing regularity, coverage, and evidence score.
- Public `OpenCvGridDetector`, immutable `GridDetection`/`CellBounds` models, typed
  diagnostics/errors, and full-screenshot half-open coordinate semantics.
- Deterministic row-major cell generation and explicit grid debug rendering/script.
- Post-failure BlueStacks resizing guidance when a captured screenshot is below
  the operational 440x470 recommendation.
- Immutable color observations/results, typed diagnostics/errors, and validated
  puzzle-neutral color-detection settings.
- Classical OpenCV LAB cell sampling, deterministic complete-link color grouping,
  full capture-to-color script, and explicit labeled debug overlay.
- Cats-specific atomic `place_cat` propagation across color, row, column, and
  neighbors, plus a non-propagating idempotent `block_cell` action.
- Stateless `SingleRemainingColorCellRule` using numeric color ordering and at
  most one atomic `place_cat` action per `apply()` call.
- Stateless `SingleRemainingLineCellRule` placing a cat when a row or column has
  exactly one unresolved cell and every other cell in that line is blocked.
- Stateless `ColorConfinedToLineRule` with atomic row/column exclusion planning,
  numeric color ordering, and `block_cell`-only mutations.
- Stateless `ColorSubsetConfinedToLinesRule` reserving N rows or columns for N
  unresolved colors and atomically blocking other colors in those lines.
- Stateless `ImpossibleCatCandidateRule` providing mutation-free one-step
  failed-candidate checks as the final Cats deduction priority.
- Pure `collect_cat_exclusion_coordinates` planning shared by `place_cat` and
  hypothetical Cats candidate analysis.
- Stateless `MonochromaticLineColorExclusionRule` for excluding a line's sole
  remaining color everywhere outside that row or column.
- Stateless `AdjacentColorPairExclusionRule` for atomic perpendicular exclusions
  around exactly two orthogonally adjacent candidates of one color.
- Minimal `apply_cats_rules_until_stalled` fixed-point loop that restarts the
  ordered Cats rules after every successful application.
- Manual `solve_bluestacks_cats.py` capture-to-deduction script with aligned board
  output, COMPLETE/STALLED diagnostics, and no gameplay automation.
- Dry-run Cats click-plan mapping from logical K coordinates through exact
  `CellBounds` centers to absolute desktop positions, without emitting clicks.
- Opt-in `--execute-clicks` Cats automation through a lazy `Win32MouseController`:
  every row-major K target receives two left clicks with a default 10 ms delay
  between consecutive clicks, while invocation without the flag remains dry-run.
- Immutable Cats screen-state contracts and a one-shot OpenCV classifier for
  `BOARD`, `RANKING`, `LEVEL_COMPLETE`, and `UNKNOWN` in overlay-first priority.
- OCR-free transition recognition using relative orange-button geometry for level
  completion, aligned bright-card stacks for ranking, and the existing board/grid
  detectors for normal gameplay screens.
- Diagnostic screenshot-space action points and a no-click
  `detect_bluestacks_cats_screen_state.py` script with an explicit debug overlay.
- Optional `autoplay_bluestacks_cats.py` state machine for continuous `BOARD`,
  optional `RANKING`, mandatory `LEVEL_COMPLETE`, and passive `UNKNOWN` handling.
- Complete pre-click Cats solution validation across rows, columns, original
  colors, non-touching cats, terminal cells, and exact duplicate-free click plans.
- Bounded transition polling, stationary-overlay retries, stale-board fingerprint
  protection, pre-action window-bounds checks, `--max-levels`, Ctrl+C summaries,
  and a single-capture no-click autoplay dry run.

### Changed

- Replaced the immutable placeholder board aggregate with one mutable nested-list
  matrix copied from `ColorDetectionResult`, plus direct query and `K`/`X` mutation
  methods.
- Made `K` and `X` terminal, idempotent board states and added `BoardStateError`
  for contradiction-safe `K -> X`, `X -> K`, and invalid-value mutations.
- Refactored MSS capture to return pixels directly instead of encoding a PNG.
- Made all disk output conditional on an explicit `debug=True` request.
- Documented deterministic candidate filtering, scoring, and tie-breaking behavior.
- Weighted final board confidence as 40% geometry and 60% grid evidence.
- Extended debug overlays with detected separators and grid diagnostics.
- Shared one internal grid-analysis and mandatory-validation path between board
  acceptance and public grid geometry extraction.
- Refactored the one-shot Cats solver into reusable capture analysis and logical
  solve functions without changing its CLI, exit codes, output, or click behavior.
- Autoplay retains the existing row-major double left click per `K` with a 10 ms
  default delay, while ranking and level-complete overlays receive one left click;
  `STALLED` and validation failures stop before any board click.
- Extended the one-shot grid diagnostic output with final horizontal/vertical
  boundaries and derived row heights/column widths.
- Treat contour-derived board rectangles as regular-grid seeds and refine a seed
  by at most one image-supported outer cell band on each axis. The same shared
  internal analyzer and mandatory validation re-evaluate every proposed envelope.

### Fixed

- Rejected advertisement-like rectangles through mandatory regular-grid evidence
  instead of relying on geometry confidence or raising the global threshold.
- Conservatively recover at most one weak separator per grid axis when one
  regular-spacing gap is approximately double width, a real normalized weak
  directional/Sobel response exists near its midpoint, and spacing CV improves;
  the original global strong-line threshold remains unchanged.
- Reject transient Cats input such as `9x8` with nine colors before constructing
  `Board`, running rules, or clicking, then retry through the existing polling and
  timeout path until square `rows == columns == color_count` geometry appears.
- Made Cats transition detection viewport-aware: advertisement panels, black
  margins, and the BlueStacks toolbar are excluded from overlay geometry;
  `LEVEL_COMPLETE` and `RANKING` are measured inside the detected vertical game
  viewport, then their rectangles and action points are translated back to full
  screenshot coordinates while `BOARD` remains a full-screenshot analysis.
- Recover a low-contrast outer row or column that contour extraction omitted by
  requiring the old seed border to become a real internal separator, continued
  orthogonal separators across the added band, one-cell sizing, regular spacing,
  and complete grid evidence. Ambiguous directions fail closed; rectangular grids,
  weak internal-line recovery, and the Cats geometry guard remain unchanged, with
  no square, Cats, or color-count assumption in generic vision.

## [0.1.0] - 2026-08-06

### Added

- Clean Architecture package structure and dependency boundaries.
- Typed domain records and empty interfaces for every planned subsystem.
- Initial Cats plugin contracts without puzzle-solving logic.
- uv project metadata and lockfile workflow.
- Ruff, Black, strict MyPy, Pytest, coverage, and GitHub Actions configuration.
- Architecture, roadmap, contribution, governance, and development documentation.

[Unreleased]: https://github.com/Bombowy/PuzzleMind/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Bombowy/PuzzleMind/releases/tag/v0.1.0
