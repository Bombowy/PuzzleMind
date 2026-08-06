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

### Changed

- Refactored MSS capture to return pixels directly instead of encoding a PNG.
- Made all disk output conditional on an explicit `debug=True` request.
- Documented deterministic candidate filtering, scoring, and tie-breaking behavior.
- Weighted final board confidence as 40% geometry and 60% grid evidence.
- Extended debug overlays with detected separators and grid diagnostics.

### Fixed

- Rejected advertisement-like rectangles through mandatory regular-grid evidence
  instead of relying on geometry confidence or raising the global threshold.

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
