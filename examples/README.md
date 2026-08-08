# Executable workflows

LogicForge uses scripts as executable examples and composition roots. The primary
workflows are:

- `scripts/autoplay_bluestacks_cats.py` — one-frame dry run by default or explicit
  opt-in continuous autoplay;
- `scripts/solve_bluestacks_cats.py` — one-shot capture, analysis, solve, and
  optional click execution;
- `scripts/detect_bluestacks_cats_tile_grid.py` — tile-lattice diagnostics;
- `scripts/detect_bluestacks_cats_colors.py` — Cats geometry and color diagnostics;
- `scripts/detect_bluestacks_cats_existing_cats.py` — occupied-cell diagnostics;
- `scripts/detect_bluestacks_cats_screen_state.py` — transition-state diagnostics;
- `scripts/detect_bluestacks_board.py` and `detect_bluestacks_grid.py` — generic
  contour-first geometry diagnostics;
- `scripts/capture_bluestacks.py` — immutable window-capture smoke test.

See the [README diagnostics section](../README.md#diagnostics) for commands and the
[architecture document](../ARCHITECTURE.md) for the boundaries demonstrated by
each workflow.
