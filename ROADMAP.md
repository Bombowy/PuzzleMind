# LogicForge Roadmap

The roadmap is capability-based. It avoids version promises until a concrete use
case and acceptance tests justify a public contract.

## Current

- End-to-end Cats analysis, deterministic solving, validation, and autoplay from
  a live BlueStacks window on Windows.
- Classical OpenCV tile-lattice, LAB color, existing-cat, and transition-state
  detection without OCR, templates, or ML.
- Seven deterministic deduction rules plus uniqueness-proving exact constraint
  search.
- Explicit opt-in Win32 automation with stale-state, moved-window, retry, timeout,
  and complete-solution guards.
- Synthetic regression suite, strict typing, linting, formatting, coverage, locked
  dependencies, CI, and reproducible package builds.

## Next

- Publish an honest screen recording showing dry-run analysis and supervised live
  autoplay, without adding private screenshots to the repository.
- Grow a legally shareable, versioned fixture corpus for animation, theme, and
  unusual board-layout regressions.
- Improve operational observability with concise structured run summaries and
  retained failure diagnostics that do not leak screenshot pixels by default.
- Evaluate an additional capture or automation adapter only when a real platform
  requirement is available for testing.
- Stabilize the smallest useful application and plugin APIs before any 1.0 claim.

## Later

- Add another puzzle family and use it to identify genuinely shared plugin
  contracts.
- Add richer deduction explanations based on real rule transitions and exact-search
  evidence.
- Generalize plugin discovery only if a second independently shipped puzzle proves
  that static composition roots are insufficient.
- Consider broader platform support after window ownership, coordinate mapping,
  and input-safety behavior can be tested on those platforms.
