# LogicForge Architecture

## Purpose

LogicForge separates screenshot interpretation, puzzle semantics, deduction,
presentation, and operating-system side effects. The architecture is designed to
support many puzzle families without turning the core solver into a collection of
game-specific conditions.

The codebase currently implements in-memory BlueStacks capture, rectangular board
localization, public grid/cell geometry, and puzzle-neutral LAB color classes.
Symbol interpretation, puzzle parsing, rules, solving, and automation remain
deliberately deferred to later milestones.

## Architectural principles

LogicForge applies Clean Architecture and SOLID principles:

- **Dependency rule:** dependencies point toward stable, puzzle-neutral policies.
- **Single responsibility:** detection, parsing, deduction, propagation,
  explanation, rendering, and automation have separate contracts.
- **Open/closed design:** new plugins and adapters extend behavior without editing
  core entities or orchestration interfaces.
- **Substitutability:** implementations honor small typed ports with deterministic
  inputs and outputs.
- **Interface segregation:** consumers depend on narrow detector, renderer, I/O,
  and automation boundaries rather than a shared service object.
- **Dependency inversion:** use cases consume abstractions; application composition
  will inject OpenCV, Pillow, desktop automation, and storage implementations.

The domain packages do not import infrastructure packages. Plugins may depend on
core and public rule/vision contracts, but not on concrete adapters.

## Package responsibilities

| Package | Responsibility | Must not contain |
| --- | --- | --- |
| `core` | Single mutable solver board and puzzle-neutral values | CV algorithms, I/O, plugin rules |
| `vision` | Screenshot models and detector/parser ports | Deduction logic, mouse actions |
| `rules` | Stateless rule contracts and engine boundary | Puzzle-specific branching, mutation |
| `solver` | Deduction use-case and state-transition boundaries | Image processing, UI code |
| `plugins` | Puzzle-specific parsing and rule composition | Core-framework special cases |
| `explain` | Semantic explanations and formatting ports | Rule execution, board mutation |
| `visualization` | Board and diagnostic rendering ports | Persistence decisions |
| `automation` | Explicit OS input boundaries | Deduction policy |
| `io` | Image loading and artifact export ports | Puzzle interpretation |
| `config` | Immutable application settings | Environment reads in domain code |
| `utils` | Narrow logging and timing ports | Puzzle behavior |

## Overall architecture

```mermaid
flowchart TB
    subgraph Infrastructure["Infrastructure boundary"]
        Files["File adapters"]
        CV["CV adapters"]
        UI["Automation adapters"]
        Renderers["Rendering adapters"]
    end

    subgraph Application["Application policies"]
        Vision["Vision ports"]
        Solver["Solver use case"]
        Engine["Rule Engine port"]
        Explain["Explainability"]
    end

    subgraph Domain["Domain core"]
        Board["Mutable Board cells"]
        Values["Coordinates / Candidate / Enums"]
    end

    subgraph Extensions["Puzzle extensions"]
        Cats["Cats plugin"]
        Future["Future plugins"]
    end

    Files --> Vision
    CV --> Vision
    Vision --> Board
    Cats --> Vision
    Future --> Vision
    Cats --> Engine
    Future --> Engine
    Engine --> Solver
    Board --> Solver
    Solver --> Explain
    Solver --> Renderers
    Solver -. explicit command .-> UI
    Application --> Domain
    Extensions --> Domain
```

## Data flow

The planned end-to-end flow is a sequence of typed transformations:

1. A capture or I/O adapter creates an immutable in-memory BGR `Screenshot`.
2. A plugin parser coordinates board, grid, color, and later symbol detectors.
3. `Board(ColorDetectionResult)` copies `color_matrix` once into the sole mutable
   `cells: list[list[str]]` solver representation.
4. The solver supplies that same board and ordered plugin rules to the Rule Engine.
5. Rules return proposed `RuleOutcome` records; validated outcomes are applied to
   the same matrix through `set_cat` and `set_blocked`.
6. No immutable board, parallel state matrix, region map, or per-change board copy
   is created.
7. Outcomes may later become semantic `Explanation` records with provenance.
8. Renderers present the current board. Automation receives only explicit,
   separately authorized commands derived from a validated state.

No stage may communicate by hidden global state. Failed or ambiguous input will
eventually use typed diagnostics rather than partial mutation or log-only errors.

## Rule Engine

A rule is a stateless policy object with an identifier, description, and one
evaluation operation. Its input is an immutable `RuleContext`; its output is zero
or more proposed outcomes.

The v0.4 engine will be responsible for deterministic ordering, failure isolation,
proposal validation, duplicate handling, conflict detection, and trace capture.
Rules will not apply their own changes. This division keeps each rule independently
testable and gives propagation one authoritative place to enforce invariants.

Puzzle-specific rules live only in plugin packages. Generic engine code must never
branch on a plugin identifier or inspect opaque plugin metadata.

## Vision module

The vision layer uses small ports for independent stages. `BoardDetector` locates
the board, `GridDetector` recovers geometry, and `ColorDetector` produces normalized
observations. `PuzzleParser` is the high-level plugin boundary that interprets
those observations as domain entities.

The public `Screenshot` owns a contiguous, read-only `numpy.ndarray` with an
explicit `uint8` BGR contract. MSS converts native BGRA frames directly into this
model; no component reloads an encoded file.

`BoardDetector` is the inward-facing port. `OpenCvBoardDetector` is an
infrastructure adapter that converts the BGR image to grayscale, applies Gaussian
blur, combines Canny and Otsu-derived masks with morphological closing, extracts
contours, and evaluates rectangular bounding boxes. A scale-relative dilated edge
envelope joins the small visual gaps between adjacent board tiles without parsing
cells. Typed relative thresholds filter title-bar, border, side-toolbar,
advertisement-like, button-sized, and
geometrically implausible candidates without fixing coordinates to one resolution.

Geometry-valid candidates then receive private ROI-level regular-grid validation.
CLAHE normalization, adaptive-threshold edges, Canny edges,
directional morphological opening, and axis projection profiles identify long
horizontal and vertical responses. Nearby responses are clustered by a relative
ROI distance so thick separators count once. Candidate-border responses are
removed, then logical outer boundaries are combined with the ordered internal
positions to estimate rows, columns, spacing variation, and line coverage.

The geometry subscore is:

```text
geometry =
    0.25 * area plausibility
  + 0.25 * rectangularity
  + 0.20 * aspect-ratio plausibility
  + 0.15 * edge-density plausibility
  + 0.15 * expected-location proximity
```

Grid evidence is:

```text
grid =
    0.25 * boundary-count adequacy
  + 0.35 * spacing regularity
  + 0.40 * line coverage

final confidence = 0.40 * geometry + 0.60 * grid
```

Horizontal and vertical values are averaged within each grid component. Spacing
regularity is `clamp(1 - coefficient_of_variation / configured_maximum)`. Boundary
adequacy reaches one at the configured minimum. Coverage is the mean peak
projection response of de-duplicated internal lines. Every component and final
score is clamped to `[0.0, 1.0]`.

Confidence cannot override validation. A candidate is rejected if either axis has
fewer than four boundaries, fewer than three estimated cells, excessive spacing
variation, insufficient coverage, or if aggregate grid evidence is below its
threshold. This fail-closed rule prevents advertisements and UI cards from passing
on geometry alone. Total ordering by confidence, area, position, and size resolves
ties reproducibly. Diagnostics contain only primitive measurements, normalized
line-position tuples, and rejection reasons; no contours or OpenCV types cross the
infrastructure boundary.

The public `OpenCvGridDetector` reuses that exact analyzer and mandatory validation
path; it does not run a second line detector. It validates the supplied
`BoardDetection`, crops only that board ROI, converts normalized boundaries into
full-screenshot integers using deterministic round-half-up, and rejects any
duplicate or reversed pixel result. Outer lines are fixed exactly to the board's
half-open bounds.

`GridDetection` contains every horizontal and vertical boundary, derived row and
column counts, row-major immutable `CellBounds`, and the grid-evidence confidence.
Coordinates are screenshot-relative. A cell includes its top-left pixel and excludes
`x + width` and `y + height`, so consecutive cells tile the complete board without
gaps or overlap. Grid confidence intentionally excludes board confidence. Typed
`GridDetectionError` diagnostics preserve normalized positions, converted lines,
spacing, coverage, score, and rejection reasons without exposing matrices.

OpenCV debug rendering is a separate infrastructure concern. It copies the
immutable source pixels and may draw selected/rejected candidates, de-duplicated
horizontal and vertical lines, estimated dimensions, coverage, and grid score.
Persistence occurs only through an explicit `debug=True` call, so ordinary
detection has no filesystem side effects.

The grid renderer separately draws the public board boundary, all public lines,
cell centers, and optional row/column labels. Its normal and rejected overlays are
also explicit debug-only persistence paths.

`OpenCvColorDetector` consumes only the immutable screenshot and public grid. It
crops four scale-relative inset corner patches from every half-open cell, avoiding
both tile edges and the symbol-prone center. Each patch is converted to OpenCV's
8-bit LAB representation and passed through the existing median/farthest-pixel
trim estimator. A channel-wise corner median identifies exactly one farthest
patch; the final representative is the channel-wise median of the other three.
At least three corner patches must form a complete-link-compatible consensus at
the unchanged LAB threshold. This estimates the background without recognizing
central symbols, cats, X marks, highlights, or animation sprites.

Cell representatives are grouped with deterministic complete-link agglomeration:
two clusters merge only when every cross-cluster LAB distance is within the typed
threshold. This prevents transitive chains from joining endpoints that are not
actually similar. Final centroids are sorted lexicographically by LAB value and
receive contiguous logical identifiers `C0..Cn`. The IDs encode equality only and
do not name colors.

Per-cell confidence is deterministic:

```text
homogeneity = clamp(1 - robust_spread / maximum_within_cell_spread)
cluster_fit = clamp(1 - distance_to_centroid / cluster_distance_threshold)
cell confidence = 0.70 * homogeneity + 0.30 * cluster_fit
global confidence = arithmetic mean of all cell confidences
```

Public immutable results contain row-major `ColorObservation` records, the direct
matrix of logical IDs, class count, global confidence, LAB representatives, and
primitive diagnostics. OpenCV arrays remain inside infrastructure. A separate
renderer labels every cell and may persist an overlay only with explicit debug
behavior. No color stage imports puzzle plugins, solver code, or automation.

The domain `Board` is intentionally different from immutable vision transport.
Its constructor copies `ColorDetectionResult.color_matrix` into nested lists once.
Thereafter `C<n>`, `K`, and `X` coexist in that single mutable matrix, and all
solver-facing queries and assignments operate on it directly. `Board` contains no
`Cell` objects, `Region` objects, snapshot conversion, or second state matrix.
Unresolved `C<n>` entries may transition once to either `K` or `X`. Both final
states accept an identical idempotent assignment but reject the opposite state
with `BoardStateError`; validation occurs before assignment, so a conflict cannot
partially mutate the board.

Cats-specific direct consequences live in `plugins.cats.board_actions`, outside
the puzzle-neutral `Board`. `place_cat()` captures the target color before setting
`K`, computes the deterministic union of same-color, row, column, and eight-neighbor
coordinates, and validates every planned `X` before the first mutation. Only after
that validation does it call `Board.set_cat()` and `Board.set_blocked()`. Existing
`X` values are preserved; any planned `K` or invalid value rejects the whole plan.
This plan-then-apply structure provides atomicity without a rollback copy. The
separate `block_cell()` action delegates one idempotent exclusion to `Board` and
does not propagate. These are operations for future rules, not rule classes or a
solver engine.

## Cats tile-grid-first vision

Cats board geometry does not depend on a single outer board contour. The plugin
contract `CatsTileGridDetector` returns one immutable pair of existing
`BoardDetection` and `GridDetection` results plus primitive diagnostics. Its
OpenCV adapter starts with individual colored components rather than a board-sized
contour: an HSV-saturation-or-LAB-chroma mask is cleaned by a small scale-relative
kernel, then similarly sized near-square components are grouped into families.

For each stable family, component centers are clustered independently on X and Y.
Clusters need repeated support in the orthogonal dimension, which excludes
isolated advertisement and UI components before pitch fitting. The maximal
regular center run is selected independently on each axis before Cartesian slot
assignment. Every fitted row and column then receives a real-component count and
a normalized support ratio against the opposite fitted dimension. A missing
intersection is allowed only inside those supported axes while pitch CV, slot
residual, occupancy, bounds, and score all pass. No empty outer row or column is
extrapolated. Candidate ordering starts with supported lattice area, then minimum
axis support, real component count, occupancy, score, combined pitch CV, residual,
and deterministic position. Confidence combines tile-size consistency,
row-pitch regularity, column-pitch regularity, occupancy, and slot residual
quality. It has no square, expected-dimension, palette, color-count, OCR,
template, or Cats-rule term.

Internal cell boundaries are midpoints of adjacent fitted centers. Outer bounds
are extrapolated by half the median pitch, so public cells include the narrow gaps
around rounded tile interiors. `CellBounds` tile the complete board without gaps
or overlap and retain fitted tile centers. This geometry is passed directly to the
existing puzzle-neutral `OpenCvColorDetector`; it is not re-detected by
`OpenCvGridDetector`.

Cats screen-state BOARD classification and reusable solve analysis use this
combined detector first. A typed failure may fall back to the retained generic
contour-first `OpenCvBoardDetector` and `OpenCvGridDetector`, which remain the
generic path for other puzzles and diagnostics. Autoplay consumes only the shared
analysis function, and its final `rows == columns == color_count` guard remains a
separate fail-closed Cats invariant.

## Cats existing-cat awareness

`CatsExistingCatDetector` is a backend-neutral plugin port over `Screenshot`,
`GridDetection`, and `ColorDetectionResult`. The OpenCV adapter never searches
outside public `CellBounds`, so avatars, counters, power-ups, advertisements, and
BlueStacks chrome are outside its input domain. Each cell supplies a central ROI
inset by 8% horizontally and 6% vertically. Its existing corner-safe
`representative_lab` is the background reference; the global complete-link color
threshold remains 18 and is not reused for occupancy.

Pixels at LAB distance 32 or greater form Cats foreground. A 3.5%-of-shorter-cell
elliptical OPEN/CLOSE kernel removes scale-relative noise. Connected-component
evidence records foreground area, largest coherent area, width and height
coverage, and normalized component-center offset. Hard acceptance requires at
least 0.26 foreground, 0.24 largest component, 0.38 width, 0.38 height, at most
0.18 center offset, and score at least 0.40. The score is 25% foreground + 25%
largest component + 20% width + 20% height + 10% centrality. Thin black or white
X marks can span a wide bounding box but fail the area gates.

Accepted evidence is checked before logical mutation: existing cats must be
unique by row, column, and immutable original color and may not touch in eight
directions. Contradictory image evidence raises a typed error without choosing a
winner. Solve composition creates one `Board`, applies existing cats in row-major
order through `place_cat()`, and only then runs unchanged Cats rules. Final
validation still covers every K and requires the full row/column/color/non-touch
solution. The click plan is exactly final K minus validated existing coordinates,
so counters and double-click execution cover only newly placed cats.

## Cats deterministic exact-search fallback

The seven ordered Cats rules remain the preferred deduction mechanism. Exact
search is called only when their fixed-point loop leaves `C<n>` cells. It consumes
the current `Board` plus the immutable `ColorDetectionResult.color_matrix`, which
is authoritative for the original color under both newly deduced and pre-existing
`K` cells.

Branching never mutates or clones `Board`. A lightweight immutable state retains
candidate coordinates, selected cats, and used rows, columns, and original
colors. Every assignment removes candidates in its row, column, color, and eight
neighbor area. Color, row, and column singleton constraints propagate to a fixed
point before branching and after every hypothetical assignment. Contradictory
zero-candidate groups close the branch.

MRV considers unresolved color groups, then rows, then columns. Ordering is
candidate count, group type, numeric color suffix or line index, and row-major
coordinates; branch coordinates are also row-major. The traversal is therefore
independent of hash/set iteration and random state. Search retains the first
solution and continues until the tree is exhausted or a second distinct solution
is found. Its public statuses are `UNIQUE`, `UNSAT`, `AMBIGUOUS`, and
`LIMIT_REACHED`; a deterministic 250,000-node default replaces a wall-clock
timeout.

Only `UNIQUE` is applied to the same logical Board, after validating the complete
row/column/color/non-touch solution. Existing `K` assignments stay fixed;
remaining solution cells use `place_cat()` and any safe residual non-solution
unknowns use `block_cell()`. All other statuses produce no click plan. Autoplay
inherits this behavior through `solve_analyzed_cats_board()` and still validates
the complete final Board while excluding detected existing cats from new click
targets.

## Solver module

The solver is an application use case, not a collection of puzzle rules. It will
coordinate repeated engine evaluations and propagation until the board is solved,
stalled, contradictory, cancelled, or limited by policy.

Solver iterations will mutate the one supplied `Board` through its narrow methods.
Lifecycle and explanation metadata may record what changed, but must not duplicate
the board matrix or create a board copy after every deduction. The Cats plugin's
bounded exact fallback is an explicit deterministic strategy; probabilistic or
first-solution guessing remains outside the v1.0 scope.

## Plugin system

`PuzzlePlugin` is the extension boundary for metadata, parser composition, and an
ordered rule catalog. The planned registry will discover plugins through Python
entry points and enforce identifier and framework-version compatibility.

Plugins may:

- map visual evidence into the generic board model;
- define plugin-owned metadata and region semantics;
- contribute small stateless rules;
- provide plugin-specific rendering assets through future contracts.

Plugins may not invoke automation, mutate solver state, configure global logging,
or require the core package to recognize their puzzle type.

## Explainability system

Explainability begins at the rule outcome, not after solving. Each proposed and
applied transition retains a rule identifier, deduction kind, affected coordinates,
summary, and future structured evidence. `Explanation` is presentation-neutral;
formatters convert it to text, Markdown, JSON, or UI content.

The v0.6 design will add structured premises, conclusions, localization keys, and
links between state transitions. A formatter must never invent evidence or rerun a
rule to explain a historical result.

## Composition and side effects

A future composition root will instantiate settings, adapters, plugin registry,
engine, propagation strategy, and solver. Constructors will receive dependencies
explicitly. Environment variables, file reads, image decoding, logs, clocks,
rendering, and desktop input remain at the outer boundary.

Automation is intentionally separated from solving. It will require dry-run mode,
screen and focus validation, rate limits, explicit user authorization, and an
emergency stop before v0.7 can emit real input events.

## Future roadmap

- **v0.2:** extend screenshot interpretation beyond the implemented board locator.
- **v0.3:** finalize the single mutable string-matrix board contract.
- **v0.4:** implement deterministic engine and atomic propagation.
- **v0.5:** implement Cats parsing, constraints, rules, and fixture corpus.
- **v0.6:** add structured, localized, replayable explanations.
- **v0.7:** add guarded automatic gameplay adapters.
- **v1.0:** stabilize public contracts and Cats compatibility.
- **Future:** add Sudoku, Nonogram, Kakuro, Hashi, and Nurikabe plugins without
  changing puzzle-neutral core policies.

Architectural decisions that change public contracts should be documented as an
ADR under `docs/decisions/` before implementation.
