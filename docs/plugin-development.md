# Puzzle plugin development

Cats is the current reference plugin. It owns puzzle-specific logical mutations,
deduction rules, exact search, and backend-neutral Cats vision contracts. Concrete
OpenCV and operating-system implementations remain outside the plugin.

LogicForge intentionally has no generic plugin registry today. Add a shared plugin
abstraction only after another real puzzle demonstrates a common runtime contract.

## Responsibility boundaries

Puzzle-specific code belongs under `logicforge/plugins/<puzzle>/` when it owns:

- logical cell states or constraints;
- domain actions over the puzzle-neutral `Board`;
- deterministic deduction rules;
- backend-neutral detector results, diagnostics, errors, or ports;
- an exact solver whose constraints are specific to that puzzle.

A plugin may import `logicforge.core` and backend-neutral vision contracts. It must
not import OpenCV, NumPy image-processing APIs, Win32, concrete infrastructure, CLI
scripts, or application orchestration.

Application flow belongs under `logicforge/application/<puzzle>/`. It coordinates
detectors, initializes logical state, runs rules/search, validates final results,
and constructs safe action plans. Concrete adapter selection belongs in scripts.

## Cats rule contract

Current Cats rules are small stateless objects with:

```python
def apply(self, board: Board) -> bool:
    ...
```

`True` means one validated logical deduction changed the board. `False` means the
rule had no applicable move. The fixed-point loop restarts from the first rule
after every `True`, so rule order is part of behavior and requires regression tests.

Rules are deterministic and have no file, network, logging, clock, CV, or desktop
side effects. They may mutate the supplied board only through plugin-owned domain
actions such as `place_cat()` and `block_cell()`. Multi-cell operations must build
and validate their complete mutation plan before the first write.

Every new or changed rule needs positive, negative, ordering, contradiction, and
idempotence coverage appropriate to its inference.

## Exact search

Puzzle-specific exact search remains backend-neutral. Cats search consumes the
current logical board plus the immutable original color matrix, explores its own
lightweight constraint state, and never mutates the real board while branching.
Traversal, propagation, tie-breaks, node limits, uniqueness proof, and outcome
statuses are observable behavior and require deterministic tests.

## Vision contracts

Define a plugin-level detector port only when a real adapter and application
consumer exist. Public results and diagnostics should contain immutable primitive
data, not OpenCV matrices or contours. Infrastructure implements the port; the
application receives it through dependency injection.

For Cats, the active plugin ports are:

- `CatsTileGridDetector`
- `CatsExistingCatDetector`
- `CatsScreenStateDetector`

## Checklist

- Keep application and infrastructure imports out of the plugin.
- Keep rules deterministic, stateless, and I/O-free.
- Preserve one authoritative mutable `Board`.
- Preserve immutable original evidence needed after board mutation.
- Fail closed on contradictory or ambiguous evidence.
- Add focused tests and architecture-boundary coverage.
- Update README, architecture, and changelog when behavior or contracts change.
