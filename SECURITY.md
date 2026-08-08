# Security Policy

## Supported versions

LogicForge is pre-1.0 software. Security fixes are applied to the latest revision
of `main` until a stable release policy is published.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities in a public issue. Contact the repository
owner privately through their GitHub profile with the affected revision,
reproduction steps, and potential impact. Do not attach credentials or unnecessary
private screenshots.

No response-time guarantee is offered while the project has no dedicated security
team.

## Local automation model

Cats autoplay runs locally and requires no network service. Mouse input is disabled
by default and requires explicit `--execute` opt-in. The current automation adapter
targets a located BlueStacks window on Windows.

Before board actions, the application validates the complete logical solution,
click-plan equality, captured board fingerprint, and current window bounds. Moved
windows and stale captured coordinates are discarded. Ambiguous, unsatisfiable,
search-limited, incomplete, contradictory, or transiently unreadable boards do not
produce board clicks.

These controls reduce accidental input but do not isolate automation from the rest
of the desktop. Users should supervise runs, keep unrelated sensitive windows away
from the target area, use `Ctrl+C` to stop, and remain responsible for where the
automation is used.

## Screenshot and artifact privacy

Screenshots and debug overlays may contain private game, emulator, toolbar,
notification, advertisement, or desktop content. Diagnostic persistence is
explicit, but users should review artifacts before sharing them.

`artifacts/`, repository exports, local reports, environment files, build outputs,
and caches are ignored by Git. Do not force-add them. Remove sensitive images from
local storage when they are no longer needed.

## Dependencies and untrusted input

Treat screenshots, local environment configuration, and dependency updates as
untrusted inputs. Review lockfile changes, use the committed `uv.lock`, and avoid
running modified automation scripts from untrusted branches with `--execute`.
