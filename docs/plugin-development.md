# Plugin development guide

Puzzle plugins are isolated extensions that translate screenshots into generic
boards and provide deterministic rule catalogs. The API is provisional until v1.0.

## Responsibilities

A plugin owns its visual interpretation, puzzle metadata, constraints, and rules.
It may depend on `core`, `vision` contracts, `rules` contracts, and the public
plugin API. It must not depend on concrete CV, storage, rendering, or desktop
automation adapters.

## Planned lifecycle

1. The registry discovers plugin metadata without performing heavy initialization.
2. Application composition supplies configured detector ports through a future
   plugin context.
3. The plugin creates a parser and a deterministically ordered rule catalog.
4. The parser produces a generic immutable board.
5. The generic solver evaluates plugin rules and retains their provenance.

## Rule guidelines

Each rule should represent one named inference, be stateless and deterministic,
inspect only its immutable context, return proposed transitions, and have positive,
negative, ambiguity, and contradiction fixtures. Rules cannot mutate the board or
call services.

The Cats namespace is only a contract placeholder. Use it as structural guidance,
not as evidence of a stable plugin API or implemented behavior.
