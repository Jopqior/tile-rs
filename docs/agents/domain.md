# Domain Docs

This repo uses a single-context layout: root `CONTEXT.md` and root `docs/adr/`.

## Before exploring

- Read root `CONTEXT.md`.
- Read ADRs in `docs/adr/` that touch the area being explored.

If these files do not exist, proceed silently. Do not flag their absence or suggest creating them upfront. `/domain-modeling` creates them lazily when terms or decisions get resolved.

## File structure

- `CONTEXT.md`: domain model and glossary.
- `docs/adr/`: architecture decision records.

## Use the glossary's vocabulary

When naming a domain concept in an issue title, refactor proposal, hypothesis, or test name, use the term defined in `CONTEXT.md`. Avoid synonyms the glossary explicitly excludes.

If a concept is missing, reconsider whether the project uses it. If it is a real gap, note it for `/domain-modeling`.

## Flag ADR conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding it:

> Contradicts ADR-0007 (event-sourced orders), but worth reopening because…
