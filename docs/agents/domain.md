# Domain docs

## Before exploring

This repo uses a single context: root `CONTEXT.md` for the glossary,
and `docs/adr/` for architectural decisions.

Read `CONTEXT.md` and ADRs relevant to the area being explored.
If a root `CONTEXT-MAP.md` is introduced, follow its pointers to the
relevant contexts and check their context-scoped ADRs too.

If these files are absent, proceed silently. The domain-modeling skill
creates them lazily when terms or decisions are resolved.

## Vocabulary

Use glossary terms in issue titles, proposals, hypotheses, and tests.
If a needed term is absent, reconsider the wording or note the gap
for domain-modeling.

## Decision conflicts

Explicitly flag contradictions with an existing ADR rather than
silently overriding it; explain why reopening the decision is warranted.
