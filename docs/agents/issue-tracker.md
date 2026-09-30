# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `Jopqior/tile-rs`. Use the `gh` CLI for all operations.

Always pass `--repo Jopqior/tile-rs` to `gh issue` and `gh pr` commands. This clone also has an upstream remote; do not rely on automatic repository selection. For `gh api`, use `repos/Jopqior/tile-rs/...` endpoints.

## Conventions

- Create: `gh issue create --repo Jopqior/tile-rs --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- Read: `gh issue view <number> --repo Jopqior/tile-rs --comments`. To fetch structured data including labels: `gh issue view <number> --repo Jopqior/tile-rs --json number,title,body,labels,comments`.
- List: `gh issue list --repo Jopqior/tile-rs --state open --json number,title,body,labels,comments`. Apply appropriate `--label` and `--state` filters.
- Comment: `gh issue comment <number> --repo Jopqior/tile-rs --body "..."`.
- Apply/remove labels: `gh issue edit <number> --repo Jopqior/tile-rs --add-label "..."` or `--remove-label "..."`.
- Close: `gh issue close <number> --repo Jopqior/tile-rs --comment "..."`.

## Pull requests as a triage surface

**PRs as a request surface: no.**

When set to `yes`, PRs use the same labels and states as issues:

- Read: `gh pr view <number> --repo Jopqior/tile-rs --comments` and `gh pr diff <number> --repo Jopqior/tile-rs`.
- List external PRs: `gh pr list --repo Jopqior/tile-rs --state open --json number,title,body,labels,author,authorAssociation,comments`. Keep `authorAssociation` of `CONTRIBUTOR`, `FIRST_TIME_CONTRIBUTOR`, or `NONE`; exclude `OWNER`, `MEMBER`, and `COLLABORATOR`.
- Comment/label/close: use `gh pr comment`, `gh pr edit`, and `gh pr close` with `--repo Jopqior/tile-rs`.

GitHub shares one number space across issues and PRs. Resolve a bare `#42` with `gh pr view 42 --repo Jopqior/tile-rs` and fall back to `gh issue view 42 --repo Jopqior/tile-rs`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --repo Jopqior/tile-rs --comments`.

## Wayfinding operations

Used by `/wayfinder`. The map is a single issue with child issues as tickets.

- Map: an issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body.
- Child ticket: link to the map as a GitHub sub-issue via `gh api`. Where sub-issues are unavailable, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`, `prototype`, `grilling`, `task`). Once claimed, assign the ticket to the driving dev.
- Blocking: use GitHub native issue dependencies: `gh api --method POST repos/Jopqior/tile-rs/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`. Obtain the numeric database ID with `gh api repos/Jopqior/tile-rs/issues/<n> --jq .id`, not the issue number or `node_id`. `issue_dependencies_summary.blocked_by` counts open blockers. Where dependencies are unavailable, use a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- Frontier query: list the map's open children, scoped to its sub-issues or task list. Exclude assigned tickets and those with open blockers. First in map order wins.
- Claim: `gh issue edit <n> --repo Jopqior/tile-rs --add-assignee @me`, the session's first write.
- Resolve: comment with the answer, close the ticket, then append a context pointer (gist + link) to the map's Decisions-so-far.
