# Issue tracker: GitHub

Issues and specs live in `Jopqior/tile-rs`, not the upstream repository.
Use `gh` with `--repo Jopqior/tile-rs`; API paths use
`repos/Jopqior/tile-rs`.

## Conventions

- Create: `gh issue create --repo Jopqior/tile-rs --title "..." --body-file <file>`.
- Read: `gh issue view <number> --repo Jopqior/tile-rs --comments`.
  Fetch labels and other metadata with `--json` as needed.
- List: `gh issue list --repo Jopqior/tile-rs --state open --json number,title,body,labels,assignees`, with appropriate filters.
- Comment: `gh issue comment <number> --repo Jopqior/tile-rs --body-file <file>`.
- Labels: `gh issue edit <number> --repo Jopqior/tile-rs --add-label "..."` / `--remove-label "..."`.
- Close: `gh issue close <number> --repo Jopqior/tile-rs`.

“Publish to the issue tracker” means create a GitHub issue.
“Fetch the relevant ticket” means read the issue and its comments.

## Pull requests as a triage surface

**PRs as a request surface: no.**

If enabled later, use `gh pr` equivalents and include external authors
with association CONTRIBUTOR, FIRST_TIME_CONTRIBUTOR, or NONE.
Issues and PRs share a number space; distinguish them when resolving
an ambiguous reference.

## Wayfinding operations

- Map: an issue labelled `wayfinder:map`, using the wayfinder map body.
- Child: create an issue labelled `wayfinder:research`, `wayfinder:prototype`,
  `wayfinder:grilling`, or `wayfinder:task`. Link it with
  `gh api --method POST repos/Jopqior/tile-rs/issues/<map>/sub_issues -F sub_issue_id=<child-database-id>`.
- IDs: obtain numeric database IDs through
  `gh api repos/Jopqior/tile-rs/issues/<number> --jq .id`;
  these are not issue numbers or node IDs.
- Blocking: use native dependencies:
  `gh api --method POST repos/Jopqior/tile-rs/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-database-id>`.
- Fallback: only if native relationships are unavailable, use a map
  task list plus `Part of #<map>` in children, and `Blocked by: #<n>`
  for dependencies.
- Frontier: fetch all pages of the map's sub-issues in tracker order;
  keep open, unassigned children with no open blockers.
  Check `issue_dependencies_summary.blocked_by`, or resolve fallback
  blocker states. Choose the first eligible child.
- Claim: before working a ticket, assign it using
  `gh issue edit <number> --repo Jopqior/tile-rs --add-assignee @me`.
- Resolve: post a resolution comment, close the ticket, then append
  a linked title and one-line gist to the map's Decisions so far.
- Concurrency: re-read the ticket before claiming and the map before
  updating; preserve other sessions' changes.

Use linked issue titles in human-facing text, not bare numbers.
