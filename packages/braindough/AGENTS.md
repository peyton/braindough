# Braindough Package Notes

- Keep top-level imports lightweight. Heavy ML/runtime dependencies belong in
  backend modules and should be imported inside execution paths.
- Prefer typed dataclasses and plain JSON-compatible structures for artifact
  boundaries.
- New behavior should have tests under `/Users/peyton/.codex/worktrees/664f/braindough/tests`.
