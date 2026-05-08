# braindough

`braindough` is a local-first monorepo for brain-modeling experiments. The
first target is Meta/Facebook TRIBE v2, with deterministic fake-backend runs in
CI and heavyweight model/cache/run data kept outside the repository.

## Quick Start

```sh
mise install --locked
just bootstrap
just ci
just run-fake
```

Local heavy data defaults to:

```sh
export BRAINDOUGH_HOME="/Volumes/Virtual Machine HD/Projects/braindough"
```

TRIBE v2 runs are opt-in because they may download large model assets and use
machine-specific acceleration:

```sh
just storage-init
just run-tribe
```

If TRIBE v2 cannot run on the current machine, `run-tribe` writes an explicit
skipped artifact with the blocker instead of failing silently.

## Useful Commands

- `just bootstrap` installs pinned tools, syncs Python deps, and installs hooks.
- `just check` runs formatting checks, lint, type checking, tests, and artifact
  validation.
- `just run-fake` runs the CI-safe deterministic smoke experiment.
- `just run-tribe` attempts the first local TRIBE v2 suite.
- `just report RUN_DIR=...` rebuilds human-readable reports for an existing run.

See `docs/` for architecture, storage layout, artifact schema, TRIBE v2 notes,
and the experiment backlog.
