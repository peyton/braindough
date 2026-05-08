# Braindough Agent Notes

## Workflow

- Use the root `justfile` for all routine tasks. Add new commands there before
  documenting them elsewhere.
- Use `mise` for tool versions and keep `mise.lock` updated with `mise lock`.
- Use `uv` for all Python dependency and command execution.
- Keep CI fake-backend only unless a future workflow has explicit credentials,
  storage, and hardware assumptions.
- Heavy models, downloaded code, caches, scratch, and run outputs belong under
  `BRAINDOUGH_HOME`, defaulting to `/Volumes/Virtual Machine HD/Projects/braindough`.

## Coding

- Keep the package import-light. TRIBE v2, torch, moviepy, pandas, and other
  heavyweight dependencies must be imported lazily in backend-specific modules.
- Do not add runtime dependency installation. If a command needs a tool or
  Python package, declare it in `mise.toml` or `pyproject.toml`.
- Store shareable run metadata with relative paths and hashes. Absolute local
  paths are allowed only in local-only diagnostics.
- Preserve the Unlicense for this repo. TRIBE v2 weights and related assets have
  their own licenses and must not be vendored here.

## Verification

- Before handing off code changes, run `just ci`.
- For TRIBE v2 work, run `just run-tribe` when the local machine has time and
  storage; otherwise make sure the skipped report explains the blocker.
