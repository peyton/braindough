# Storage Contract

Braindough storage must be deterministic, external to the Git checkout, and
safe to delete or rebuild unless a path is explicitly documented as
user-provided source data.

## Path Roots

Default local root:

    /Volumes/Virtual Machine HD/Projects/braindough

Override with:

    BRAINDOUGH_HOME=/path/to/braindough

Expected layout:

- `shared/models/tribe-v2/facebook-tribev2-f894e783/`
- `shared/code/tribe-v2/72399081ed3f1040c4d996cefb2864a4c46f5b8e/`
- `shared/hf-cache/`
- `shared/torch-cache/`
- `blobs/sha256/`
- `worktrees/<sha256-realpath-12>/scratch`
- `worktrees/<sha256-realpath-12>/tmp`
- `worktrees/<sha256-realpath-12>/logs`
- `runs/YYYY/MM/<run_id>/`

## Run IDs

Use a microsecond timestamp plus experiment/backend hash:

    20260508T085212123456Z-smoke-fake-first-suite-fake-51bd2ac1

Re-running the same config creates a new run directory and must not overwrite an
existing manifest.

## Input Hashes

Record SHA-256 hashes for every user-provided input file, generated control, and
prepared media file. Hashes let later runs prove that two outputs used the same
stimulus without storing private or licensed source material in Git.

## Clean Checkout Rules

A clean checkout should be enough to run small fixture tests:

    just bootstrap
    just check
    just run-fake

Full TRIBE v2 inference may require model weights, TRIBE source, and larger
local media. The failure mode for missing weights or data should be explicit:
commands should write a skipped artifact naming the missing path or dependency.

## Retention

Generated data is disposable unless a manifest has been promoted to a reviewed
artifact. Cleanup should happen under `BRAINDOUGH_HOME` and must not touch user
source datasets outside that root.
