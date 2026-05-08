# Bootstrap Braindough as a Reproducible TRIBE v2 Research Workbench

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

The repository does not currently include `PLANS.md`; this plan follows the local instructions in `~/.agents/PLANS.md`. Keep this file self-contained so a future contributor can continue from the working tree alone.

## Purpose / Big Picture

Braindough should let a researcher run small, repeatable in-silico stimulus probes through TRIBE v2 and inspect model-predicted neural response artifacts without relying on hidden local state. After the bootstrap, a user should be able to clone the repo, run `just bootstrap`, run tests and lint checks with `just ci`, execute the fake first-suite experiment with `just run-fake`, attempt the local TRIBE v2 suite with `just run-tribe`, and inspect a manifest under `$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/manifest.json`.

The first working behavior is intentionally narrow: prepare simple visual or audio stimuli, run them through a TRIBE v2 wrapper or test double, write a manifest, and validate the artifact contract. Broader research ideas such as semantic stories, image-caption alignment, ROI-targeted search, and dataset adapters are deferred until the core path is reliable.

## Progress

- [x] (2026-05-08 08:42Z) Created the documentation slice under `docs/`, including architecture, storage, artifact, TRIBE v2, legal/data, backlog, and Deep Research status notes.
- [x] (2026-05-08 08:42Z) Recorded the first-suite experiment names: `image_activation`, `visual_controls`, `visual_perturbations`, `temporalization`, and `audio_controls`.
- [x] (2026-05-08 08:52Z) Added Python packaging under `pyproject.toml` using `uv`, with app code under `packages/braindough/src/braindough/` and tests under `tests/`.
- [x] (2026-05-08 08:52Z) Added a root `justfile` exposing `bootstrap`, `fmt`, `fmt-check`, `lint`, `typecheck`, `test`, `check`, `ci`, `doctor`, `storage-init`, `storage-doctor`, `run-fake`, `run-tribe`, `artifact-validate`, and `report`.
- [x] (2026-05-08 08:52Z) Implemented storage, artifact manifest, metrics, report, and checksum helpers.
- [x] (2026-05-08 08:52Z) Implemented the fake backend and lazy TRIBE v2 backend with skipped-report behavior.
- [x] (2026-05-08 08:52Z) Implemented generated stimuli for the five first-suite experiment families.
- [x] (2026-05-08 08:52Z) Added fixture tests and tooling-contract tests; `just ci` passed locally and produced a fake run.
- [x] (2026-05-08 08:56Z) Rechecked Browser Deep Research; it completed after 26 minutes with 11 citations and 529 searches, and the visible executive direction matched the implemented backend/experiment/artifact layering.

## Surprises & Discoveries

- Observation: At the start of this documentation slice, the branch only contained `README.md`, `LICENSE`, and `.gitignore`, so the docs define the intended bootstrap contract rather than describing existing code.
  Evidence: `rg --files` returned only `README.md` and `LICENSE`; `git ls-files` also listed `.gitignore`.

- Observation: TRIBE v2 is video/audio/text oriented, so still-image probes need an explicit temporalization step.
  Evidence: The TRIBE v2 model card and source repository describe inference from events derived from video, audio, and text, with predictions shaped as timesteps by cortical vertices.

## Decision Log

- Decision: Keep model access behind backend modules, especially `packages/braindough/src/braindough/backends/tribe_v2.py`.
  Rationale: A narrow backend wrapper lets experiments stay stable if TRIBE v2 installation, event preparation, or cache behavior changes.
  Date/Author: 2026-05-08 / Codex.

- Decision: Treat image experiments as short silent video experiments.
  Rationale: TRIBE v2's public interface is multimodal over video, audio, and text, not direct standalone image tensors.
  Date/Author: 2026-05-08 / Codex.

- Decision: Make manifests the source of truth for runs.
  Rationale: A manifest captures command, model revision, input hashes, output hashes, warnings, and redactions in one inspectable file.
  Date/Author: 2026-05-08 / Codex.

- Decision: Defer NSD, Algonauts, semantic-map stories, audiovisual mismatch, ROI search, and NeuroGen-style optimization.
  Rationale: These require data-access rules, stronger stimulus controls, or optimization infrastructure that would make the bootstrap too broad.
  Date/Author: 2026-05-08 / Codex.

## Outcomes & Retrospective

The documentation slice establishes the bootstrap contract and records the scientific and legal boundaries before code is added. The implementation remains incomplete until package, command, storage, artifact, wrapper, experiment, and test slices land.

## Context and Orientation

Braindough is a new repository. The root `README.md` currently contains only the project name, so this plan defines the initial system shape.

TRIBE v2 is a multimodal brain encoding model from Meta. A brain encoding model predicts brain response measurements from stimuli; in this project, the outputs are model predictions, not human measurements. TRIBE v2 predicts fMRI-style responses for naturalistic video, audio, and text. The public quick start returns an array shaped like `(n_timesteps, n_vertices)`, where timesteps are model time points and vertices are positions on a cortical surface mesh.

The Natural Scenes Dataset, or NSD, is a large 7T fMRI dataset for natural scene responses. Algonauts 2023 is a challenge that used NSD to benchmark models that predict responses to natural images. NeuroGen is a research framework for optimizing generated images toward target predicted brain activations. Semantic maps refer to voxel-wise models that map language or visual categories across cortex. Audiovisual integration refers to the brain combining auditory and visual information; the McGurk effect is a classic speech illusion where mismatched mouth movement and sound can change perceived syllables.

Key docs in this bootstrap:

- `docs/architecture.md`: intended module boundaries and command surface.
- `docs/storage.md`: local data, cache, run, and artifact paths.
- `docs/artifacts.md`: manifest and output schema.
- `docs/tribe-v2.md`: model facts and wrapper contract.
- `docs/legal-data.md`: license, privacy, and claims policy.
- `docs/experiment-backlog.md`: first suite and deferred research ideas.
- `docs/research/2026-05-08-deep-research-status.md`: external research status snapshot.

## Plan of Work

First, add project packaging and command runners. Create root and package `pyproject.toml` files with declared dependencies, a `packages/braindough/src/` layout, Ruff, Pyright, and Pytest configuration. Add a `justfile` that delegates to `mise` and `uv` and exposes the exact commands in this plan. The clean-checkout setup path is `just bootstrap`, followed by `just check` and `just ci`.

Next, add storage and artifact helpers. In `packages/braindough/src/braindough/storage.py`, define external-home discovery, path resolution, SHA-256 hashing, run ID generation, and safe directory creation. In `packages/braindough/src/braindough/artifacts.py`, define manifest creation, response serialization, metrics, checksums, and validation.

Then add the TRIBE v2 backend. In `packages/braindough/src/braindough/backends/tribe_v2.py`, define a lazy backend that imports heavy dependencies only when selected, tries MPS then CPU, avoids TRIBE text/TTS/WhisperX paths by using direct video/audio events, and writes skipped artifacts when local inference cannot run.

Then add stimulus helpers and YAML specs. `packages/braindough/src/braindough/stimuli.py` generates the first-suite stimulus families, while `experiments/smoke/fake_first_suite.yaml` and `experiments/local/tribe_v2_first_suite.yaml` select the fake or TRIBE backend.

Finally, add concise example configs and tests. Fixture tests should exercise the command path without requiring large models or datasets. Documentation should be updated if commands, paths, or manifest fields change.

## Concrete Steps

From the repository root:

    pwd
    # Expect: /Users/peyton/.codex/worktrees/664f/braindough or another clone path.

Install and verify the local toolchain:

    just bootstrap
    just fmt-check
    just lint
    just typecheck
    just test
    just check

Run the first suite with fixture or local data:

    just run-fake
    just run-tribe

Inspect one run:

    just artifact-validate RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"
    just report RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"

Expected successful summary shape:

    status: completed
    backend: fake
    responses: <stimulus-count>
    report: $BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/report.md

## Validation and Acceptance

The bootstrap is accepted when a clean checkout can run:

    just ci

The experiment path is accepted when `just run-fake` writes a run manifest, records input and output hashes, and can be validated with `just artifact-validate RUN_DIR=<run_dir>`.

Default tests must not require network access, model downloads, private datasets, or global Python packages. Full TRIBE v2 inference is tested separately with:

    just run-tribe

If full inference is unavailable, the command must fail with a clear message naming the missing cache, model, data path, or environment variable.

## Idempotence and Recovery

All setup and test commands should be safe to run repeatedly. Experiment commands must not overwrite an existing `manifest.json`; they should create a new run ID or fail before writing partial output.

If an experiment fails after creating a run directory, leave the partial directory in place with a log file and a failed manifest when possible. A later cleanup command may remove failed runs, but it must not touch user source datasets outside `BRAINDOUGH_HOME`.

Never write credentials, Deep Research browser contents, private stimuli, or raw human participant data to committed files.

## Artifacts and Notes

First-suite runs write under:

    $BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/

The required manifest fields are documented in `docs/artifacts.md`. The legal and data boundaries are documented in `docs/legal-data.md`.

Deep Research status: a ChatGPT Deep Research browser run at `https://chatgpt.com/c/69fd9a0f-5a78-83e8-bd4f-ba586989517e` was active at documentation dispatch time, then completed later in the implementation pass. Browser showed "Research completed in 26m - 11 citations - 529 searches" and a report titled "Braindough research agenda for local in-silico brain experiments." The full body was visible only inside an inaccessible report card, so committed docs record the visible status and source-backed compatible direction rather than a full transcript.

## Interfaces and Dependencies

Use Python with `uv`, `ruff`, `pyright`, and `pytest`. Use `just` as the public command runner. Do not mix Python environment managers.

Define the model backend in `packages/braindough/src/braindough/backends/tribe_v2.py`:

    TribeV2Backend.run(spec, stimuli, paths, run_dir) -> BackendResult

Define artifact helpers in `packages/braindough/src/braindough/artifacts.py`:

    RunArtifact.write_manifest(...)
    validate_artifact(run_dir) -> list[str]

Define storage helpers in `packages/braindough/src/braindough/storage.py`:

    BraindoughPaths.discover(...) -> BraindoughPaths
    sha256_file(path) -> str
    make_run_id(experiment_id, backend) -> str

The exact type names can change if the implementation language or framework requires it, but the command behavior, storage contract, and artifact contract should remain stable unless this plan is updated first.

## Revision Note

2026-05-08 / Codex: Created this plan as part of the documentation slice so implementation workers have one self-contained bootstrap contract. The plan records the first-suite scope, deferred research ideas, command surface, and validation expectations before code is added.
