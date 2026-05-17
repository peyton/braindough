# Add a focused-ultrasound protocol bridge

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This repository does not check in `PLANS.md`; the plan follows the local instruction file at `~/.agents/PLANS.md`.

## Purpose / Big Picture

Braindough already has deterministic fake-backend experiments and optional bounded TRIBE v2 runs for stimulus perturbations. This change adds a focused-ultrasound-inspired bridge so a user can test target, sham, and spatial-control experiment bookkeeping without pretending the repo simulates ultrasound physics. After this change, `just run-fake-focused-ultrasound` writes a valid run with protocol metadata, paired response-delta tables, figures, and explicit claim-scope limitations. `just run-tribe-focused-ultrasound` provides an opt-in bounded local TRIBE path over the same synthetic protocol cards.

## Progress

- [x] (2026-05-17 01:21Z) Created branch `codex/focused-ultrasound-bridge` from current `origin/master`.
- [x] (2026-05-17 01:24Z) Used Browser to ask ChatGPT Extended Pro for focused-ultrasound experiment planning and critique.
- [x] (2026-05-17 01:28Z) Checked primary-source anchors for reporting, human LIFU framing, S1 targeting, 2026 tFUS/tEAS caveats, and transmit-blocked sham pads.
- [x] (2026-05-17 01:38Z) Added `focused_ultrasound_bridge` suite generation, fake-backend paired behavior, derived tables, report figures, YAML specs, and `just` targets.
- [x] (2026-05-17 01:45Z) Adjusted metadata to keep real acoustic fields blank and use software-only virtual envelope fields after ChatGPT feedback.
- [x] (2026-05-17 01:49Z) Added tests, README/docs updates, and research metadata capture.
- [x] (2026-05-17 01:33Z) Ran formatting/lint repair with `mise exec -- ruff format packages scripts tests` and `mise exec -- ruff check --fix packages scripts tests`.
- [x] (2026-05-17 01:34Z) Ran focused validation after `just bootstrap`: `mise exec -- pyright`, focused pytest over 50 tests, and `just research-validate`.
- [x] (2026-05-17 01:32Z) Ran full `just ci`; 66 tests passed, fixture validation passed, research validation passed, and all fake run targets including focused-ultrasound completed.
- [x] (2026-05-17 01:34Z) Ran `just artifact-validate` against the generated focused-ultrasound fake run and confirmed `{"valid": true}`.
- [ ] Open PR, monitor CI, merge when ready, and verify post-merge state.

## Surprises & Discoveries

- Observation: ChatGPT Extended Pro strongly recommended null or explicitly out-of-scope physical acoustic fields rather than nominal frequency, pressure, or duty-cycle values.
  Evidence: The captured response in `docs/research/2026-05-17-focused-ultrasound-bridge-chatgpt-extended-pro.md` says real acoustic fields should be null or out of scope unless a validated acoustic simulator exists.
- Observation: The best minimal bridge is a protocol/provenance assay, not a model-space perturbation that claims focal biological effects.
  Evidence: The implementation records `acoustic_modeling_status: not_modeled`, `itrusst_reporting_status: synthetic_proxy_fields_only`, and `safety_claim: software_proxy_no_sonication_or_clinical_claim`.
- Observation: Full local CI completed with the new fake focused-ultrasound target in the root `ci` recipe.
  Evidence: `just ci` ran 66 tests, fixture artifact validation, research validation, and produced `/Volumes/Virtual Machine HD/Projects/braindough/runs/2026/05/20260517T013207387468Z-smoke-fake-focused-ultrasound-bridge-fake-8045fa9f`.

## Decision Log

- Decision: Name the suite `focused_ultrasound_bridge`.
  Rationale: The name signals a bridge to FUS study design without claiming ultrasound delivery, acoustic simulation, treatment, or neuromodulation.
  Date/Author: 2026-05-17 / Codex
- Decision: Generate synthetic protocol cards as short videos instead of creating a new acoustic or neural-operator backend.
  Rationale: The current TRIBE integration consumes generated media; cards keep the run path shared with existing suites and make the limitation obvious.
  Date/Author: 2026-05-17 / Codex
- Decision: Keep pressure, intensity, mechanical index, thermal index, and related physical fields blank.
  Rationale: These values require real device/acoustic measurements or validated simulation; fake values would invite unsafe overclaiming.
  Date/Author: 2026-05-17 / Codex
- Decision: Let the fake backend apply small metadata-conditioned paired deltas only for software validation.
  Rationale: CI needs deterministic complete-pair artifacts, but fake-backend deltas must remain explicitly non-biological.
  Date/Author: 2026-05-17 / Codex

## Outcomes & Retrospective

The local implementation is complete and verified. `just ci` passed, and the generated focused-ultrasound fake artifact validates against the current schema. The remaining work is remote PR creation, GitHub CI monitoring, merge, and post-merge verification.

## Context and Orientation

The relevant code lives under `packages/braindough/src/braindough/`. `stimuli.py` creates generated images, video clips, and metadata. `suites.py` lists valid suite names. `backends/fake.py` produces deterministic CI-safe response arrays. `analysis.py` derives sidecar tables from stimuli and response arrays. `artifacts.py` writes outputs and validates required suite-specific files. `report.py` writes Markdown, HTML, PDF, and figures for each run. YAML specs live under `experiments/smoke/` for CI-safe fake runs and `experiments/local/` for opt-in local TRIBE v2 runs. The root `justfile` is the command surface.

Focused ultrasound here means low-intensity focused ultrasound neuromodulation as a research topic. In this repo, it is represented only as a software protocol/provenance proxy. No code computes acoustic propagation, skull transmission, real pressure, heating, cavitation, mechanical index, thermal index, safety, clinical endpoints, or human causal effects.

## Plan of Work

Add `focused_ultrasound_bridge` to `suites.py` and wire a generator in `stimuli.py`. The generator creates baseline, active proxy, transmit-blocked sham proxy, and spatial-control proxy cards over a small set of target labels such as `S1` and `hMT_plus`. It records target, condition, virtual envelope, and claim-scope metadata. Physical acoustic fields remain blank.

Extend `FakeBackend` so focused-ultrasound proxy children receive deterministic small paired deltas from their baseline parent. Extend `analysis.py` with `focused_ultrasound_protocols.csv` and `focused_ultrasound_comparisons.csv`. Extend `artifacts.py` so completed focused-ultrasound runs require those tables, delta arrays, and focused-ultrasound figures. Extend `report.py` with contact-sheet, protocol-effect, and software-dose figures plus a report section that states the claim limitation.

Add smoke and local TRIBE specs, `just run-fake-focused-ultrasound`, and `just run-tribe-focused-ultrasound`. Update README, architecture, experiment backlog, research capture docs, and tests. Validate with formatting, focused pytest, `just research-validate`, and `just ci`.

## Concrete Steps

From `/Users/peyton/.codex/worktrees/c71b/braindough`, run:

    git status --short --branch
    mise exec -- ruff format packages scripts tests
    mise exec -- ruff check --fix packages scripts tests
    mise exec -- uv run --python 3.12.13 pytest tests/test_stimuli.py tests/test_analysis.py tests/test_runner_artifacts.py tests/test_config.py tests/test_cli.py tests/test_tooling_contract.py -q
    just research-validate
    just ci

When local validation passes, push the branch, open a PR with a Conventional Commit title, monitor GitHub checks, and merge only after required checks pass.

## Validation and Acceptance

The feature is accepted when `just run-fake-focused-ultrasound` writes a completed artifact with `outputs/tables/focused_ultrasound_protocols.csv`, `outputs/tables/focused_ultrasound_comparisons.csv`, `outputs/deltas.npz`, `figures/focused_ultrasound_contact_sheet.png`, `figures/focused_ultrasound_protocol_effects.png`, `figures/focused_ultrasound_dose_proxy.png`, `report.md`, `report.html`, `executive_summary.pdf`, and `checksums.sha256`. `braindough validate <run-dir>` must return `{"valid": true}`. The report must state that the suite is a synthetic protocol/provenance bridge only and makes no acoustic, safety, clinical, or causal neuromodulation claim. `just ci` must pass from a clean checkout, and GitHub PR checks must pass before merge.

## Idempotence and Recovery

Generated run outputs go under `BRAINDOUGH_HOME` or test temporary directories, not git. Re-running fake experiments creates new timestamped run directories and does not modify tracked files. If formatting changes files, rerun tests. If TRIBE v2 is unavailable or slow, rely on the fake run for CI and leave the local TRIBE target as opt-in.

## Artifacts and Notes

Primary captured planning note:

    docs/research/2026-05-17-focused-ultrasound-bridge-chatgpt-extended-pro.md

New experiment specs:

    experiments/smoke/fake_focused_ultrasound_bridge.yaml
    experiments/local/tribe_v2_focused_ultrasound_bridge.yaml

Expected command output excerpt after validation:

    just research-validate
    {
      "valid": true
    }

    braindough validate <run-dir>
    {
      "valid": true
    }

## Interfaces and Dependencies

In `packages/braindough/src/braindough/stimuli.py`, `generate_stimuli()` must accept `focused_ultrasound_bridge` and produce `Stimulus` records with `parent_id`, `pair_id`, `target_label`, `condition`, `software_dose_index`, `virtual_envelope`, `acoustic_modeling_status`, and claim-scope metadata.

In `packages/braindough/src/braindough/analysis.py`, `build_derived_tables()` must return `focused_ultrasound_protocols` and `focused_ultrasound_comparisons`.

In `packages/braindough/src/braindough/report.py`, `write_report()` must write focused-ultrasound suite figures and a Markdown section when the manifest contains the suite.

No new third-party dependencies are required.

Plan update note, 2026-05-17: Initial plan written after implementation began so a future contributor can reconstruct the intent, scope boundaries, verification commands, and remaining PR lifecycle work.

Plan update note, 2026-05-17: Recorded successful local verification and the generated focused-ultrasound fake run path before starting the PR lifecycle.
