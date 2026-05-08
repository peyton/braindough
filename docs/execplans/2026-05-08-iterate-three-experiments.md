# Iterate Three Braindough Experiments

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This plan follows `~/.agents/PLANS.md`.

## Purpose / Big Picture

Braindough can already run a first TRIBE v2-oriented experiment suite and a broad perturbation/optimization suite, but the second suite is too coarse for iterative work. After this change, a user can run the three most immediately useful experiment families independently: a virtual lesion lab, a discrete stimulus optimizer, and a counterfactual editing workbench. Each focused run will produce richer machine-readable artifacts, suite-specific charts, and an executive summary PDF that explains what happened without requiring the user to inspect raw arrays.

The result is observable through repo-local commands such as `just run-fake-lesion`, `just run-fake-optimizer`, `just run-fake-counterfactual`, and `just ci`. Completed fake runs must validate with `braindough validate` and include `executive_summary.pdf`, richer tables under `outputs/tables/`, and suite-specific figures under `figures/`.

## Progress

- [x] (2026-05-08T22:53:41Z) Created branch `codex/iterate-braindough-experiments` from the clean detached checkout after `git fetch origin`.
- [x] (2026-05-08T22:53:41Z) Reviewed the current artifact, stimulus, fake backend, TRIBE backend, report, and CI command surfaces.
- [x] (2026-05-08T23:12:20Z) Added focused fake/local specs and root `just` targets for the three chosen experiment families.
- [x] (2026-05-08T23:12:20Z) Added pair-aware TRIBE prediction budgets through `backend_config.max_predictions_by_suite`.
- [x] (2026-05-08T23:12:20Z) Added configurable lesion/counterfactual generation, stable optimizer cataloging, sham/random-patch controls, and corrected low-contrast behavior.
- [x] (2026-05-08T23:12:20Z) Added suite-specific derived tables, delta arrays, richer optimizer status accounting, and summary metrics.
- [x] (2026-05-08T23:12:20Z) Added suite-specific report sections, charts, and per-run `executive_summary.pdf`.
- [x] (2026-05-08T23:12:20Z) Added and updated tests, docs, and CI wiring.
- [x] (2026-05-08T23:12:20Z) Ran `just fmt`, `just check`, `just ci`, and the focused fake runs.
- [x] (2026-05-08T23:37:56Z) Fixed final review gaps for suite figure manifest outputs, checksum validation, lesion-only top-delta rows, empty delta sidecars for partial pair suites, and optimizer trace status filtering.
- [x] (2026-05-08T23:38:04Z) Re-ran `just check`, `just ci`, and the three focused fake run targets after the review fixes.
- [x] (2026-05-08T23:46:51Z) Fixed re-review gaps in cross-run executive summary optimizer charts and shared figure validation, then re-ran local validation.
- [x] (2026-05-08T23:53:59Z) Added backend-error executive-summary regression coverage, re-ran local validation, and produced `/Volumes/Virtual Machine HD/Projects/braindough/reports/2026-05-08-iterate-three-experiments/executive-summary.pdf` from the three latest focused fake runs.
- [x] (2026-05-08T23:53:59Z) Decided not to run focused local TRIBE targets in this turn because each target can require heavyweight TRIBE model setup and multi-minute predictions; fake CI remains the merge gate and partial real runs now preserve incomplete-pair accounting.
- [ ] Open a PR, monitor GitHub CI, fix failures, and merge when green.

## Surprises & Discoveries

- Observation: The existing local combined TRIBE optimization spec uses `max_predictions_per_suite: 1`, which prevents lesion and counterfactual parent/child pairs from both being predicted in a single suite.
  Evidence: `packages/braindough/src/braindough/backends/tribe_v2.py` enforces per-suite attempted prediction limits before prediction.

- Observation: The expanded fake CI gate is still practical locally.
  Evidence: `just ci` completed successfully on 2026-05-08 with `52 passed in 36.64s` and produced fake run artifacts for the first suite, combined optimization suite, virtual lesion lab, discrete optimizer, and counterfactual workbench.

- Observation: The in-app Browser path cannot open the local generated `file://` report artifact in this environment.
  Evidence: Browser policy blocked `file:///Volumes/Virtual%20Machine%20HD/Projects/braindough/runs/.../report.html`; report and PDF verification therefore relies on CLI artifact validation and tests.

## Decision Log

- Decision: Iterate `virtual_lesion_lab`, `discrete_stimulus_optimizer`, and `counterfactual_editing_workbench`, while keeping `latent_network_ica_explorer` unchanged except for shared artifact/report compatibility.
  Rationale: These three experiments produce directly inspectable pair deltas, objective traces, and replayable edit/lesion records even with bounded fake and TRIBE runs. The latent explorer becomes more valuable when more real responses exist.
  Date/Author: 2026-05-08 / Codex

- Decision: Generate `executive_summary.pdf` with Matplotlib `PdfPages`.
  Rationale: The project already depends on Matplotlib, so this avoids browser, Chromium, WeasyPrint, or system font dependencies in clean CI.
  Date/Author: 2026-05-08 / Codex

## Outcomes & Retrospective

No implementation outcome has been completed yet. This section will be updated after local verification and again after PR CI.

2026-05-08T23:12:20Z: Local implementation is complete enough for review. `just check` and `just ci` pass locally, focused fake artifacts validate, and per-run PDFs are present. Remaining work is subagent review, optional focused local TRIBE feasibility, PR creation, GitHub CI, and merge.

2026-05-08T23:39:00Z: Final review gaps were patched. Fresh local verification passed: `just check` (`52 passed`), `just ci` (`52 passed` plus five fake run targets), and explicit `just run-fake-lesion && just run-fake-optimizer && just run-fake-counterfactual`. Focused local TRIBE targets were not run before PR because bounded TRIBE predictions are still heavyweight relative to this CI-oriented iteration.

2026-05-08T23:53:59Z: Re-review found two remaining gaps in shared figure validation and the cross-run executive summary optimizer chart, followed by one backend-error test coverage gap. All were patched. Fresh verification passed: `just fmt && just check` (`55 passed`), `just ci` (`55 passed` plus five fake run targets), explicit focused fake runs, direct validation of the three latest focused run directories, and `just executive-summary RUN_DIRS=... OUTPUT_DIR=/Volumes/Virtual Machine HD/Projects/braindough/reports/2026-05-08-iterate-three-experiments`.

## Context and Orientation

Braindough is a Python monorepo. The package lives under `packages/braindough/src/braindough/`, tests live under `tests/`, experiment YAML files live under `experiments/`, and common commands live in the root `justfile`. Heavy data, model caches, generated stimuli, and run outputs must stay under `BRAINDOUGH_HOME`, which defaults to `/Volumes/Virtual Machine HD/Projects/braindough`, not inside the repository.

An experiment run follows this flow: `packages/braindough/src/braindough/config.py` loads a YAML spec, `runner.py` creates a run directory, `stimuli.py` generates deterministic media, a backend in `backends/` predicts responses, `artifacts.py` writes responses, tables, metrics, manifest, reports, and checksums, and `report.py` generates human-readable output and figures.

The fake backend is deterministic and CI-safe. The TRIBE v2 backend imports heavyweight TRIBE dependencies lazily and can skip with an explicit blocker when the local machine cannot run it. TRIBE runs must remain bounded because the model and feature extractors are large and slow.

## Plan of Work

First, add focused experiment specs under `experiments/smoke/` and `experiments/local/` for the virtual lesion lab, discrete optimizer, and counterfactual workbench. Add matching `just` targets and include the three fake focused targets in `just ci`. Update CLI and tooling tests so the new specs and targets are part of the repo contract.

Second, add `max_predictions_by_suite` support to the TRIBE backend. This mapping will let local specs request more predictions for pair-oriented suites without weakening the global `max_predictions` limit. If both `max_predictions_per_suite` and `max_predictions_by_suite` are present, the suite-specific mapping wins for named suites and the scalar limit remains the fallback.

Third, improve stimulus generation. `virtual_lesion_lab` should accept config keys for base count, lesion types, lesion strengths, and sham controls. Its `low_contrast` lesion must reduce contrast by blending toward gray or using `ImageEnhance.Contrast`, not autocontrast. `discrete_stimulus_optimizer` should write stable candidate parameter hashes in metadata. `counterfactual_editing_workbench` should accept base count and edit type config and write enough metadata to replay edits.

Fourth, expand derived analysis and artifacts. `artifacts.py` should write new CSV/JSONL tables and a `deltas.npz` array sidecar from `analysis.py`. Pair-oriented tables must include incomplete rows when a parent or child response is missing, so bounded TRIBE runs explain budget gaps instead of silently omitting pairs. Optimizer history must account for unscored candidates and record a versioned objective summary.

Fifth, improve reporting. `report.py` should generate suite-specific charts and a compact `executive_summary.pdf`. The report text must label all responses as model-predicted, descriptive sensitivity results rather than causal measurements. Completed artifacts must include the PDF in the manifest and checksums, and validation must require it.

Finally, update docs and tests. Add unit tests for stimulus config, lineage, low contrast, optimizer hashes, objective tie breaks, incomplete pairs, delta metrics, and PDF creation. Add focused runner tests for the three fake specs. Run the repo's native commands before opening the PR.

## Concrete Steps

All commands run from `/Users/peyton/.codex/worktrees/22a0/braindough`.

1. Prepare branch:

       git fetch origin
       git switch -c codex/iterate-braindough-experiments

2. Implement the code and tests described above using `apply_patch` and repo-local commands only.

3. Verify locally:

       just fmt
       just check
       just ci
       just run-fake-lesion
       just run-fake-optimizer
       just run-fake-counterfactual

4. Attempt focused local TRIBE runs only if time and storage are reasonable:

       just run-tribe-lesion
       just run-tribe-optimizer
       just run-tribe-counterfactual

   If these skip or are stopped for runtime reasons, preserve the skipped artifacts and record the blocker.

5. Open and merge the PR:

       git push -u origin codex/iterate-braindough-experiments
       gh pr create --title "feat: iterate braindough experiment artifacts" --body-file <generated-body>
       gh pr checks --watch
       gh pr merge --squash --delete-branch

## Validation and Acceptance

Acceptance requires `just ci` to pass locally and on GitHub. The fake CI run must execute the first suite, the combined perturbation/optimization suite, and the three focused fake suites. Each completed fake run must validate, include `executive_summary.pdf`, include suite-specific tables under `outputs/tables/`, include suite-specific figures, and contain no absolute local paths in manifest, metrics, reports, or table outputs.

For `virtual_lesion_lab`, acceptance means a fake lesion run includes lesion provenance, lesion comparisons, ROI-ready summaries, top delta vertices, and delta arrays. For `discrete_stimulus_optimizer`, acceptance means all candidates are cataloged, scored or marked with a status, and summarized through a versioned objective. For `counterfactual_editing_workbench`, acceptance means every base/edit pair has replay metadata, edit magnitude metrics, response delta fields, and complete or incomplete status.

## Idempotence and Recovery

All new run commands write under `BRAINDOUGH_HOME`, so rerunning them should create new timestamped run directories without modifying tracked source files. Formatting is deterministic through `just fmt`. If a TRIBE run is too slow, stop it and rely on fake CI plus the skipped or partial artifact blocker. Do not delete unrelated run directories or model caches.

If a GitHub PR merge reports a local cleanup error because another worktree owns `master`, verify the remote PR state with `gh pr view --json state,mergedAt,mergeCommit`, then delete only the remote branch if needed.

## Artifacts and Notes

The main generated artifacts for completed runs should include:

    outputs/tables/lesion_manifest.csv
    outputs/tables/lesion_comparisons.csv
    outputs/tables/lesion_roi_summary.csv
    outputs/tables/top_delta_vertices.csv
    outputs/tables/candidate_catalog.jsonl
    outputs/tables/optimization_history.jsonl
    outputs/tables/objectives.json
    outputs/tables/counterfactual_edits.jsonl
    outputs/tables/counterfactual_pairs.csv
    outputs/deltas.npz
    figures/lesion_scoreboard.png
    figures/optimization_score_components.png
    figures/counterfactual_tradeoff.png
    executive_summary.pdf

## Interfaces and Dependencies

Use only existing runtime dependencies: NumPy, Pillow, ImageIO, Matplotlib, and PyYAML. Do not add browser/PDF system dependencies.

In `packages/braindough/src/braindough/analysis.py`, keep `build_derived_tables(stimuli, responses) -> dict[str, list[dict[str, Any]]]` as the table entry point, and add helper functions for suite-specific rows. Add a separate function for delta arrays so `artifacts.py` can write `outputs/deltas.npz`.

In `packages/braindough/src/braindough/report.py`, keep `write_report(run_dir) -> tuple[Path, Path]` compatible, but add the PDF as a side effect and expose enough helper behavior for tests.

In `packages/braindough/src/braindough/backends/tribe_v2.py`, treat `backend_config.max_predictions_by_suite` as a mapping from suite name to integer prediction limit.

Revision note: Initial ExecPlan created before implementation to satisfy the repo's complex-feature workflow and make the work restartable from this file.
