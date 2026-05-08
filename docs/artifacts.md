# Artifact Contract

An artifact is the durable record of one Braindough experiment run. It must be
useful without the caller remembering command flags, local paths, or model
state.

## Run Directory Layout

Each run should write:

    $BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/
      manifest.json
      config.lock.json
      metrics.json
      events.ndjson
      checksums.sha256
      next_experiments.json
      outputs/
        responses.npz
        responses.index.jsonl
        tables/
          stimuli.csv
          response_metrics.csv
          perturbation_comparisons.csv
          latent_components.csv
          latent_loadings.csv
          optimization_history.jsonl
          objectives.json
      figures/
        response_similarity.png
        mean_abs_activation.png
        perturbation_deltas.png
        optimization_trace.png
      report.md
      report.html

Completed runs must include response outputs. Skipped or failed runs omit
response outputs, keep `outputs` empty, and record the reason in
`manifest.json`. Do not create placeholder outputs that look successful.

## Manifest Fields

`manifest.json` is the source of truth. Required fields:

- `schema_version`: start at `1`.
- `config.experiment_id`: the YAML experiment identifier.
- `run_id`: storage run ID.
- `status`: `completed`, `skipped`, or `failed`.
- `created_at` and `completed_at`: UTC timestamps.
- `workspace.git_sha` and `workspace.dirty`: repository state.
- `inputs`: paths, media types, SHA-256 hashes, and license notes.
- `backend`: backend name, package version, model reference, and model hash when
  available.
- `outputs`: generated file paths, hashes, and media types.
- `metrics`: machine-readable summary metrics.
- `blocker`: explicit reason when a run is skipped or failed.

## Shareable Exports

Shareable exports may include the manifest, metrics, figures, reports, and
small derived tables. They must not include:

- private source stimuli unless the license permits redistribution;
- raw fMRI data or subject-level human participant records;
- credentials, tokens, browser transcripts, or private Deep Research content;
- TRIBE v2 weights unless the license and distribution channel permit it.

## Artifact Commands

The implementation should expose:

    just artifact-validate RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"
    just report RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"
    just executive-summary

`just artifact-validate` fails when required files are missing, schema versions
are wrong, output paths are not relative, hashes do not match, a completed run
has no outputs, output table sidecars contain machine-local absolute paths, or
the manifest status is invalid.

## Derived Tables

Perturbation and optimization suites add small sidecar tables that are stable
enough for downstream agents:

- `stimuli.csv`: one row per generated stimulus, including suite, role, parent,
  pair, and response-present flags.
- `response_metrics.csv`: scalar response summaries by stimulus.
- `perturbation_comparisons.csv`: parent/child response deltas and
  correlations for perturbations, lesions, and counterfactual edits.
- `latent_components.csv` and `latent_loadings.csv`: PCA/SVD-style component
  summaries when enough responses exist; otherwise an explicit
  `insufficient_samples` row.
- `optimization_history.jsonl`: one row per candidate with score, diversity
  penalty, best-so-far, and replayable parameters.
- `objectives.json`: compact optimizer summary with the objective name, best
  candidate, and stopping reason.

## Provenance Notes

TRIBE v2 predictions are model predictions, not measurements. Figures and CSVs
must label them as predicted neural responses. When comparing suites, include
the same model revision and inference settings in every manifest before drawing
scientific conclusions.

## Executive Summary Exports

Executive summaries are generated reports across one or more run artifacts.
They are local outputs and should stay under external storage, normally:

    $BRAINDOUGH_HOME/reports/YYYY/MM/<timestamp>-executive-summary/
      executive-summary.pdf
      executive-summary.md
      executive-summary.json
      sources.json
      figures/
        suite_response_coverage.png
        mean_abs_by_suite.png
        optimizer_trace.png
        artifact_capability_comparison.png

Run discovery defaults to the latest fake and TRIBE perturbation/optimization
runs. The JSON file is intentionally path-neutral: it contains run IDs, backend
names, experiment IDs, chart-relative paths, source URLs, and scalar metrics,
but not machine-local run paths.

Use explicit run directories when comparing a specific pair:

    just executive-summary RUN_DIRS="$FAKE_RUN|$TRIBE_RUN"

The separator is `|` so paths under `/Volumes/Virtual Machine HD/...` can keep
their spaces. The PDF must distinguish fake-backend software validation from
TRIBE model-predicted responses and must not describe model predictions as
measured human data.
