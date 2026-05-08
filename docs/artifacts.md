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
      figures/
        response_similarity.png
        mean_abs_activation.png
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

`just artifact-validate` fails when required files are missing, schema versions
are wrong, output paths are not relative, hashes do not match, a completed run
has no outputs, or the manifest status is invalid.

## Provenance Notes

TRIBE v2 predictions are model predictions, not measurements. Figures and CSVs
must label them as predicted neural responses. When comparing suites, include
the same model revision and inference settings in every manifest before drawing
scientific conclusions.
