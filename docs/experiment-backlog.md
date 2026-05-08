# Experiment Backlog

This backlog separates the first bootstrap suite from deferred research ideas so
the initial implementation stays small and verifiable.

## First Suite

These suites are in scope for the bootstrap implementation:

### `image_activation`

Purpose: establish the simplest end-to-end path from one image to a TRIBE v2
prediction artifact.

Command:

    just run-fake
    just run-tribe

Acceptance: the command writes a run under
`$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/` with `manifest.json`,
`outputs/responses.npz`, `outputs/responses.index.jsonl`, `metrics.json`,
`events.ndjson`, `report.md`, `report.html`, figures, checksums, and
`next_experiments.json`.

### `visual_controls`

Purpose: compare a target image with control images that preserve selected
low-level properties.

Command:

    just run-fake
    just run-tribe

Acceptance: the manifest records every generated control, its hash, and a
per-control predicted activation summary.

### `visual_perturbations`

Purpose: test how controlled visual changes alter predictions while timing and
audio stay fixed.

Command:

    just run-fake
    just run-tribe

Acceptance: outputs include a perturbation table with one row per transform and
summary metrics for each predicted response.

### `temporalization`

Purpose: measure whether still-image-to-video policy changes the prediction.

Command:

    just run-fake
    just run-tribe

Acceptance: outputs compare hold, pan, zoom, loop, and cut policies with the
same source image hash.

### `audio_controls`

Purpose: isolate audio effects through silence, tones, noise, speech, and
audio-only or video-only baselines.

Command:

    just run-fake
    just run-tribe

Acceptance: outputs label every audio condition and record whether the visual
track was present, muted, or replaced.

## Perturbation And Optimization PR

These suites are now in scope for the perturbation/optimization implementation:

### `latent_network_ica_explorer`

Purpose: discover response components across controlled visual factors and make
the sample-count limit explicit.

Command:

    just run-fake-optimization
    just run-tribe-optimization

Acceptance: outputs include `latent_components.csv`, `latent_loadings.csv`, and
an `insufficient_samples` component row when fewer than two predictions are
available.

### `virtual_lesion_lab`

Purpose: compare baseline stimuli with v1 stimulus-factor lesions without
claiming internal model ablation.

Acceptance: outputs include parent/lesion response deltas and correlations in
`perturbation_comparisons.csv`; model-internal layer ablation remains deferred.

### `discrete_stimulus_optimizer`

Purpose: run a deterministic finite search over generated candidates and rank
them by a response objective with a diversity penalty.

Acceptance: outputs include `optimization_history.jsonl`, `objectives.json`,
and a report summary naming the best candidates and stopping reason.

### `counterfactual_editing_workbench`

Purpose: create replayable paired minimal edits and quantify predicted-response
deltas between each base stimulus and edit.

Acceptance: every edit has parent/pair metadata, and pair deltas appear in the
common perturbation comparison table.

## Deferred Ideas

### Image-caption Alignment

Compare an image, its caption as text or speech, and mismatched captions. This
is deferred until the text/audio event timing contract is stable.

### Audiovisual Congruence and Mismatch

Probe congruent and incongruent audio-video pairs, including speech-inspired
McGurk-style controls. This should wait for a careful stimulus policy because
classic audiovisual speech effects depend on precise timing and stimulus design.

Source anchors:

- McGurk and MacDonald, "Hearing lips and seeing voices":
  https://www.nature.com/articles/264746a0
- pSTS audiovisual processing review article:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2536697/

### Semantic-map Story Probes

Use short narrative stories to compare predicted language responses against
semantic-map expectations. This is deferred until text fixtures and ROI labels
are stable.

Source anchor:

- Huth et al., "Natural speech reveals the semantic maps that tile human
  cerebral cortex": https://www.nature.com/articles/nature17637

### ROI-targeted Stimulus Search

Search over stimulus variants for high or low predicted response in selected
regions of interest. This needs a pinned ROI map, stable metrics, and safeguards
against overinterpreting model predictions.

### NeuroGen-style Optimization

Generate or optimize images against neural objectives. This is deferred because
it introduces a generator, optimization loop, and stronger legal/research claims.

Source anchor:

- NeuroGen: https://www.sciencedirect.com/science/article/pii/S1053811921010831

### Algonauts and NSD Adapters

Add dataset adapters for benchmark-style evaluation and fixture generation only
after access terms and storage boundaries are documented.

Source anchors:

- NSD: https://naturalscenesdataset.org/
- Algonauts 2023: https://algonautsproject.com/2023/index.html
