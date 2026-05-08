# Project Deep Research Report Capture

Status: partial capture from the in-app ChatGPT Deep Research conversation.

Source URL: https://chatgpt.com/c/69fd9a0f-5a78-83e8-92a3-8b0d3581258b

The full report body was visible in ChatGPT but not extractable through the
accessible browser DOM during this implementation pass. The captured visible
section titled "Candidate experiments for perturbation and optimization" named
the four required follow-up experiments for this PR:

- `latent_network_ica_explorer`: decompose predicted-response libraries with
  PCA, ICA, or related factor methods, then inspect components, loadings, and
  exemplar stimuli.
- `virtual_lesion_lab`: run stimulus-factor lesions or ablations and summarize
  response loss, delta maps, and sensitivity scoreboards.
- `discrete_stimulus_optimizer`: score a deterministic candidate library and
  optimize for high/low responses, diversity, and contrastive pairs.
- `counterfactual_editing_workbench`: create paired minimal edits and produce
  before/after response deltas, manifests, and replayable edit records.

The implementation should not claim hidden report text was captured. The
corresponding Deep Research conversations for each candidate were launched and
their prompts, URLs, and in-progress status are recorded in sibling files.

## Implementation Decisions Taken From The Visible Report

- Treat all four candidates as mandatory suites in this PR.
- Keep CI fake-backend-only and deterministic.
- Keep TRIBE v2 local-only, lazy-imported, and bounded by `max_predictions`.
- Store sidecar artifacts under `outputs/tables/` so agents and humans can
  evaluate deltas, objective traces, and decomposition status.
- For v1 `virtual_lesion_lab`, use stimulus-factor lesions rather than
  internal TRIBE hooks until a stable public ablation interface is identified.
