# Deep Research: virtual_lesion_lab

Status: launched in ChatGPT Deep Research; final report not yet extractable.

Conversation URL: https://chatgpt.com/c/69fe21da-4604-83e8-aff7-17bc4f2d0a40

## Prompt

Use ChatGPT Deep Research. Please start the detailed report immediately and do
not ask clarifying questions. Research `virtual_lesion_lab` for Braindough, a
public Python uv/mise/hk/just monorepo for local in-silico neuroscience
experiments using Meta/Facebook TRIBE v2. Heavy models/cache/scratch live under
`/Volumes/Virtual Machine HD/Projects/braindough`; CI must use a deterministic
fake backend; local TRIBE runs must be bounded. Find state-of-the-art literature
on virtual lesions, ablations, causal perturbation, in-silico lesioning,
representational knockout, and descriptive sensitivity analysis. Distinguish
causal inference from sensitivity analysis. Propose local implementation paths
including feature/channel ablation, subspace lesion, stimulus-factor lesions,
artifacts, metrics, risks, and citations.

## Provisional Implementation While Research Runs

- Use stimulus-factor lesions instead of internal TRIBE layer hooks.
- Generate baseline clips and paired `mask_left`, `mask_right`,
  `central_occlusion`, `low_contrast`, `blur_suppression`, and
  `blank_suppression` variants.
- Report parent/lesion response deltas and correlations.
- Defer model-internal ablation until a stable TRIBE public hook is identified.
