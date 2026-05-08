# Deep Research: latent_network_ica_explorer

Status: launched in ChatGPT Deep Research; final report not yet extractable.

Conversation URL: https://chatgpt.com/c/69fe21d3-9fbc-83e8-ba25-4bef3a80e101

## Prompt

Use ChatGPT Deep Research. Please start the detailed report immediately and do
not ask clarifying questions. Research `latent_network_ica_explorer` for
Braindough, a public Python uv/mise/hk/just monorepo for local in-silico
neuroscience experiments using Meta/Facebook TRIBE v2. Heavy
models/cache/scratch live under `/Volumes/Virtual Machine HD/Projects/braindough`;
CI must use a deterministic fake backend; local TRIBE runs must be bounded.
Find state-of-the-art literature on ICA, PCA, sparse coding, latent
factorization, neural response decomposition, and interpretability in
computational neuroscience and multimodal models. Separate established claims
from speculative ones. Propose an end-to-end locally tractable experiment with
implementation details, artifacts, metrics, failure modes, and citations.

## Provisional Implementation While Research Runs

- Generate a compact visual-factor basis from deterministic base images and
  derived grayscale, blur, and edge variants.
- Decompose completed response rows with PCA/SVD-style components when at least
  two responses exist.
- Emit an explicit `insufficient_samples` component table when local TRIBE only
  produces one response.
- Write component summaries and stimulus loadings under `outputs/tables/`.
