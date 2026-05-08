# Deep Research: counterfactual_editing_workbench

Status: launched in ChatGPT Deep Research; final report not yet extractable.

Conversation URL: https://chatgpt.com/c/69fe21ea-4804-83e8-a487-72a70b148928

## Prompt

Use ChatGPT Deep Research. Please start the detailed report immediately and do
not ask clarifying questions. Research `counterfactual_editing_workbench` for
Braindough, a public Python uv/mise/hk/just monorepo for local in-silico
neuroscience experiments using Meta/Facebook TRIBE v2. Heavy
models/cache/scratch live under `/Volumes/Virtual Machine HD/Projects/braindough`;
CI must use a deterministic fake backend; local TRIBE runs must be bounded.
Find state-of-the-art literature on counterfactual explanations, minimal edits,
image/stimulus editing, causal counterfactuals, and edited-neighbor comparisons
in neuroscience and ML. Distinguish semantic-preserving edits from
meaning-changing edits. Compare rule-based edits, masked/local inpainting-style
edits, and constrained paired stimulus generation. Propose local
implementation, artifacts, metrics/thresholds, risks/license/data constraints,
now-vs-deferred scope, and citations.

## Provisional Implementation While Research Runs

- Generate paired base/edit stimuli with stable `pair_id` and `parent_id`
  metadata.
- Include color swap, local blur, background change, object mask, tile scramble,
  and temporal/static style edits.
- Report paired response deltas in the common perturbation comparison table.
- Keep generative inpainting and external datasets deferred for license/runtime
  control.
