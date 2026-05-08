# Deep Research: discrete_stimulus_optimizer

Status: launched in ChatGPT Deep Research; final report not yet extractable.

Conversation URL: https://chatgpt.com/c/69fe21e2-2fd0-83e8-90b5-3742d50ec004

## Prompt

Use ChatGPT Deep Research. Please start the detailed report immediately and do
not ask clarifying questions. Research `discrete_stimulus_optimizer` for
Braindough, a public Python uv/mise/hk/just monorepo for local in-silico
neuroscience experiments using Meta/Facebook TRIBE v2. Heavy
models/cache/scratch live under `/Volumes/Virtual Machine HD/Projects/braindough`;
CI must use a deterministic fake backend; local TRIBE runs must be bounded.
Find state-of-the-art literature on discrete optimization, combinatorial
search, neuroevolution, Bayesian optimization over discrete spaces, active
learning, adversarial stimulus search, and model-based stimulus design. Compare
greedy/beam search, evolutionary search, and surrogate-guided search. Propose a
locally tractable experiment with exact artifacts, metrics/acceptance criteria,
risks/license/data constraints, now-vs-deferred scope, and citations.

## Provisional Implementation While Research Runs

- Generate a seeded finite candidate library from shape, palette, and angle
  parameters.
- Score completed responses with whole-response mean absolute activation minus
  a diversity penalty based on correlation to prior candidates.
- Record every candidate in `optimization_history.jsonl`.
- Record the best candidate and stopping reason in `objectives.json`.
