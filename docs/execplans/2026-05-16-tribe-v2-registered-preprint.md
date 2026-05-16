# TRIBE v2 Schematic-Response Preprint ExecPlan

This plan tracks the scoped manuscript and artifact work for the May 16, 2026
TRIBE v2 run.

## Goal

Produce a timestamped LaTeX preprint for the structurally validated local TRIBE
v2 run at:

`/Volumes/Virtual Machine HD/Projects/braindough/runs/2026/05/20260516T230219131728Z-local-tribe-v2-first-suite-tribe-v2-eac66ffd`

The preprint should report a bounded, reproducible methods result: one real
TRIBE v2 prediction for a generated visual stimulus, plus a vertex-index
schematic visualization on a two-hemisphere brain diagram. It must not claim
measured neural activity, anatomical ROI localization, subject-specific
registration, or a new cognitive-neuroscience discovery.

## Progress

- [x] Created branch `codex/tribe-v2-registered-preprint` from current
  `origin/master`.
- [x] Confirmed the run artifact is structurally valid and contains one
  response shaped `2 x 20484`.
- [x] Added a small derivative artifact set under
  `docs/preprints/2026-05-16T233000Z-tribe-v2-schematic-response/`.
- [x] Draft LaTeX manuscript and README.
- [x] Run ChatGPT Extended Pro critique through Browser for multiple
  iterations; preserve concise critique notes.
- [x] Build PDF with `just preprint-build`.
- [x] Run `just check` and targeted preprint checks.
- [ ] Open PR, monitor CI, merge when green.

## Constraints

- Heavy run outputs, model weights, and scratch stay under `BRAINDOUGH_HOME`.
- The repo should only contain small derivative figures/tables and manuscript
  source/PDF.
- Claims must be grounded in the local run artifact and primary sources.
- ChatGPT critique is advisory; final claims remain limited to verified
  artifacts and cited sources.

## Acceptance

- `docs/preprints/2026-05-16T233000Z-tribe-v2-schematic-response/main.tex`
  builds to `main.pdf`.
- The preprint includes the brain schematic, response summary figures, run
  metrics, limitations, reproducibility commands, and source citations.
- Browser/ChatGPT critique notes document at least three review iterations.
- Local checks pass or any blocker is explicit.
- PR is merged after required checks pass.
