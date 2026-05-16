# TRIBE v2 Schematic-Response Preprint

Timestamp: 2026-05-16T233000Z

This directory contains a LaTeX preprint and small derivative artifacts for a
bounded local TRIBE v2 run:

`/Volumes/Virtual Machine HD/Projects/braindough/runs/2026/05/20260516T230219131728Z-local-tribe-v2-first-suite-tribe-v2-eac66ffd`

Build the PDF from the repository root:

```sh
just preprint-build docs/preprints/2026-05-16T233000Z-tribe-v2-schematic-response
```

The manuscript reports one model-predicted average-subject TRIBE v2 response
for one generated visual stimulus. It includes a vertex-index brain schematic,
not a subject-specific anatomical registration or ROI atlas.

Small committed derivatives:

- `figures/`: generated source image, response summary plots, and brain
  schematic.
- `tables/`: copied run metrics, response metrics, stimulus table, and
  normalized vertex-index schematic metadata.
- `review/`: concise ChatGPT Extended Pro critique notes captured during
  manuscript iteration.

Heavy model caches, full run outputs, and response arrays remain outside the
repo under `BRAINDOUGH_HOME`.
