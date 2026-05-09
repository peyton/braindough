# BOLD5000 Metadata-Control Preprint

Timestamp: 2026-05-09T200240Z

This directory contains a LaTeX preprint and small supporting artifacts for the
BOLD5000 Release 1.0 metadata-control benchmark.

Build the PDF from the repository root:

```sh
just preprint-build docs/preprints/2026-05-09T200240Z-bold5000-metadata-control
```

Primary committed experiment configs:

- `experiments/local/bold5000_roi_encoding_preprint_tr34.yaml`
- `experiments/local/bold5000_roi_encoding_preprint_tr3.yaml`
- `experiments/local/bold5000_roi_encoding_preprint_tr34_grouped_sensitivity.yaml`

The manuscript reports only measured BOLD5000 ROI benchmark results produced by
those configs. Heavy run artifacts remain under `BRAINDOUGH_HOME`; the copied
figures plus result, split-balance, and leakage-diagnostic CSV summaries here
are small, reviewable derivatives.
