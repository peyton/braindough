# ChatGPT Extended Pro: focused_ultrasound_bridge

Status: completed with Browser at
https://chatgpt.com/c/6a09174e-4b1c-83e8-a2e1-56b77e30bc10.

## Prompt Summary

Plan a focused-ultrasound bridge for Braindough that is concrete enough to
implement in the repo while remaining a software-only, CI-safe experiment. The
scope explicitly excludes real sonication, acoustic simulation, device safety,
clinical efficacy, causal human neuromodulation, and human-subject inference.

## External Feedback Used

- Implement the MVP as a metadata-rich, claim-limited computational bridge.
- Keep CI fake-backend only and use optional bounded TRIBE v2 runs as model-card
  visual-response checks, not ultrasound predictions.
- Keep real acoustic fields null or explicitly out of scope. Do not store
  pressure, intensity, mechanical index, thermal index, temperature rise, skull
  model, transducer, or hydrophone values unless a future validated acoustic
  simulator exists.
- Include baseline, active proxy, transmit-blocked sham proxy, and spatial
  control proxy conditions.
- Record target/protocol metadata, claim-scope flags, and control design
  limitations in tables and reports.
- Prefer names such as `focused_ultrasound_bridge`,
  `focused_ultrasound_protocol_proxy`, and `software_dose_index`; avoid wording
  that implies treatment, dose delivery, or real neuromodulation.

## Implementation Decisions

- The suite generates synthetic protocol cards rather than applying a
  model-space acoustic operator. This keeps it compatible with the existing
  stimulus-generation pipeline and avoids implying that TRIBE v2 receives a real
  ultrasound exposure.
- Physical/acoustic reporting fields are present but blank in generated
  metadata. The suite records a `software_dose_index` and virtual envelope fields
  for deterministic fake-backend validation only.
- The fake backend may apply metadata-conditioned paired deltas so CI can verify
  tables, figures, and report behavior. These fake deltas are software
  validation artifacts only.
- Bounded TRIBE v2 runs, when requested locally, produce responses to the visual
  protocol cards. They do not predict response to sonication.

## Source Anchors Checked Locally

- ITRUSST reporting consensus:
  https://arxiv.org/abs/2402.10027
- Human LIFU methods primer:
  https://www.nature.com/articles/s43586-024-00368-6
- Human S1 focused-ultrasound study:
  https://www.nature.com/articles/nn.3620
- 2026 human tFUS/tEAS cautionary evidence and auditory-confound discussion:
  https://www.nature.com/articles/s41467-026-69853-8
- Transmit-blocked coupling-pad sham preprint:
  https://pubmed.ncbi.nlm.nih.gov/41659432/

## Claim Scope

This suite is a focused-ultrasound-inspired software bridge. It does not
simulate acoustic propagation, skull transmission, beam shape, pressure, heating,
cavitation, mechanical index, thermal index, device safety, or human
neuromodulation. It does not define a safe protocol. It does not support
clinical or causal claims. Its artifacts are suitable for provenance, report,
and software-control development.
