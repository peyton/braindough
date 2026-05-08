# TRIBE v2 Integration Notes

TRIBE v2 is the first model target for Braindough. The Hugging Face model card
describes it as a multimodal brain encoding model that predicts fMRI responses
to naturalistic video, audio, and text stimuli. The Meta paper page describes a
tri-modal foundation model trained on a unified fMRI corpus across many subjects
and evaluated on naturalistic and experimental conditions.

## Source Facts

Use these sources as the stable baseline:

- Model card and weights: https://huggingface.co/facebook/tribev2
- Source repository: https://github.com/facebookresearch/tribev2
- Meta paper page: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/

Important integration facts:

- License is `cc-by-nc-4.0`; treat the model and derived use as non-commercial
  unless legal review says otherwise.
- Inputs are represented as timed events derived from video, audio, or text.
- The public quick start returns predictions shaped like
  `(n_timesteps, n_vertices)`.
- The GitHub README says predictions are for an average subject, live on the
  `fsaverage5` cortical mesh at roughly 20k vertices, and are shifted to account
  for hemodynamic lag.
- The model card names LLaMA 3.2, V-JEPA2, and Wav2Vec-BERT as feature
  extractors combined by a Transformer architecture.

## Wrapper Boundary

Only `packages/braindough/src/braindough/backends/tribe_v2.py` should import
`tribev2`, `torch`, or `pandas`. Experiment code calls the backend registry
instead of reaching into the third-party package directly.

The backend provides:

- `TribeV2Backend.run(spec, stimuli, paths, run_dir)`: prepare direct
  video/audio events and return `BackendResult`.
- A device ladder that tries MPS when available, then CPU.
- Skipped artifacts when imports, model loading, or predictions fail.

## First Suite Usage

The first suite uses TRIBE v2 conservatively:

- `image_activation`: convert a still image to a short silent video and record
  predicted activation.
- `visual_controls`: compare the same temporalization against visual controls.
- `visual_perturbations`: vary visual content while holding timing constant.
- `temporalization`: vary timing while holding visual content constant.
- `audio_controls`: vary audio while holding visual content constant, and also
  support audio-only controls.

Run surface:

    just run-tribe

The checked-in local spec sets `max_predictions: 1` so the default laptop run
can produce at least one TRIBE-backed response without spending hours on the
full generated suite. Remove or raise that cap for longer local studies after
the first artifact is validated.

TRIBE v2's public checkpoint emits 100 prediction steps, so the checked-in
local spec also uses a 50-second event duration for short generated clips. The
visual content is still a short generated MP4; the longer event window keeps the
wrapper's segment mask aligned with the checkpoint output.

## Interpretation Rules

Write "predicted response" or "model-predicted activation", not "brain scan" or
"measured neural activity". Avoid claims about individual people because the
default public example is an average-subject prediction. Do not call outputs
diagnostic or medical.

## Open Questions

- Which TRIBE v2 revision should be pinned for reproducible local inference?
- Which ROI label map should back `roi_summary.csv`?
- Should a future fixture use a tiny mocked wrapper in addition to the current
  deterministic fake backend?
- How should image temporalization balance scientific control against natural
  video priors learned by TRIBE v2?
