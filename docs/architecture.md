# Braindough Architecture

Braindough is a repo-local research workbench for running small, reproducible
stimulus probes through TRIBE v2 and saving enough provenance to compare neural
response predictions across experiments. The bootstrap should stay lean: one
Python package, one local storage contract, one artifact contract, and `just`
recipes that work from a clean checkout.

## Repository Shape

The implementation keeps source code under `packages/braindough/src/braindough/`,
tests under `tests/`, reusable scripts under `scripts/`, and documentation under
`docs/`. Local data, model caches, generated media, and run outputs must stay
outside version control.

Expected paths:

- `packages/braindough/src/braindough/cli.py`: command-line entry point.
- `packages/braindough/src/braindough/backends/tribe_v2.py`: lazy TRIBE v2
  backend.
- `packages/braindough/src/braindough/backends/fake.py`: deterministic CI
  backend.
- `packages/braindough/src/braindough/stimuli.py`: image, video, and audio
  generation helpers.
- `packages/braindough/src/braindough/storage.py`: external path and
  content-hash helpers.
- `packages/braindough/src/braindough/artifacts.py`: artifact writer and schema
  checks.
- `experiments/smoke/`: CI-safe fake-backend YAML specs.
- `experiments/local/`: opt-in local TRIBE v2 YAML specs.
- `$BRAINDOUGH_HOME/runs/`: local-only run directories produced by experiments.

## Runtime Flow

Every experiment follows the same pipeline:

1. Read a checked-in config or explicit CLI flags.
2. Resolve input files from repo-relative paths.
3. Create a run directory under `$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>/`.
4. Prepare TRIBE v2 events from the input stimulus.
5. Run prediction through the TRIBE v2 wrapper.
6. Write predictions, summaries, figures, and a manifest.
7. Print the manifest path and the key metric summary.

The architecture deliberately separates stimulus preparation from model
inference. Static image experiments use `temporalization` helpers to make short,
silent video clips because TRIBE v2 is documented for video, audio, and text
inputs rather than standalone image tensors.

## First Experiment Suite

The bootstrap suite consists of:

- `image_activation`: predict responses for one static image after conversion
  to a short silent video.
- `visual_controls`: compare a target image against visual controls such as
  blank, color-matched, blurred, or scrambled versions.
- `visual_perturbations`: apply controlled visual changes such as crop, blur,
  saturation, object masking, and frame jitter.
- `temporalization`: compare still-image-to-video policies such as hold, slow
  pan, zoom, loop, and cut timing.
- `audio_controls`: compare silence, tone, noise, speech, and audio-only or
  video-only baselines.

Each suite should be callable through the same command surface:

    just run-fake
    just run-tribe

## Repo Commands

The implementation should expose these repo-local commands:

    just bootstrap
    just fmt
    just fmt-check
    just lint
    just typecheck
    just test
    just check
    just ci
    just run-fake
    just run-tribe
    just artifact-validate RUN_DIR=<run-dir>
    just report RUN_DIR=<run-dir>

`just bootstrap` uses `mise` and `uv` with declared dependencies only. Runtime
code must not install dependencies implicitly, write heavy outputs outside
`BRAINDOUGH_HOME`, or depend on undeclared global mutable state.

## Boundaries

Braindough predicts model outputs; it does not measure human participants,
diagnose people, infer private mental states, or make clinical claims. Any
future adapter for NSD, Algonauts, or other human-neuroscience datasets must be
added as an explicit data-ingestion module with documented license and access
rules before it is used by experiments.

## Source Anchors

- TRIBE v2 model card: https://huggingface.co/facebook/tribev2
- Meta TRIBE v2 paper page: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- TRIBE v2 source repository: https://github.com/facebookresearch/tribev2
- NSD: https://naturalscenesdataset.org/
- Algonauts 2023: https://algonautsproject.com/2023/index.html
