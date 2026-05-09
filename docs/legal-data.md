# Legal and Data Boundaries

Braindough is a research tool for model-predicted neural responses. It is not a
clinical device, a human-subjects collection system, or a private mental-state
decoder.

## Licensing Baseline

TRIBE v2 is published under `cc-by-nc-4.0` on Hugging Face. Braindough should
treat TRIBE v2 inference and derivative artifacts as non-commercial unless a
separate legal review approves a different use.

Stable source:

- https://huggingface.co/facebook/tribev2

External datasets have their own terms. NSD and Algonauts are important
research anchors, but adapters must not land until their access, citation,
storage, and redistribution rules are documented in the adapter itself.

BOLD5000 is staged as a public real-data benchmark. The BOLD5000 terms license
the fMRI dataset under CC0 except for stimulus images and their original
annotations, which retain upstream terms. The repository may store adapter code,
fixture data, hashes, and derived metrics, but the Release 1.0 processed ROI
matrices and stimulus metadata archives still live under `BRAINDOUGH_HOME`.
The first BOLD5000 benchmark uses ROI response matrices and filenames/labels; it
does not redistribute raw stimulus images or original annotations. BOLD5000
Release 2.0 is author-recommended for new functional analyses and is not
evaluated by the current adapter.

Stable sources:

- BOLD5000: https://bold5000-dataset.github.io/website/
- BOLD5000 terms: https://bold5000-dataset.github.io/website/terms.html
- BOLD5000 Figshare: https://figshare.com/articles/dataset/BOLD5000/6459449
- BOLD5000 Release 2.0 code: https://github.com/BOLD5000-dataset/BOLD5000
- NSD: https://naturalscenesdataset.org/
- Algonauts 2023: https://algonautsproject.com/2023/index.html

## Data Classes

- Public code and docs: safe to commit.
- Example configs: safe to commit only when they contain no secrets and use
  small redistributable fixture paths.
- User stimuli: local only unless the user confirms redistribution rights.
- Model weights and caches: local only unless the model license and hosting
  channel permit redistribution.
- Human neuroscience datasets: local only, access-controlled, and adapter-gated.
- Deep Research browser output: private working material unless the user
  explicitly provides text for inclusion.

## Privacy Rules

Do not persist credentials, browser session content, private research chats, or
unredacted local absolute paths in committed files. Run manifests may include
absolute paths locally for reproducibility, but shareable exports must redact
user home paths and private source locations.

Do not store raw human participant data in Git. If a future NSD or Algonauts
adapter needs subject-level data, keep it under `BRAINDOUGH_HOME` or another
explicit external path and write only hashes, citations, and derived metrics to
run manifests.

## Claims Policy

Allowed language:

- "TRIBE v2 predicted a response pattern for this stimulus."
- "This control changed the model-predicted activation summary."
- "This run is an in-silico probe."

Avoid:

- "This image activates your brain."
- "This result diagnoses cognition or attention."
- "This predicts what a specific person thinks."

## Review Before Sharing

Before sharing any artifact, run:

    just artifact-validate RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"
    just report RUN_DIR="$BRAINDOUGH_HOME/runs/YYYY/MM/<run_id>"

Then check the report and manifest manually for private stimuli, raw data,
local paths, and license-sensitive files.
