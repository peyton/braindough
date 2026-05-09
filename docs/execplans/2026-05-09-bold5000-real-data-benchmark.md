# Add a Real BOLD5000 ROI Encoding Benchmark

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds. This plan follows `~/.agents/PLANS.md`.

## Purpose / Big Picture

Braindough currently proves its artifact and optimization machinery with generated stimuli and fake or TRIBE-predicted responses. After this change, a user can run a real public-data benchmark against BOLD5000 Release 1.0 processed ROI vectors and stimulus-name metadata. The BOLD5000 terms license the fMRI dataset under CC0 except for stimulus images and original annotations, which retain upstream terms. The first benchmark will not claim to beat hidden challenge leaderboards or evaluate the author-recommended Release 2.0 functional data. It will answer a narrower, defensible question: how much ROI response variance can be predicted from stimulus presentation metadata and dataset labels, and which visual ROIs show source/category sensitivity in a bounded local run.

The result is visible by running `just dataset-bold5000-download`, `just run-bold5000-real`, `just report RUN_DIR=...`, and `just executive-summary RUN_DIRS=...`. A completed run writes a normal Braindough artifact under `BRAINDOUGH_HOME/runs/YYYY/MM/...`, includes machine-readable benchmark tables under `outputs/tables/`, and includes an `executive_summary.pdf` that states what was tested and what was not tested.

## Progress

- [x] (2026-05-09T00:19:47Z) Reviewed repo conventions, prior Braindough memory, the current runner/artifact/report surfaces, and the `subagent-driven-development` workflow.
- [x] (2026-05-09T00:19:47Z) Surveyed candidate datasets and literature using the life-science research workflow, web sources, ChatGPT Deep Research setup, and two read-only subagents.
- [x] (2026-05-09T00:19:47Z) Chose BOLD5000 for the immediate automated real-data benchmark because the Release 1.0 fMRI/non-image materials are publicly usable under the dataset terms and downloadable without a Google-form gate, while Algonauts/NSD remains the stronger future leaderboard benchmark.
- [x] (2026-05-09T00:19:47Z) Started downloading the BOLD5000 stimuli metadata zip and ROI response zip into `/Volumes/Virtual Machine HD/Projects/braindough/datasets/bold5000/figshare-v5/downloads`.
- [x] (2026-05-09T00:39:11Z) Implemented explicit BOLD5000 dataset download/doctor commands, storage docs, and real local experiment config.
- [x] (2026-05-09T00:39:11Z) Implemented stimulus metadata ingestion, ROI HDF5 loading, ridge baselines, permutation controls, bootstrap intervals, benchmark sidecar outputs, and path-neutral provenance.
- [x] (2026-05-09T00:39:11Z) Added fixture dataset tests plus report, artifact, and tooling support for BOLD5000 tables and figures.
- [x] (2026-05-09T00:39:11Z) Ran a bounded real BOLD5000 benchmark over CSI1 and six ROIs; the valid run is `/Volumes/Virtual Machine HD/Projects/braindough/runs/2026/05/20260509T003703071357Z-local-bold5000-roi-encoding-bold5000-ridge-795f6ee7`.
- [x] (2026-05-09T00:49:00Z) Ran full local `just check` and `just ci`, generated the BOLD5000-focused executive summary PDF, and addressed reviewer blockers around Release 1.0 naming, provenance, licensing, and exploratory statistics.
- [ ] Open a PR, monitor CI, and merge when green.

## Surprises & Discoveries

- Observation: The official BOLD5000 stimuli zip downloaded from Figshare is only about 4.9 MB and contains presentation lists, labels, and COCO annotation metadata, but not the pixel image files themselves.
  Evidence: Listing `BOLD5000_Stimuli.zip` showed 516 `.txt`, 510 `.mat`, one `.pkl`, and no `.jpg`, `.JPEG`, or `.png` files.

- Observation: BOLD5000 is still a good first real-data target because the Release 1.0 ROI response vectors are a published downloadable package, but Release 2.0 is recommended by the dataset authors for new functional analyses.
  Evidence: The BOLD5000 download page lists Release 1.0 processed ROI brain responses and image names/labels, while the Release 2.0 GitHub README recommends Release 2.0 for all functional-data analyses.

- Observation: URLs triggered the artifact validator's Windows-drive path detector because strings like `https://...` contain the substring `s:/`.
  Evidence: The first completed BOLD5000 run failed validation with public URL fields in metadata and provenance; `_WINDOWS_PATH_FRAGMENT_RE` matched `s://figshare...`. The validator now strips non-file URLs before checking local path fragments.

- Observation: The first real metadata-only benchmark produced small, non-significant validation correlations.
  Evidence: The valid run reported best model `source_family`, best ROI `RHEarlyVis`, mean voxel Pearson r `0.03137362181447747`, mean improvement over mean baseline `0.0098652387649491`, and `n_nominally_significant: 0` across six ROI results.

## Decision Log

- Decision: Use BOLD5000 Release 1.0 ROI vectors for the first automated real-data benchmark rather than Algonauts 2023/NSD or BOLD5000 Release 2.0.
  Rationale: Algonauts 2023 is the cleaner benchmark, but the public training data starts behind a Google form and would make a fully automated repo run brittle. BOLD5000 Release 1.0 has direct Figshare download URLs and exposes ROI response matrices plus stimulus name/label metadata suitable for a bounded run on this machine. Release 2.0 is the recommended future target for functional analyses.
  Date/Author: 2026-05-09 / Codex

- Decision: Make the first benchmark metadata/label-to-ROI encoding, not image-pixel encoding.
  Rationale: The readily downloadable BOLD5000 stimuli package does not contain actual images. A pixel, CLIP, DINO, or TRIBE feature benchmark should be added after a separate image retrieval and provenance layer is implemented for COCO, ImageNet, and SUN stimuli.
  Date/Author: 2026-05-09 / Codex

- Decision: Keep downloads explicit through `just dataset-bold5000-download` and make runtime skip with a clear blocker if data are absent.
  Rationale: Repo rules forbid implicit runtime dependency installation or hidden global state. Dataset acquisition is an explicit user-triggered command that writes only under `BRAINDOUGH_HOME`.
  Date/Author: 2026-05-09 / Codex

## Outcomes & Retrospective

No implementation outcome has been completed yet. This section will be updated after the fake fixture tests pass, after the real BOLD5000 run completes, and after PR CI is known.

2026-05-09T00:39:11Z: The BOLD5000 adapter and benchmark are implemented and the first real run validates. The run is scientifically useful mostly as a negative/control result: simple source-family and filename/label features do not produce strong ROI prediction on the bounded split. This is a better base than a fake result because it gives a reproducible lower bound and clearly motivates the next pixel-feature or Release 2.0 benchmark.

2026-05-09T00:49:00Z: Reviewer feedback identified over-broad release, license, and provenance wording. The implementation now records Release 1.0 file IDs, URLs, terms, observed hashes, Release 2.0 caveats, seed, trial count, split, permutation count, bootstrap count, and exploratory uncorrected p-value context in the artifact and reports.

## Context and Orientation

Braindough is a Python monorepo. Source code lives in `packages/braindough/src/braindough/`, tests live in `tests/`, experiment YAML files live in `experiments/`, and routine commands live in the root `justfile`. Heavy data and run artifacts must live under `BRAINDOUGH_HOME`, which defaults to `/Volumes/Virtual Machine HD/Projects/braindough`.

BOLD5000 is a public fMRI dataset where subjects viewed 5,254 stimulus trials drawn from COCO, ImageNet, and SUN/scene images. This adapter uses Release 1.0 processed ROI matrices such as `CSI1_ROIs_TR1.h5` or `.mat`, where each ROI matrix has rows ordered by the subject stimulus list and columns corresponding to ROI voxels. Release 2.0 provides improved GLM outputs and is recommended by the dataset authors for new functional analyses, but this adapter does not yet evaluate it. A region of interest, or ROI, is a named brain region mask such as early visual cortex, lateral occipital cortex, parahippocampal place area, retrosplenial cortex, or occipital place area. A ridge model is a linear regression model with an L2 penalty that stabilizes estimates when features are correlated or data are noisy.

The existing run flow is: `config.py` loads a YAML spec, `runner.py` creates a run directory, `stimuli.py` generates or stages stimuli, a backend predicts or measures responses, `artifacts.py` writes outputs and validates paths/hashes, and `report.py` builds Markdown/HTML/PDF reports. This feature will add a new BOLD5000 suite and backend while preserving the current fake and TRIBE flows.

## Plan of Work

First, add dataset storage helpers under a new `braindough.datasets.bold5000` module. The helper will know the Figshare file IDs, expected filenames, sizes, and MD5 values for the stimuli metadata zip and ROI response zip. It will provide an explicit download/staging command that writes to `BRAINDOUGH_HOME/datasets/bold5000/figshare-v5`, extracts archives idempotently, and writes a small provenance JSON file. It must not download during normal `braindough run`.

Second, add an experiment suite named `bold5000_roi_encoding` and a backend named `bold5000-ridge`. The suite builder will create lightweight `Stimulus` records from real BOLD5000 presentation-list rows and copy or synthesize tiny label-card PNGs into the run directory as inspectable stand-ins for the absent raw images. Each stimulus record must include the real image filename, subject, trial index, source family inferred from the filename, label tokens, and source archive hashes. The metadata must not contain absolute local paths.

Third, implement the benchmark backend. It will load ROI response matrices from the extracted BOLD5000 ROI package using `h5py`, select configurable subjects and ROIs, standardize responses on the train split, and evaluate multiple small baselines: an intercept/mean baseline, a source-family one-hot model, a token hashing model from image filenames and labels, and an interaction model combining source family with hashed tokens. For each ROI, the backend will fit ridge models with a small grid of alpha values using a train/validation split. It will compute held-out Pearson correlation, R2, permutation p-values from label-shuffled controls, bootstrap confidence intervals over validation trials, and improvement over the mean baseline.

Fourth, add artifact sidecar outputs. The backend will write `outputs/tables/bold5000_trials.csv`, `bold5000_roi_scores.csv`, `bold5000_model_comparison.csv`, `bold5000_permutation_scores.csv`, and `bold5000_feature_weights.csv` when applicable. It will also return compact prediction arrays in `outputs/responses.npz` keyed by subject, ROI, and model so the existing response similarity and activation summaries still work. `artifacts.py` will accept backend sidecar outputs and include them in the manifest, checksums, and validation.

Fifth, update reports and executive summaries. `report.py` will detect `metrics["bold5000_benchmark"]` and add a section describing dataset provenance, subjects, ROIs, split policy, best models, confidence intervals, and caveats. It will generate figures such as ROI score bars and model comparison plots. The PDF must clearly state that this v1 benchmark uses released ROI responses and stimulus metadata/labels, not raw image pixels or hidden challenge tests.

Finally, update docs and tests. Tests will use a tiny generated BOLD5000-like fixture under a temporary directory, not the real dataset. The fixture will contain small HDF5 ROI matrices and presentation lists. Tests will cover staging, feature extraction, ridge scoring, permutation controls, path redaction, artifact validation, report generation, and justfile/tooling exposure. `just ci` remains fake/fixture only; the real download and real benchmark are opt-in local commands.

## Concrete Steps

All commands run from `/Users/peyton/.codex/worktrees/8621/braindough`.

1. Implement the code, docs, tests, and experiment specs described above.

2. Run formatting and tests:

       just fmt
       just check
       just ci

3. Download and stage real data under external storage:

       just dataset-bold5000-download
       just dataset-bold5000-doctor

4. Run the bounded real benchmark:

       just run-bold5000-real

5. Validate and summarize the real run:

       just artifact-validate RUN_DIR=<printed run dir>
       just report RUN_DIR=<printed run dir>
       just executive-summary RUN_DIRS=<printed run dir> OUTPUT_DIR="/Volumes/Virtual Machine HD/Projects/braindough/reports/2026-05-09-bold5000-real-data-benchmark"

6. Open a PR, watch GitHub CI, fix failures, and merge only when green.

## Validation and Acceptance

Local acceptance requires `just check` and `just ci` to pass. The fixture benchmark must produce a valid artifact with no absolute local paths, include benchmark tables in the manifest, and include an executive-summary PDF. The real-data acceptance requires a completed or explicitly skipped `just run-bold5000-real` artifact. If the ROI zip is present and readable, the run must complete on a bounded subset and report held-out ROI metrics. If the external data are absent or corrupt, the artifact must skip with a specific blocker and the doctor command must identify the missing file.

The benchmark is successful if it compares at least two nontrivial metadata models against a mean baseline, produces ROI-level validation metrics and confidence intervals, includes a permutation control, and honestly reports whether any model exceeds the baseline under the bounded split.

## Idempotence and Recovery

Dataset downloads are explicit and idempotent. If an archive already exists and matches its MD5, the downloader will skip it. If an archive exists but has a wrong checksum, it will be left in place with a `.bad` suffix and re-downloaded. Extraction writes marker files so it can be safely re-run. Real benchmark runs write new timestamped run directories and do not modify tracked files.

If a large download is interrupted, rerun `just dataset-bold5000-download`; the command may restart the file download if the server does not support resuming. If the ROI package layout differs from the README, record the observed layout in `Surprises & Discoveries`, adapt the loader, and add fixture tests for the observed shape.

## Artifacts and Notes

Important source facts used for this plan:

    BOLD5000 paper: almost 5,000 real-world images, overlapping SUN, COCO, and ImageNet.
    BOLD5000 website: four subjects, each observing 5,254 images over 15 scanning sessions.
    BOLD5000 Release 1.0 download page: ROI package contains processed ROI voxel matrices in image-by-voxel form.
    BOLD5000 terms: fMRI data are CC0 except stimulus images and original annotations.
    BOLD5000 Release 2.0 README: Release 2.0 is recommended for functional-data analyses.
    Algonauts 2023: NSD/Algonauts remains the future scale-up target with per-vertex Pearson and noise-normalized scoring.

Expected output files for a completed real run include:

    outputs/tables/bold5000_trials.csv
    outputs/tables/bold5000_roi_scores.csv
    outputs/tables/bold5000_model_comparison.csv
    outputs/tables/bold5000_permutation_scores.csv
    outputs/tables/bold5000_feature_weights.csv
    figures/bold5000_roi_scores.png
    figures/bold5000_model_comparison.png
    report.md
    report.html
    executive_summary.pdf

## Interfaces and Dependencies

Add `h5py` as a runtime dependency of `packages/braindough/pyproject.toml` because the BOLD5000 ROI files are HDF5 or MATLAB v7.3-style HDF5 containers. Keep `h5py` imports lazy inside the BOLD5000 dataset/backend code so normal package import remains light.

Define `braindough.datasets.bold5000.BOLD5000Dataset` with methods to resolve paths, download, extract, doctor, load presentation rows, load ROI matrices, and build fixture data. Define `braindough.backends.bold5000_ridge.Bold5000RidgeBackend` implementing the existing backend protocol. Extend `BackendResult` with optional sidecar output rows so backends can include first-class tables in the artifact manifest without bypassing validation.

Revision note: Initial plan created after dataset/literature survey and before source edits to make the work restartable and to document the scope reduction from pixel features to metadata/label encoding.
