# Deep Research Status: 2026-05-08

At documentation dispatch time, a ChatGPT Deep Research run was still active:

- URL: https://chatgpt.com/c/69fd9a0f-5a78-83e8-bd4f-ba586989517e
- Visible status: "Planning report structure..."
- Visible search count: "403 searches"

Later in the implementation pass, Browser showed the run as completed:

- Visible completion summary: "Research completed in 26m - 11 citations -
  529 searches"
- Visible report title: "Braindough research agenda for local in-silico brain
  experiments"
- Visible executive-direction excerpt: the strongest direction is a compact,
  public experiment system with a stable backend, stable experiment layer, and
  artifact layer rather than only an activation viewer.

The in-app browser could display the report card, but its full report body was
not exposed through the accessible DOM, and downloads are not supported in the
Codex in-app browser. This note therefore records the completed status and the
visible excerpt instead of pretending a full transcript was captured.

## Stable Sources Incorporated Now

The bootstrap docs use stable, directly citable sources that do not depend on
the unfinished Deep Research run:

- TRIBE v2 model card: https://huggingface.co/facebook/tribev2
- Meta TRIBE v2 paper page:
  https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- TRIBE v2 source repository: https://github.com/facebookresearch/tribev2
- Natural Scenes Dataset: https://naturalscenesdataset.org/
- Algonauts 2023:
  https://algonautsproject.com/2023/index.html
- NeuroGen:
  https://www.sciencedirect.com/science/article/pii/S1053811921010831
- Semantic maps from natural speech:
  https://www.nature.com/articles/nature17637
- McGurk effect:
  https://www.nature.com/articles/264746a0
- Posterior superior temporal sulcus and audiovisual processing:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2536697/

## Reconciled Direction

The visible report direction matches the implemented bootstrap:

- CI-safe fake backend.
- Local TRIBE v2 backend behind lazy imports and explicit storage boundaries.
- First suite that includes image activation, visual controls, perturbations,
  temporalization, and audio controls.
- Machine-readable and human-readable artifacts for judging run quality and
  suggesting next experiments.

Open questions remain for a later pass if the full report text is exported by
the user or by a browser surface with download support:

- Whether the report identifies a better TRIBE v2 revision pin or install path.
- Whether it recommends a specific ROI atlas for first-suite summaries.
- Whether it changes the safety boundary for audiovisual mismatch or
  NeuroGen-style optimization.
- Whether it provides clearer NSD or Algonauts adapter requirements.
