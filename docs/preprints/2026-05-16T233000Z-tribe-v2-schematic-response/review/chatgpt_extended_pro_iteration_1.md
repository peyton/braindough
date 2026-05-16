# ChatGPT Extended Pro Critique Iteration 1

Status: completed after stopping an oversized full-manuscript request and
resubmitting a focused framing prompt.

Prompt summary:

- Review a planned methods-note preprint for one local Braindough TRIBE v2 run.
- Facts supplied: 39 generated stimuli, `max_predictions=1`, one completed
  model-predicted response, shape `2 x 20484`, mean absolute predicted response
  `0.102670`, vertex-index split into `10242` and `10242` values, no fsaverage
  coordinates or ROI atlas.
- Requested highest-risk framing problems before release.

Key critique:

- Remove or downgrade "registration" throughout; use "vertex-index schematic
  visualization" or "QA schematic" instead.
- Make title and abstract unmistakably engineering-only.
- Do not frame the run as capable of scientific inference.
- Say "TRIBE v2-predicted average-subject fMRI-like response" wherever wording
  could imply measured neural activity.
- Treat the 39 generated stimuli as context, not analyzed data.
- Clarify that left/right split is an index convention unless independently
  verified from a TRIBE output specification.
- Do not interpret left/right means as hemispheric symmetry or biological
  plausibility.
- Replace "plausible scale and distribution" with weaker artifact-summary
  language.
- Explain what the two response segments mean.
- Add exact provenance: commit, config hash, checkpoint/package information,
  seed, environment, response checksum, and validation criteria.
- Separate artifact validation from scientific/model validation.
- Tighten citation support for TRIBE output shape, average-subject framing,
  hemodynamic shift, mesh convention, and license.

Applied changes:

- Retitled manuscript to avoid "registration."
- Renamed manuscript directory from `registered-response` to
  `schematic-response`.
- Renamed committed metadata copy to `vertex_index_schematic_metadata.json`.
- Added stronger artifact-vs-model-validity wording.
- Added unscored-stimulus context wording.
- Added index-convention caveats and removed biological symmetry language.
- Added run commit, config hash, seed, and response archive checksum.
