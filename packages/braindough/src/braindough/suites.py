"""Known experiment suite registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteDefinition:
    """Static metadata for an experiment suite."""

    name: str
    description: str
    ci_safe: bool = True


SUITE_DEFINITIONS: dict[str, SuiteDefinition] = {
    "image_activation": SuiteDefinition(
        name="image_activation",
        description="Static image clips for baseline image-to-response probes.",
    ),
    "visual_controls": SuiteDefinition(
        name="visual_controls",
        description="Blank, gray, checker, noise, and simple-shape controls.",
    ),
    "visual_perturbations": SuiteDefinition(
        name="visual_perturbations",
        description="Controlled visual perturbations of base images.",
    ),
    "temporalization": SuiteDefinition(
        name="temporalization",
        description="Still-image timing policies such as hold, pan/zoom, and montage.",
    ),
    "audio_controls": SuiteDefinition(
        name="audio_controls",
        description="Generated silence, tone, chirp, and noise controls.",
    ),
    "latent_network_ica_explorer": SuiteDefinition(
        name="latent_network_ica_explorer",
        description="Response-factor probes for PCA/ICA-style component summaries.",
    ),
    "virtual_lesion_lab": SuiteDefinition(
        name="virtual_lesion_lab",
        description="Stimulus-factor lesions for in-silico sensitivity analysis.",
    ),
    "discrete_stimulus_optimizer": SuiteDefinition(
        name="discrete_stimulus_optimizer",
        description="Bounded discrete stimulus search with objective trace artifacts.",
    ),
    "counterfactual_editing_workbench": SuiteDefinition(
        name="counterfactual_editing_workbench",
        description="Paired minimal-edit stimuli for counterfactual response deltas.",
    ),
}


def known_suite_names() -> tuple[str, ...]:
    """Return known suite names in stable order."""

    return tuple(sorted(SUITE_DEFINITIONS))


def validate_suite_names(suites: tuple[str, ...] | list[str]) -> None:
    """Raise for unknown experiment suites."""

    known = set(SUITE_DEFINITIONS)
    unknown = sorted({suite for suite in suites if suite not in known})
    if unknown:
        raise ValueError(
            "Unknown experiment suite(s): "
            + ", ".join(unknown)
            + ". Known suites: "
            + ", ".join(known_suite_names())
        )
