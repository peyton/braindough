from pathlib import Path

import pytest

from braindough.stimuli import generate_stimuli


def test_generate_first_suite_stimuli(tmp_path: Path) -> None:
    stimuli = generate_stimuli(
        suites=(
            "image_activation",
            "visual_controls",
            "visual_perturbations",
            "temporalization",
            "audio_controls",
        ),
        output_dir=tmp_path,
        seed=123,
    )

    assert len(stimuli) >= 20
    assert {stimulus.modality for stimulus in stimuli} == {"video", "audio"}
    assert all(stimulus.path.is_file() for stimulus in stimuli)
    assert all(len(stimulus.sha256) == 64 for stimulus in stimuli)


def test_generate_perturbation_optimization_stimuli_have_lineage(
    tmp_path: Path,
) -> None:
    stimuli = generate_stimuli(
        suites=(
            "latent_network_ica_explorer",
            "virtual_lesion_lab",
            "discrete_stimulus_optimizer",
            "counterfactual_editing_workbench",
        ),
        output_dir=tmp_path,
        seed=123,
        config={"optimizer_candidate_count": 4},
    )

    by_suite = {stimulus.suite for stimulus in stimuli}
    assert by_suite == {
        "latent_network_ica_explorer",
        "virtual_lesion_lab",
        "discrete_stimulus_optimizer",
        "counterfactual_editing_workbench",
    }
    assert all(stimulus.path.is_file() for stimulus in stimuli)
    assert any(
        stimulus.metadata.get("role") == "component_probe" for stimulus in stimuli
    )
    assert any(
        stimulus.metadata.get("intervention_family") == "stimulus_factor_lesion"
        for stimulus in stimuli
    )
    optimizer_candidates = [
        stimulus
        for stimulus in stimuli
        if stimulus.suite == "discrete_stimulus_optimizer"
    ]
    assert len(optimizer_candidates) == 4
    assert all(
        stimulus.metadata.get("objective")
        == "mean_abs_activation_minus_similarity_penalty"
        for stimulus in optimizer_candidates
    )
    assert any(
        stimulus.metadata.get("pair_id") and stimulus.metadata.get("parent_id")
        for stimulus in stimuli
        if stimulus.suite == "counterfactual_editing_workbench"
    )


def test_generate_stimuli_rejects_unknown_suite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown experiment suite"):
        generate_stimuli(
            suites=("not_a_real_suite",),
            output_dir=tmp_path,
            seed=123,
        )


def test_generate_stimuli_rejects_missing_configured_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Configured stimulus image"):
        generate_stimuli(
            suites=("image_activation",),
            output_dir=tmp_path,
            seed=123,
            config={"images": [str(tmp_path / "missing.png")]},
        )
