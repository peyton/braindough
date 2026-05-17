from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def test_virtual_lesion_config_controls_types_and_low_contrast(
    tmp_path: Path,
) -> None:
    stimuli = generate_stimuli(
        suites=("virtual_lesion_lab",),
        output_dir=tmp_path,
        seed=123,
        config={
            "virtual_lesion_base_count": 1,
            "virtual_lesion_types": ["low_contrast", "sham_reencode"],
            "lesion_strengths": [1.0],
        },
    )

    baseline = next(stimulus for stimulus in stimuli if stimulus.kind == "baseline")
    low_contrast = next(
        stimulus for stimulus in stimuli if stimulus.kind == "low_contrast"
    )
    sham = next(stimulus for stimulus in stimuli if stimulus.kind == "sham_reencode")
    baseline_image = Image.open(baseline.metadata["source_image"]).convert("RGB")
    low_contrast_image = Image.open(low_contrast.metadata["source_image"]).convert(
        "RGB"
    )

    assert len(stimuli) == 3
    assert low_contrast.metadata["lesion_base_type"] == "low_contrast"
    assert low_contrast.metadata["mask_sha256"]
    assert sham.metadata["strength"] == 0.0
    assert np.asarray(low_contrast_image).std() < np.asarray(baseline_image).std()


def test_optimizer_candidates_have_stable_param_hashes(tmp_path: Path) -> None:
    first = generate_stimuli(
        suites=("discrete_stimulus_optimizer",),
        output_dir=tmp_path / "first",
        seed=123,
        config={"optimizer_candidate_count": 3},
    )
    second = generate_stimuli(
        suites=("discrete_stimulus_optimizer",),
        output_dir=tmp_path / "second",
        seed=123,
        config={"optimizer_candidate_count": 3},
    )

    assert [item.metadata["param_hash"] for item in first] == [
        item.metadata["param_hash"] for item in second
    ]
    assert all(len(item.metadata["param_hash"]) == 64 for item in first)


def test_counterfactual_config_records_edit_magnitude(tmp_path: Path) -> None:
    stimuli = generate_stimuli(
        suites=("counterfactual_editing_workbench",),
        output_dir=tmp_path,
        seed=123,
        config={
            "counterfactual_base_count": 1,
            "counterfactual_edit_types": ["local_blur"],
        },
    )

    edits = [stimulus for stimulus in stimuli if stimulus.metadata.get("parent_id")]

    assert len(edits) == 1
    assert edits[0].metadata["edit_type"] == "local_blur"
    assert edits[0].metadata["changed_pixel_fraction"] > 0
    assert edits[0].metadata["semantic_class"] == "low_level"


def test_focused_ultrasound_bridge_records_protocol_metadata(tmp_path: Path) -> None:
    stimuli = generate_stimuli(
        suites=("focused_ultrasound_bridge",),
        output_dir=tmp_path,
        seed=123,
        config={
            "focused_ultrasound_base_count": 1,
            "focused_ultrasound_targets": ["S1"],
            "focused_ultrasound_protocols": [
                "active_low_duty",
                "sham_transmit_blocked",
            ],
        },
    )

    baseline = next(stimulus for stimulus in stimuli if stimulus.kind == "baseline")
    active = next(
        stimulus for stimulus in stimuli if stimulus.kind == "active_low_duty"
    )
    sham = next(
        stimulus for stimulus in stimuli if stimulus.kind == "sham_transmit_blocked"
    )

    assert len(stimuli) == 3
    assert all(stimulus.path.is_file() for stimulus in stimuli)
    assert active.metadata["parent_id"] == baseline.stimulus_id
    assert active.metadata["intervention_family"] == "focused_ultrasound_protocol_proxy"
    assert active.metadata["target_label"] == "S1"
    assert active.metadata["acoustic_modeling_status"] == "not_modeled"
    assert active.metadata["estimated_in_situ_pressure_mpa"] == ""
    assert active.metadata["nominal_center_frequency_mhz"] == ""
    assert active.metadata["virtual_burst_count"] == 2
    assert active.metadata["software_dose_index"] > 0
    assert sham.metadata["condition"] == "sham"
    assert "coupling-pad" in sham.metadata["sham_mode"]


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
