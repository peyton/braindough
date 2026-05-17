from pathlib import Path

import numpy as np

from braindough.analysis import (
    build_delta_arrays,
    build_derived_tables,
    build_objectives_summary,
)
from braindough.stimuli import Stimulus


def test_optimization_history_is_deterministic_and_budgeted() -> None:
    stimuli = [
        _stimulus(
            "discrete_stimulus_optimizer:candidate_00",
            "discrete_stimulus_optimizer",
            {"candidate_index": 0, "params": {"shape": "circle"}},
        ),
        _stimulus(
            "discrete_stimulus_optimizer:candidate_01",
            "discrete_stimulus_optimizer",
            {"candidate_index": 1, "params": {"shape": "square"}},
        ),
    ]
    responses = {
        stimuli[0].stimulus_id: np.array([[1.0, -1.0, 1.0]], dtype=np.float32),
        stimuli[1].stimulus_id: np.array([[0.5, -0.25, 0.5]], dtype=np.float32),
    }

    tables = build_derived_tables(stimuli, responses)
    history = tables["optimization_history"]
    objectives = build_objectives_summary(history)

    assert [row["candidate_index"] for row in history] == [0, 1]
    assert history[0]["objective_score"] == history[0]["best_score_so_far"]
    assert objectives["n_candidates"] == 2
    assert objectives["best_candidate_id"] == stimuli[0].stimulus_id
    assert objectives["stopping_reason"] == "candidate_budget_exhausted"


def test_optimization_history_marks_prediction_budget_stop() -> None:
    stimuli = [
        _stimulus(
            "discrete_stimulus_optimizer:candidate_00",
            "discrete_stimulus_optimizer",
            {"candidate_index": 0, "params": {"shape": "circle"}},
        ),
        _stimulus(
            "discrete_stimulus_optimizer:candidate_01",
            "discrete_stimulus_optimizer",
            {"candidate_index": 1, "params": {"shape": "square"}},
        ),
    ]
    responses = {
        stimuli[0].stimulus_id: np.array([[1.0, -1.0, 1.0]], dtype=np.float32),
    }

    history = build_derived_tables(stimuli, responses)["optimization_history"]
    objectives = build_objectives_summary(history)

    assert history[0]["completed_candidates"] == 1
    assert history[0]["candidate_budget"] == 2
    assert history[1]["status"] == "skipped_prediction_budget"
    assert objectives["stopping_reason"] == "prediction_budget_reached"


def test_pair_tables_include_incomplete_pairs_and_delta_arrays() -> None:
    parent = _stimulus(
        "counterfactual_editing_workbench:base:baseline",
        "counterfactual_editing_workbench",
        {"pair_id": "base", "role": "counterfactual_source"},
    )
    edit = _stimulus(
        "counterfactual_editing_workbench:base:local_blur",
        "counterfactual_editing_workbench",
        {
            "parent_id": parent.stimulus_id,
            "pair_id": "base",
            "edit_type": "local_blur",
            "changed_pixel_fraction": 0.25,
        },
    )
    missing = _stimulus(
        "counterfactual_editing_workbench:base:object_mask",
        "counterfactual_editing_workbench",
        {
            "parent_id": parent.stimulus_id,
            "pair_id": "base",
            "edit_type": "object_mask",
            "changed_pixel_fraction": 0.5,
        },
    )
    responses = {
        parent.stimulus_id: np.array([[1.0, 2.0]], dtype=np.float32),
        edit.stimulus_id: np.array([[1.5, 1.0]], dtype=np.float32),
    }

    tables = build_derived_tables([parent, edit, missing], responses)
    arrays, index_rows = build_delta_arrays([parent, edit, missing], responses)

    pairs = tables["counterfactual_pairs"]
    assert len(pairs) == 2
    assert pairs[0]["complete_pair"] is True
    assert pairs[0]["normalized_l2_delta"] > 0
    assert pairs[1]["complete_pair"] is False
    assert pairs[1]["missing_reason"] == "missing_child_response"
    assert list(arrays) == ["delta_0000"]
    assert index_rows[0]["stimulus_id"] == edit.stimulus_id


def test_top_delta_vertices_are_lesion_specific() -> None:
    lesion_parent = _stimulus(
        "virtual_lesion_lab:base_00:baseline",
        "virtual_lesion_lab",
        {"role": "lesion_source"},
    )
    lesion_child = _stimulus(
        "virtual_lesion_lab:base_00:low_contrast",
        "virtual_lesion_lab",
        {"parent_id": lesion_parent.stimulus_id, "lesion_type": "low_contrast"},
    )
    counterfactual_parent = _stimulus(
        "counterfactual_editing_workbench:base_00:baseline",
        "counterfactual_editing_workbench",
        {"role": "counterfactual_source"},
    )
    counterfactual_child = _stimulus(
        "counterfactual_editing_workbench:base_00:local_blur",
        "counterfactual_editing_workbench",
        {"parent_id": counterfactual_parent.stimulus_id, "edit_type": "local_blur"},
    )
    responses = {
        lesion_parent.stimulus_id: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        lesion_child.stimulus_id: np.array([[1.0, 4.0, 6.0]], dtype=np.float32),
        counterfactual_parent.stimulus_id: np.array(
            [[3.0, 2.0, 1.0]], dtype=np.float32
        ),
        counterfactual_child.stimulus_id: np.array([[6.0, 4.0, 1.0]], dtype=np.float32),
    }

    rows = build_derived_tables(
        [
            lesion_parent,
            lesion_child,
            counterfactual_parent,
            counterfactual_child,
        ],
        responses,
    )["top_delta_vertices"]

    assert rows
    assert {row["suite"] for row in rows} == {"virtual_lesion_lab"}


def test_candidate_catalog_records_unscored_candidates() -> None:
    stimuli = [
        _stimulus(
            "discrete_stimulus_optimizer:candidate_00",
            "discrete_stimulus_optimizer",
            {"candidate_index": 0, "param_hash": "a", "params": {"shape": "circle"}},
        ),
        _stimulus(
            "discrete_stimulus_optimizer:candidate_01",
            "discrete_stimulus_optimizer",
            {"candidate_index": 1, "param_hash": "b", "params": {"shape": "square"}},
        ),
    ]
    responses = {
        stimuli[0].stimulus_id: np.array([[1.0, -1.0, 1.0]], dtype=np.float32),
    }

    catalog = build_derived_tables(stimuli, responses)["candidate_catalog"]

    assert [row["status"] for row in catalog] == [
        "evaluated",
        "skipped_prediction_budget",
    ]
    assert catalog[0]["param_hash"] == "a"


def test_optimizer_status_distinguishes_backend_errors() -> None:
    stimuli = [
        _stimulus(
            "discrete_stimulus_optimizer:candidate_00",
            "discrete_stimulus_optimizer",
            {"candidate_index": 0, "param_hash": "a", "params": {"shape": "circle"}},
        ),
        _stimulus(
            "discrete_stimulus_optimizer:candidate_01",
            "discrete_stimulus_optimizer",
            {"candidate_index": 1, "param_hash": "b", "params": {"shape": "square"}},
        ),
    ]
    responses = {
        stimuli[0].stimulus_id: np.array([[1.0, -1.0, 1.0]], dtype=np.float32),
    }

    tables = build_derived_tables(
        stimuli,
        responses,
        missing_statuses={stimuli[1].stimulus_id: "backend_error"},
    )
    objectives = build_objectives_summary(tables["optimization_history"])

    assert tables["candidate_catalog"][1]["status"] == "backend_error"
    assert tables["optimization_history"][1]["status"] == "backend_error"
    assert objectives["stopping_reason"] == "backend_error"


def test_latent_components_report_insufficient_samples() -> None:
    stimulus = _stimulus(
        "latent_network_ica_explorer:base_00:static",
        "latent_network_ica_explorer",
        {},
    )
    tables = build_derived_tables(
        [stimulus],
        {stimulus.stimulus_id: np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
    )

    assert tables["latent_components"][0]["status"] == "insufficient_samples"
    assert tables["latent_loadings"] == []


def test_focused_ultrasound_tables_preserve_proxy_limitations() -> None:
    parent = _stimulus(
        "focused_ultrasound_bridge:base:S1:baseline",
        "focused_ultrasound_bridge",
        {
            "pair_id": "base:S1",
            "target_label": "S1",
            "target_network": "somatosensory",
            "protocol_id": "baseline",
            "condition": "baseline",
            "software_dose_index": 0.0,
            "acoustic_modeling_status": "not_modeled",
            "safety_claim": "software_proxy_no_sonication_or_clinical_claim",
            "itrusst_reporting_status": "synthetic_proxy_fields_only",
        },
    )
    active = _stimulus(
        "focused_ultrasound_bridge:base:S1:active_low_duty",
        "focused_ultrasound_bridge",
        {
            "parent_id": parent.stimulus_id,
            "pair_id": "base:S1",
            "target_label": "S1",
            "target_network": "somatosensory",
            "protocol_id": "active_low_duty",
            "condition": "active",
            "software_dose_index": 0.35,
            "sham_mode": "none",
            "acoustic_modeling_status": "not_modeled",
            "safety_claim": "software_proxy_no_sonication_or_clinical_claim",
            "itrusst_reporting_status": "synthetic_proxy_fields_only",
            "nominal_prf_hz": 100,
        },
    )
    responses = {
        parent.stimulus_id: np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        active.stimulus_id: np.array([[1.1, 2.2, 3.6]], dtype=np.float32),
    }

    tables = build_derived_tables([parent, active], responses)

    protocols = tables["focused_ultrasound_protocols"]
    comparisons = tables["focused_ultrasound_comparisons"]
    assert len(protocols) == 2
    assert protocols[1]["itrusst_reporting_status"] == "synthetic_proxy_fields_only"
    assert protocols[1]["acoustic_modeling_status"] == "not_modeled"
    assert comparisons[0]["complete_pair"] is True
    assert comparisons[0]["condition"] == "active"
    assert comparisons[0]["software_dose_index"] == 0.35
    assert comparisons[0]["normalized_l2_delta"] > 0


def _stimulus(stimulus_id: str, suite: str, metadata: dict[str, object]) -> Stimulus:
    return Stimulus(
        stimulus_id=stimulus_id,
        suite=suite,
        modality="video",
        kind="test",
        path=Path("stimuli/test.mp4"),
        sha256="1" * 64,
        license="generated-unlicense",
        metadata=metadata,
    )
