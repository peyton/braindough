from pathlib import Path

import numpy as np

from braindough.analysis import build_derived_tables, build_objectives_summary
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
    assert objectives["stopping_reason"] == "prediction_budget_reached"


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
