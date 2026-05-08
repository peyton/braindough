"""Analysis helpers for response artifacts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from braindough.stimuli import Stimulus


def response_metrics(responses: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute small, stable metrics for human and machine reports."""

    if not responses:
        return {
            "n_responses": 0,
            "mean_abs_activation": None,
            "pairwise_correlation_mean": None,
            "pairwise_correlation_min": None,
            "pairwise_correlation_max": None,
        }

    flattened = {
        key: value.reshape(-1).astype(np.float64) for key, value in responses.items()
    }
    mean_abs = {key: float(np.mean(np.abs(value))) for key, value in flattened.items()}
    correlations: list[float] = []
    for left, right in combinations(flattened, 2):
        correlations.append(_correlation(flattened[left], flattened[right]))

    metrics: dict[str, Any] = {
        "n_responses": len(responses),
        "mean_abs_activation": float(np.mean(list(mean_abs.values()))),
        "mean_abs_activation_by_stimulus": mean_abs,
        "pairwise_correlation_mean": float(np.mean(correlations))
        if correlations
        else None,
        "pairwise_correlation_min": float(np.min(correlations))
        if correlations
        else None,
        "pairwise_correlation_max": float(np.max(correlations))
        if correlations
        else None,
    }
    metrics["top_activation_stimuli"] = [
        {"stimulus_id": key, "mean_abs_activation": value}
        for key, value in sorted(
            mean_abs.items(), key=lambda item: item[1], reverse=True
        )[:5]
    ]
    return metrics


def derived_metrics(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Compute experiment-family metrics used by reports and agents."""

    tables = build_derived_tables(stimuli, responses)
    metrics: dict[str, Any] = {
        "suite_summary": _suite_summary(stimuli, responses),
        "n_perturbation_comparisons": len(tables["perturbation_comparisons"]),
        "n_optimizer_candidates": len(tables["optimization_history"]),
    }
    objectives = build_objectives_summary(tables["optimization_history"])
    if objectives:
        metrics["optimization"] = objectives
    latent_status = "insufficient_samples"
    if tables["latent_components"] and "component_id" in tables["latent_components"][0]:
        latent_status = "computed"
    metrics["latent_components"] = {
        "status": latent_status,
        "n_components": len(tables["latent_components"])
        if latent_status == "computed"
        else 0,
    }
    return metrics


def build_derived_tables(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> dict[str, list[dict[str, Any]]]:
    """Return derived table rows for artifact sidecar outputs."""

    return {
        "stimuli": _stimulus_rows(stimuli, responses),
        "response_metrics": _response_rows(stimuli, responses),
        "perturbation_comparisons": _perturbation_rows(stimuli, responses),
        "optimization_history": _optimization_rows(stimuli, responses),
        **_latent_component_rows(stimuli, responses),
    }


def build_objectives_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact optimization objective summary."""

    if not history:
        return {}
    ranked = sorted(
        history,
        key=lambda row: (
            float(row["objective_score"]),
            -int(row["candidate_index"]),
            str(row["stimulus_id"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        "objective": "mean_abs_activation_minus_similarity_penalty",
        "n_candidates": len(history),
        "best_candidate_id": best["stimulus_id"],
        "best_score": float(best["objective_score"]),
        "best_mean_abs_activation": float(best["mean_abs_activation"]),
        "stopping_reason": str(
            best.get("stopping_reason", "candidate_budget_exhausted")
        ),
    }


def similarity_matrix(responses: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return stimulus ids and a Pearson-correlation similarity matrix."""

    ids = sorted(responses)
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.eye(len(ids), dtype=np.float32)
    flattened = {key: responses[key].reshape(-1).astype(np.float64) for key in ids}
    for i, left in enumerate(ids):
        for j, right in enumerate(ids[i + 1 :], start=i + 1):
            value = _correlation(flattened[left], flattened[right])
            matrix[i, j] = value
            matrix[j, i] = value
    return ids, matrix


def response_summary(responses: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Return scalar summaries per response."""

    summaries: dict[str, dict[str, Any]] = {}
    for key, value in responses.items():
        flat = value.reshape(-1).astype(np.float64)
        summaries[key] = {
            "mean_abs_activation": float(np.mean(np.abs(flat))),
            "mean_activation": float(np.mean(flat)),
            "std_activation": float(np.std(flat)),
            "shape": list(value.shape),
        }
    return summaries


def next_experiment_suggestions(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Generate lightweight follow-up suggestions from the run metrics."""

    if metrics.get("n_responses", 0) == 0:
        return [
            {
                "id": "resolve_backend_blocker",
                "title": "Resolve backend blocker and rerun first suite",
                "rationale": "No response arrays were produced, so the next step is runtime repair.",
            }
        ]

    suggestions = [
        {
            "id": "roi_contrast_pack",
            "title": "Add ROI-level contrasts once fsaverage labels are available",
            "rationale": "The current artifact stores whole-surface predictions; ROI summaries will make visual and auditory contrasts easier to inspect.",
        },
        {
            "id": "cross_modal_alignment",
            "title": "Compare image clips with caption/audio descriptions",
            "rationale": "TRIBE v2 is multimodal, so matching and mismatching stimuli can probe model-level semantic alignment.",
        },
        {
            "id": "stimulus_search",
            "title": "Optimize simple generated stimuli for response fingerprints",
            "rationale": "The deterministic artifact format can support closed-loop search over stimulus parameters.",
        },
    ]
    if metrics.get("pairwise_correlation_mean") is not None:
        suggestions.append(
            {
                "id": "fingerprint_browser",
                "title": "Build a cortical-fingerprint browser",
                "rationale": "Pairwise response similarities are already available and can drive clustering or nearest-neighbor views.",
            }
        )
    return suggestions


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size != right.size:
        size = min(left.size, right.size)
        left = left[:size]
        right = right[:size]
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _suite_summary(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    suites = sorted({stimulus.suite for stimulus in stimuli})
    return {
        suite: {
            "stimuli": sum(1 for stimulus in stimuli if stimulus.suite == suite),
            "responses": sum(
                1
                for stimulus in stimuli
                if stimulus.suite == suite and stimulus.stimulus_id in responses
            ),
        }
        for suite in suites
    }


def _stimulus_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "suite": stimulus.suite,
                "modality": stimulus.modality,
                "kind": stimulus.kind,
                "sha256": stimulus.sha256,
                "response_present": stimulus.stimulus_id in responses,
                "base_id": stimulus.metadata.get("base_id", ""),
                "parent_id": stimulus.metadata.get("parent_id", ""),
                "pair_id": stimulus.metadata.get("pair_id", ""),
                "role": stimulus.metadata.get("role", ""),
            }
        )
    return rows


def _response_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    summaries = response_summary(responses)
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        summary = summaries.get(stimulus.stimulus_id)
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "suite": stimulus.suite,
                "kind": stimulus.kind,
                "response_present": summary is not None,
                "mean_abs_activation": _optional_float(summary, "mean_abs_activation"),
                "mean_activation": _optional_float(summary, "mean_activation"),
                "std_activation": _optional_float(summary, "std_activation"),
                "response_shape": "x".join(map(str, summary["shape"]))
                if summary
                else "",
            }
        )
    return rows


def _perturbation_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    by_id = {stimulus.stimulus_id: stimulus for stimulus in stimuli}
    summaries = response_summary(responses)
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if not parent_id or stimulus.stimulus_id not in responses:
            continue
        parent = by_id.get(parent_id)
        if parent_id not in responses:
            continue
        current = summaries[stimulus.stimulus_id]
        baseline = summaries[parent_id]
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "parent_id": parent_id,
                "suite": stimulus.suite,
                "kind": stimulus.kind,
                "base_id": stimulus.metadata.get("base_id", ""),
                "pair_id": stimulus.metadata.get("pair_id", ""),
                "edit_type": stimulus.metadata.get("edit_type", ""),
                "lesion_type": stimulus.metadata.get("lesion_type", ""),
                "comparison_family": stimulus.metadata.get(
                    "intervention_family",
                    stimulus.metadata.get("role", "perturbation"),
                ),
                "parent_kind": parent.kind if parent else "",
                "mean_abs_activation": current["mean_abs_activation"],
                "parent_mean_abs_activation": baseline["mean_abs_activation"],
                "mean_abs_delta": float(
                    current["mean_abs_activation"] - baseline["mean_abs_activation"]
                ),
                "response_correlation": _correlation(
                    responses[parent_id].reshape(-1).astype(np.float64),
                    responses[stimulus.stimulus_id].reshape(-1).astype(np.float64),
                ),
            }
        )
    return rows


def _optimization_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    all_candidates = [
        stimulus
        for stimulus in stimuli
        if stimulus.suite == "discrete_stimulus_optimizer"
    ]
    candidates = [
        stimulus for stimulus in all_candidates if stimulus.stimulus_id in responses
    ]
    candidate_budget = len(all_candidates)
    stopping_reason = (
        "candidate_budget_exhausted"
        if len(candidates) == candidate_budget
        else "prediction_budget_reached"
    )
    summaries = response_summary(responses)
    flattened = {
        stimulus.stimulus_id: responses[stimulus.stimulus_id]
        .reshape(-1)
        .astype(np.float64)
        for stimulus in candidates
    }
    rows: list[dict[str, Any]] = []
    best_so_far = float("-inf")
    for index, stimulus in enumerate(
        sorted(
            candidates, key=lambda item: int(item.metadata.get("candidate_index", 0))
        )
    ):
        prior_ids = [row["stimulus_id"] for row in rows]
        penalty = 0.0
        if prior_ids:
            penalty = max(
                abs(_correlation(flattened[prior], flattened[stimulus.stimulus_id]))
                for prior in prior_ids
            )
        mean_abs = float(summaries[stimulus.stimulus_id]["mean_abs_activation"])
        score = mean_abs - 0.05 * penalty
        best_so_far = max(best_so_far, score)
        params = stimulus.metadata.get("params", {})
        rows.append(
            {
                "step": index,
                "candidate_index": int(stimulus.metadata.get("candidate_index", index)),
                "stimulus_id": stimulus.stimulus_id,
                "mean_abs_activation": mean_abs,
                "similarity_penalty": float(penalty),
                "objective_score": float(score),
                "best_score_so_far": float(best_so_far),
                "candidate_budget": candidate_budget,
                "completed_candidates": len(candidates),
                "stopping_reason": stopping_reason,
                "params": params,
            }
        )
    return rows


def _latent_component_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> dict[str, list[dict[str, Any]]]:
    ids = [
        stimulus.stimulus_id
        for stimulus in stimuli
        if stimulus.suite == "latent_network_ica_explorer"
        and stimulus.stimulus_id in responses
    ]
    ids = sorted(ids)
    if len(ids) < 2:
        return {
            "latent_components": [
                {
                    "status": "insufficient_samples",
                    "n_responses": len(ids),
                    "message": "At least two responses are required for PCA-style components.",
                }
            ],
            "latent_loadings": [],
        }
    matrix = np.vstack([responses[stimulus_id].reshape(-1) for stimulus_id in ids])
    matrix = matrix.astype(np.float64)
    matrix -= matrix.mean(axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    component_count = min(3, len(singular_values))
    total = float(np.sum(singular_values**2)) or 1.0
    component_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    for component_idx in range(component_count):
        component = vt[component_idx]
        component_id = f"component_{component_idx + 1}"
        component_rows.append(
            {
                "component_id": component_id,
                "method": "pca_svd",
                "singular_value": float(singular_values[component_idx]),
                "variance_ratio": float((singular_values[component_idx] ** 2) / total),
                "mean_abs_weight": float(np.mean(np.abs(component))),
            }
        )
        for stimulus_id in ids:
            loading = float(
                np.dot(
                    matrix[ids.index(stimulus_id)],
                    component,
                )
            )
            loading_rows.append(
                {
                    "component_id": component_id,
                    "stimulus_id": stimulus_id,
                    "loading": loading,
                }
            )
    return {"latent_components": component_rows, "latent_loadings": loading_rows}


def _optional_float(summary: dict[str, Any] | None, key: str) -> float | str:
    if summary is None:
        return ""
    return float(summary[key])
