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
    stimuli: list[Stimulus],
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute experiment-family metrics used by reports and agents."""

    tables = build_derived_tables(stimuli, responses, missing_statuses=missing_statuses)
    metrics: dict[str, Any] = {
        "suite_summary": _suite_summary(stimuli, responses),
        "n_perturbation_comparisons": len(tables["perturbation_comparisons"]),
        "n_optimizer_candidates": len(tables["optimization_history"]),
        "n_lesion_comparisons": len(tables["lesion_comparisons"]),
        "n_counterfactual_pairs": len(tables["counterfactual_pairs"]),
        "n_focused_ultrasound_comparisons": len(
            tables["focused_ultrasound_comparisons"]
        ),
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
    stimuli: list[Stimulus],
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return derived table rows for artifact sidecar outputs."""

    missing_statuses = missing_statuses or {}
    return {
        "stimuli": _stimulus_rows(stimuli, responses),
        "response_metrics": _response_rows(stimuli, responses),
        "perturbation_comparisons": _perturbation_rows(stimuli, responses),
        "lesion_manifest": _lesion_manifest_rows(stimuli),
        "lesion_comparisons": _lesion_comparison_rows(stimuli, responses),
        "lesion_roi_summary": _lesion_roi_rows(stimuli, responses),
        "top_delta_vertices": _top_delta_vertex_rows(stimuli, responses),
        "candidate_catalog": _candidate_catalog_rows(
            stimuli, responses, missing_statuses
        ),
        "optimization_history": _optimization_rows(
            stimuli, responses, missing_statuses
        ),
        "counterfactual_edits": _counterfactual_edit_rows(stimuli),
        "counterfactual_pairs": _counterfactual_pair_rows(stimuli, responses),
        "focused_ultrasound_protocols": _focused_ultrasound_protocol_rows(stimuli),
        "focused_ultrasound_comparisons": _focused_ultrasound_comparison_rows(
            stimuli, responses
        ),
        **_latent_component_rows(stimuli, responses),
    }


def build_delta_arrays(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Return parent-child response delta arrays and their index rows."""

    arrays: dict[str, np.ndarray] = {}
    index_rows: list[dict[str, Any]] = []
    sequence = 0
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if (
            not parent_id
            or parent_id not in responses
            or stimulus.stimulus_id not in responses
        ):
            continue
        key = f"delta_{sequence:04d}"
        delta = _aligned_delta(responses[parent_id], responses[stimulus.stimulus_id])
        arrays[key] = delta.astype(np.float32)
        index_rows.append(
            {
                "array_key": key,
                "stimulus_id": stimulus.stimulus_id,
                "parent_id": parent_id,
                "suite": stimulus.suite,
                "shape": list(delta.shape),
                "dtype": "float32",
            }
        )
        sequence += 1
    return arrays, index_rows


def build_objectives_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact optimization objective summary."""

    if not history:
        return {}
    evaluated = [row for row in history if row.get("status") == "evaluated"]
    candidate_budget = int(history[0].get("candidate_budget", len(history)))
    statuses = [str(row.get("status", "")) for row in history]
    base = {
        "objective": "mean_abs_activation_minus_similarity_penalty",
        "objective_version": "discrete_optimizer.v2",
        "score_direction": "maximize",
        "activation_metric": "mean_abs_activation",
        "similarity_metric": "max_abs_pearson_to_prior",
        "penalty_weight": 0.05,
        "candidate_budget": candidate_budget,
        "evaluated_candidates": len(evaluated),
        "stopping_reason": _history_stopping_reason(statuses),
    }
    if not evaluated:
        return {
            **base,
            "n_candidates": 0,
            "best_candidate_id": None,
            "best_score": None,
            "best_mean_abs_activation": None,
            "best_params": {},
            "top_candidates": [],
        }
    ranked = sorted(
        evaluated,
        key=lambda row: (
            float(row["objective_score"]),
            -int(row["candidate_index"]),
            str(row["stimulus_id"]),
        ),
        reverse=True,
    )
    best = ranked[0]
    return {
        **base,
        "n_candidates": len(evaluated),
        "best_candidate_id": best["stimulus_id"],
        "best_score": float(best["objective_score"]),
        "best_mean_abs_activation": float(best["mean_abs_activation"]),
        "best_params": best.get("params", {}),
        "top_candidates": [
            {
                "stimulus_id": row["stimulus_id"],
                "objective_score": float(row["objective_score"]),
                "mean_abs_activation": float(row["mean_abs_activation"]),
                "params": row.get("params", {}),
            }
            for row in ranked[:5]
        ],
    }


def _history_stopping_reason(statuses: list[str]) -> str:
    if statuses and all(status == "evaluated" for status in statuses):
        return "candidate_budget_exhausted"
    if any(status == "backend_error" for status in statuses):
        return "backend_error"
    return "prediction_budget_reached"


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
    if left.size < 2 or right.size < 2:
        return 0.0
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _suite_summary(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> dict[str, dict[str, int]]:
    suites = sorted({stimulus.suite for stimulus in stimuli})
    return {
        suite: {
            "stimuli": sum(1 for stimulus in stimuli if stimulus.suite == suite),
            "responses": sum(
                1
                for response_id in responses
                if response_id.startswith(f"{suite}:")
                or any(
                    stimulus.suite == suite and stimulus.stimulus_id == response_id
                    for stimulus in stimuli
                )
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
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if not parent_id:
            continue
        rows.append(_comparison_row(stimulus, parent_id, by_id, responses))
    return rows


def _lesion_manifest_rows(stimuli: list[Stimulus]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        if stimulus.suite != "virtual_lesion_lab" or not stimulus.metadata.get(
            "parent_id"
        ):
            continue
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "parent_id": stimulus.metadata.get("parent_id", ""),
                "base_id": stimulus.metadata.get("base_id", ""),
                "lesion_type": stimulus.metadata.get("lesion_type", ""),
                "lesion_base_type": stimulus.metadata.get("lesion_base_type", ""),
                "strength": stimulus.metadata.get("strength", ""),
                "fill_rgb": stimulus.metadata.get("fill_rgb", ""),
                "masked_fraction": stimulus.metadata.get("masked_fraction", ""),
                "bbox": stimulus.metadata.get("bbox", ""),
                "mask_sha256": stimulus.metadata.get("mask_sha256", ""),
                "source_image_sha256": stimulus.metadata.get("source_image_sha256", ""),
                "sha256": stimulus.sha256,
            }
        )
    return rows


def _lesion_comparison_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    by_id = {stimulus.stimulus_id: stimulus for stimulus in stimuli}
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if stimulus.suite != "virtual_lesion_lab" or not parent_id:
            continue
        row = _comparison_row(stimulus, parent_id, by_id, responses)
        row.update(
            {
                "lesion_base_type": stimulus.metadata.get("lesion_base_type", ""),
                "strength": stimulus.metadata.get("strength", ""),
                "masked_fraction": stimulus.metadata.get("masked_fraction", ""),
            }
        )
        rows.append(row)
    return rows


def _counterfactual_edit_rows(stimuli: list[Stimulus]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        if (
            stimulus.suite != "counterfactual_editing_workbench"
            or not stimulus.metadata.get("parent_id")
        ):
            continue
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "parent_id": stimulus.metadata.get("parent_id", ""),
                "pair_id": stimulus.metadata.get("pair_id", ""),
                "edit_type": stimulus.metadata.get("edit_type", ""),
                "edit_base_type": stimulus.metadata.get("edit_base_type", ""),
                "edit_version": stimulus.metadata.get("edit_version", ""),
                "semantic_class": stimulus.metadata.get("semantic_class", ""),
                "changed_pixel_fraction": stimulus.metadata.get(
                    "changed_pixel_fraction", ""
                ),
                "mean_rgb_l1": stimulus.metadata.get("mean_rgb_l1", ""),
                "mean_rgb_l2": stimulus.metadata.get("mean_rgb_l2", ""),
                "edge_change_score": stimulus.metadata.get("edge_change_score", ""),
                "edit_bbox": stimulus.metadata.get("edit_bbox", ""),
                "source_image_sha256": stimulus.metadata.get("source_image_sha256", ""),
                "sha256": stimulus.sha256,
            }
        )
    return rows


def _counterfactual_pair_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    by_id = {stimulus.stimulus_id: stimulus for stimulus in stimuli}
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if stimulus.suite != "counterfactual_editing_workbench" or not parent_id:
            continue
        row = _comparison_row(stimulus, parent_id, by_id, responses)
        row.update(
            {
                "semantic_class": stimulus.metadata.get("semantic_class", ""),
                "changed_pixel_fraction": stimulus.metadata.get(
                    "changed_pixel_fraction", ""
                ),
                "mean_rgb_l1": stimulus.metadata.get("mean_rgb_l1", ""),
                "mean_rgb_l2": stimulus.metadata.get("mean_rgb_l2", ""),
                "edge_change_score": stimulus.metadata.get("edge_change_score", ""),
                "minimality_ratio": _minimality_ratio(row, stimulus),
            }
        )
        rows.append(row)
    return rows


def _focused_ultrasound_protocol_rows(stimuli: list[Stimulus]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        if stimulus.suite != "focused_ultrasound_bridge":
            continue
        rows.append(
            {
                "stimulus_id": stimulus.stimulus_id,
                "parent_id": stimulus.metadata.get("parent_id", ""),
                "base_id": stimulus.metadata.get("base_id", ""),
                "pair_id": stimulus.metadata.get("pair_id", ""),
                "target_id": stimulus.metadata.get("target_id", ""),
                "target_label": stimulus.metadata.get("target_label", ""),
                "target_network": stimulus.metadata.get("target_network", ""),
                "task_family": stimulus.metadata.get("task_family", ""),
                "target_coordinate_space": stimulus.metadata.get(
                    "target_coordinate_space", ""
                ),
                "target_coordinate": stimulus.metadata.get("target_coordinate", ""),
                "protocol_id": stimulus.metadata.get("protocol_id", ""),
                "condition": stimulus.metadata.get("condition", ""),
                "software_dose_index": stimulus.metadata.get("software_dose_index", ""),
                "virtual_duty_cycle_bins": stimulus.metadata.get(
                    "virtual_duty_cycle_bins", ""
                ),
                "virtual_burst_count": stimulus.metadata.get("virtual_burst_count", ""),
                "virtual_envelope": stimulus.metadata.get("virtual_envelope", ""),
                "sham_mode": stimulus.metadata.get("sham_mode", ""),
                "acoustic_modeling_status": stimulus.metadata.get(
                    "acoustic_modeling_status", ""
                ),
                "safety_claim": stimulus.metadata.get("safety_claim", ""),
                "itrusst_reporting_status": stimulus.metadata.get(
                    "itrusst_reporting_status", ""
                ),
                "nominal_center_frequency_mhz": stimulus.metadata.get(
                    "nominal_center_frequency_mhz", ""
                ),
                "nominal_prf_hz": stimulus.metadata.get("nominal_prf_hz", ""),
                "nominal_duty_cycle": stimulus.metadata.get("nominal_duty_cycle", ""),
                "nominal_sonication_seconds": stimulus.metadata.get(
                    "nominal_sonication_seconds", ""
                ),
                "estimated_in_situ_pressure_mpa": stimulus.metadata.get(
                    "estimated_in_situ_pressure_mpa", ""
                ),
                "estimated_in_situ_ispta_mw_cm2": stimulus.metadata.get(
                    "estimated_in_situ_ispta_mw_cm2", ""
                ),
                "mechanical_index": stimulus.metadata.get("mechanical_index", ""),
                "thermal_index": stimulus.metadata.get("thermal_index", ""),
                "transducer_model": stimulus.metadata.get("transducer_model", ""),
                "drive_system": stimulus.metadata.get("drive_system", ""),
                "source_image_sha256": stimulus.metadata.get("source_image_sha256", ""),
                "sha256": stimulus.sha256,
            }
        )
    return rows


def _focused_ultrasound_comparison_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    by_id = {stimulus.stimulus_id: stimulus for stimulus in stimuli}
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if stimulus.suite != "focused_ultrasound_bridge" or not parent_id:
            continue
        row = _comparison_row(stimulus, parent_id, by_id, responses)
        row.update(
            {
                "target_id": stimulus.metadata.get("target_id", ""),
                "target_label": stimulus.metadata.get("target_label", ""),
                "target_network": stimulus.metadata.get("target_network", ""),
                "protocol_id": stimulus.metadata.get("protocol_id", ""),
                "condition": stimulus.metadata.get("condition", ""),
                "software_dose_index": stimulus.metadata.get("software_dose_index", ""),
                "virtual_duty_cycle_bins": stimulus.metadata.get(
                    "virtual_duty_cycle_bins", ""
                ),
                "virtual_burst_count": stimulus.metadata.get("virtual_burst_count", ""),
                "sham_mode": stimulus.metadata.get("sham_mode", ""),
                "acoustic_modeling_status": stimulus.metadata.get(
                    "acoustic_modeling_status", ""
                ),
                "safety_claim": stimulus.metadata.get("safety_claim", ""),
            }
        )
        rows.append(row)
    return rows


def _comparison_row(
    stimulus: Stimulus,
    parent_id: str,
    by_id: dict[str, Stimulus],
    responses: dict[str, np.ndarray],
) -> dict[str, Any]:
    summaries = response_summary(responses)
    parent = by_id.get(parent_id)
    complete = parent_id in responses and stimulus.stimulus_id in responses
    row: dict[str, Any] = {
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
        "complete_pair": complete,
        "missing_reason": ""
        if complete
        else _missing_pair_reason(stimulus, parent_id, responses),
    }
    if not complete:
        row.update(
            {
                "mean_abs_activation": "",
                "parent_mean_abs_activation": "",
                "mean_abs_delta": "",
                "response_correlation": "",
                "l2_delta": "",
                "normalized_l2_delta": "",
                "signed_mean_delta": "",
                "std_delta": "",
                "peak_abs_delta": "",
                "sign_flip_fraction": "",
            }
        )
        return row

    current = summaries[stimulus.stimulus_id]
    baseline = summaries[parent_id]
    delta_summary = _delta_summary(
        responses[parent_id], responses[stimulus.stimulus_id]
    )
    row.update(
        {
            "mean_abs_activation": current["mean_abs_activation"],
            "parent_mean_abs_activation": baseline["mean_abs_activation"],
            "mean_abs_delta": float(
                current["mean_abs_activation"] - baseline["mean_abs_activation"]
            ),
            "response_correlation": _correlation(
                responses[parent_id].reshape(-1).astype(np.float64),
                responses[stimulus.stimulus_id].reshape(-1).astype(np.float64),
            ),
            **delta_summary,
        }
    )
    return row


def _missing_pair_reason(
    stimulus: Stimulus, parent_id: str, responses: dict[str, np.ndarray]
) -> str:
    missing: list[str] = []
    if parent_id not in responses:
        missing.append("missing_parent_response")
    if stimulus.stimulus_id not in responses:
        missing.append("missing_child_response")
    return ",".join(missing)


def _delta_summary(parent: np.ndarray, child: np.ndarray) -> dict[str, float]:
    delta = _aligned_delta(parent, child).reshape(-1).astype(np.float64)
    parent_flat = parent.reshape(-1).astype(np.float64)[: delta.size]
    child_flat = child.reshape(-1).astype(np.float64)[: delta.size]
    l2 = float(np.linalg.norm(delta))
    parent_norm = float(np.linalg.norm(parent_flat))
    sign_flips = np.sign(parent_flat) != np.sign(child_flat)
    active = (parent_flat != 0) | (child_flat != 0)
    return {
        "l2_delta": l2,
        "normalized_l2_delta": l2 / max(parent_norm, 1e-12),
        "signed_mean_delta": float(np.mean(delta)),
        "std_delta": float(np.std(delta)),
        "peak_abs_delta": float(np.max(np.abs(delta))),
        "sign_flip_fraction": float(
            np.count_nonzero(sign_flips & active) / max(np.count_nonzero(active), 1)
        ),
    }


def _aligned_delta(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    left = parent.astype(np.float32)
    right = child.astype(np.float32)
    if left.shape == right.shape:
        return right - left
    size = min(left.size, right.size)
    return right.reshape(-1)[:size] - left.reshape(-1)[:size]


def _minimality_ratio(row: dict[str, Any], stimulus: Stimulus) -> float | str:
    response_delta = row.get("normalized_l2_delta")
    changed = stimulus.metadata.get("changed_pixel_fraction")
    if not isinstance(response_delta, float):
        return ""
    if not isinstance(changed, int | float) or float(changed) <= 0:
        return ""
    return float(response_delta / float(changed))


def _lesion_roi_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if (
            stimulus.suite != "virtual_lesion_lab"
            or parent_id not in responses
            or stimulus.stimulus_id not in responses
        ):
            continue
        delta = _as_time_vertex_delta(
            responses[parent_id], responses[stimulus.stimulus_id]
        )
        vertex_count = delta.shape[1]
        for bin_index, indices in enumerate(np.array_split(np.arange(vertex_count), 4)):
            if indices.size == 0:
                continue
            chunk = delta[:, indices]
            rows.append(
                {
                    "stimulus_id": stimulus.stimulus_id,
                    "parent_id": parent_id,
                    "lesion_type": stimulus.metadata.get("lesion_type", ""),
                    "vertex_bin": f"bin_{bin_index + 1}",
                    "vertex_start": int(indices[0]),
                    "vertex_end": int(indices[-1]),
                    "mean_abs_delta": float(np.mean(np.abs(chunk))),
                    "max_abs_delta": float(np.max(np.abs(chunk))),
                }
            )
    return rows


def _top_delta_vertex_rows(
    stimuli: list[Stimulus], responses: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in stimuli:
        parent_id = str(stimulus.metadata.get("parent_id", ""))
        if (
            stimulus.suite != "virtual_lesion_lab"
            or parent_id not in responses
            or stimulus.stimulus_id not in responses
        ):
            continue
        delta = _as_time_vertex_delta(
            responses[parent_id], responses[stimulus.stimulus_id]
        )
        vertex_delta = np.mean(np.abs(delta), axis=0)
        for rank, vertex_index in enumerate(
            np.argsort(vertex_delta)[::-1][:5], start=1
        ):
            rows.append(
                {
                    "stimulus_id": stimulus.stimulus_id,
                    "parent_id": parent_id,
                    "suite": stimulus.suite,
                    "rank": rank,
                    "vertex_index": int(vertex_index),
                    "mean_abs_delta": float(vertex_delta[vertex_index]),
                }
            )
    return rows


def _as_time_vertex_delta(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    delta = _aligned_delta(parent, child)
    if delta.ndim == 1:
        return delta.reshape(1, -1)
    if delta.ndim >= 2:
        return delta.reshape(delta.shape[0], -1)
    return delta.reshape(1, 1)


def _candidate_catalog_rows(
    stimuli: list[Stimulus],
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stimulus in sorted(
        (item for item in stimuli if item.suite == "discrete_stimulus_optimizer"),
        key=lambda item: int(item.metadata.get("candidate_index", 0)),
    ):
        rows.append(
            {
                "candidate_index": int(stimulus.metadata.get("candidate_index", 0)),
                "stimulus_id": stimulus.stimulus_id,
                "status": _response_status(
                    stimulus.stimulus_id, responses, missing_statuses
                ),
                "sha256": stimulus.sha256,
                "param_hash": stimulus.metadata.get("param_hash", ""),
                "params": stimulus.metadata.get("params", {}),
                "generation_policy": stimulus.metadata.get("generation_policy", ""),
                "response_present": stimulus.stimulus_id in responses,
            }
        )
    return rows


def _optimization_rows(
    stimuli: list[Stimulus],
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str],
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
    stopping_reason = _optimizer_stopping_reason(
        all_candidates, responses, missing_statuses
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
    evaluated_ids: list[str] = []
    for index, stimulus in enumerate(
        sorted(
            all_candidates,
            key=lambda item: int(item.metadata.get("candidate_index", 0)),
        )
    ):
        params = stimulus.metadata.get("params", {})
        base_row: dict[str, Any] = {
            "step": index,
            "candidate_index": int(stimulus.metadata.get("candidate_index", index)),
            "stimulus_id": stimulus.stimulus_id,
            "candidate_budget": candidate_budget,
            "completed_candidates": len(candidates),
            "stopping_reason": stopping_reason,
            "params": params,
            "param_hash": stimulus.metadata.get("param_hash", ""),
            "status": _response_status(
                stimulus.stimulus_id, responses, missing_statuses
            ),
        }
        if stimulus.stimulus_id not in responses:
            rows.append(
                {
                    **base_row,
                    "mean_abs_activation": "",
                    "similarity_penalty": "",
                    "objective_score": "",
                    "best_score_so_far": best_so_far
                    if best_so_far != float("-inf")
                    else "",
                }
            )
            continue
        prior_ids = evaluated_ids
        penalty = 0.0
        if prior_ids:
            penalty = max(
                abs(_correlation(flattened[prior], flattened[stimulus.stimulus_id]))
                for prior in prior_ids
            )
        mean_abs = float(summaries[stimulus.stimulus_id]["mean_abs_activation"])
        score = mean_abs - 0.05 * penalty
        best_so_far = max(best_so_far, score)
        evaluated_ids.append(stimulus.stimulus_id)
        rows.append(
            {
                **base_row,
                "mean_abs_activation": mean_abs,
                "similarity_penalty": float(penalty),
                "objective_score": float(score),
                "best_score_so_far": float(best_so_far),
            }
        )
    return rows


def _response_status(
    stimulus_id: str,
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str],
) -> str:
    if stimulus_id in responses:
        return "evaluated"
    return missing_statuses.get(stimulus_id, "skipped_prediction_budget")


def _optimizer_stopping_reason(
    candidates: list[Stimulus],
    responses: dict[str, np.ndarray],
    missing_statuses: dict[str, str],
) -> str:
    statuses = [
        _response_status(stimulus.stimulus_id, responses, missing_statuses)
        for stimulus in candidates
    ]
    if statuses and all(status == "evaluated" for status in statuses):
        return "candidate_budget_exhausted"
    if any(status == "backend_error" for status in statuses):
        return "backend_error"
    return "prediction_budget_reached"


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
