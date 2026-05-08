"""Analysis helpers for response artifacts."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


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
