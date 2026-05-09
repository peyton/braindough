"""BOLD5000 metadata-to-ROI ridge benchmark backend."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from braindough.backends.base import BackendResult
from braindough.config import ExperimentSpec
from braindough.datasets.bold5000 import (
    BOLD5000_RELEASE,
    BOLD5000_RELEASE_LABEL,
    BOLD5000_VERSION,
    DATASET_WEBSITE,
    DEFAULT_ROIS,
    DEFAULT_SUBJECTS,
    DEFAULT_TR,
    DOWNLOADS,
    FIGSHARE_ARTICLE,
    RELEASE_2_REPO_URL,
    RELEASE_2_URL,
    SOURCE_LICENSE,
    SUITE,
    TERMS_URL,
    BOLD5000Dataset,
    TrialRecord,
    write_csv_rows,
    write_json_rows,
)
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths


class Bold5000RidgeBackend:
    """Fit small ridge baselines from BOLD5000 stimulus metadata to ROI matrices."""

    name = "bold5000-ridge"

    def run(
        self,
        spec: ExperimentSpec,
        stimuli: list[Stimulus],
        paths: BraindoughPaths,
        run_dir: Path,
    ) -> BackendResult:
        del paths
        config = {**spec.stimuli, **spec.backend_config}
        dataset = BOLD5000Dataset(config.get("dataset_root"))
        doctor = dataset.doctor()
        if not doctor.ready:
            return BackendResult(
                status="skipped",
                events=[
                    {
                        "event": "dataset_missing",
                        "backend": self.name,
                        "dataset": "BOLD5000",
                        "blocker": doctor.blocker(),
                    }
                ],
                metrics={
                    "backend": self.name,
                    "dataset": "BOLD5000",
                    "dataset_ready": False,
                    "blocker": doctor.blocker(),
                },
                blocker=doctor.blocker() or "BOLD5000 dataset is not staged",
            )

        subjects = tuple(str(item) for item in config.get("subjects", DEFAULT_SUBJECTS))
        rois = tuple(str(item) for item in config.get("rois", DEFAULT_ROIS))
        tr = str(config.get("tr", DEFAULT_TR))
        trial_limit = int(config.get("trial_limit", 256))
        trial_offset = int(config.get("trial_offset", 0))
        validation_fraction = float(config.get("validation_fraction", 0.25))
        hash_features = int(config.get("hash_features", 64))
        alpha_grid = tuple(
            float(item) for item in config.get("alpha_grid", [0.1, 1.0, 10.0, 100.0])
        )
        permutations = int(config.get("permutations", 16))
        bootstraps = int(config.get("bootstraps", 64))
        rng = np.random.default_rng(spec.seed)

        trial_rows: list[dict[str, object]] = []
        score_rows: list[dict[str, object]] = []
        comparison_rows: list[dict[str, object]] = []
        permutation_rows: list[dict[str, object]] = []
        weight_rows: list[dict[str, object]] = []
        responses: dict[str, np.ndarray] = {}
        events: list[dict[str, object]] = []

        for subject in subjects:
            trials = dataset.load_trials(
                subject, limit=trial_limit, offset=trial_offset
            )
            if len(trials) < 8:
                raise ValueError("BOLD5000 benchmark needs at least 8 trials")
            trial_rows.extend(_trial_rows(trials))
            split = _split_indices(len(trials), validation_fraction, rng)
            feature_sets = _feature_sets(trials, hash_features=hash_features)
            for roi in rois:
                y = dataset.load_roi_matrix(
                    subject, roi, tr=tr, limit=trial_limit + trial_offset
                )[trial_offset : trial_offset + len(trials)]
                roi_result = _score_roi(
                    y,
                    split,
                    feature_sets,
                    alpha_grid=alpha_grid,
                    permutations=permutations,
                    bootstraps=bootstraps,
                    rng=rng,
                )
                best = roi_result["best"]
                response_key = f"{SUITE}:{subject}:{roi}:{best['model']}"
                responses[response_key] = best["prediction"].astype(np.float32)
                events.append(
                    {
                        "event": "benchmark_roi",
                        "backend": self.name,
                        "subject": subject,
                        "roi": roi,
                        "best_model": best["model"],
                        "pearson_mean": float(best["pearson_mean"]),
                        "validation_trials": len(split.validation),
                    }
                )
                score_rows.extend(_score_rows(subject, roi, tr, roi_result["scores"]))
                comparison_rows.append(_comparison_row(subject, roi, tr, roi_result))
                permutation_rows.extend(
                    _permutation_rows(subject, roi, tr, roi_result["permutations"])
                )
                weight_rows.extend(_weight_rows(subject, roi, tr, best, max_rows=10))

        outputs = _write_benchmark_outputs(
            run_dir,
            trial_rows=trial_rows,
            score_rows=score_rows,
            comparison_rows=comparison_rows,
            permutation_rows=permutation_rows,
            weight_rows=weight_rows,
            provenance={
                "dataset": "BOLD5000",
                "dataset_release": BOLD5000_RELEASE,
                "dataset_release_label": BOLD5000_RELEASE_LABEL,
                "storage_version": BOLD5000_VERSION,
                "license": SOURCE_LICENSE,
                "article": FIGSHARE_ARTICLE,
                "website": DATASET_WEBSITE,
                "terms": TERMS_URL,
                "release_2_recommended_url": RELEASE_2_URL,
                "release_2_code_url": RELEASE_2_REPO_URL,
                "subjects": list(subjects),
                "rois": list(rois),
                "tr": tr,
                "trial_limit": trial_limit,
                "trial_offset": trial_offset,
                "validation_fraction": validation_fraction,
                "hash_features": hash_features,
                "alpha_grid": list(alpha_grid),
                "permutations": permutations,
                "bootstraps": bootstraps,
                "seed": spec.seed,
                "dataset_ready": doctor.ready,
                "downloads": _download_provenance(doctor.downloads),
                "source_caveat": (
                    "This adapter uses BOLD5000 Release 1.0 processed ROI "
                    "vectors and stimulus name/label metadata. The dataset "
                    "authors recommend Release 2.0 for new functional analyses. "
                    "The Release 1.0 stimuli metadata archive staged here does "
                    "not include raw pixel images."
                ),
                "p_value_note": (
                    "Permutation p-values are exploratory and uncorrected for "
                    "multiple ROI/model comparisons."
                ),
            },
        )
        metrics = {
            "backend": self.name,
            "dataset": "BOLD5000",
            "dataset_release": BOLD5000_RELEASE,
            "dataset_release_label": BOLD5000_RELEASE_LABEL,
            "storage_version": BOLD5000_VERSION,
            "license": SOURCE_LICENSE,
            "terms": TERMS_URL,
            "article": FIGSHARE_ARTICLE,
            "website": DATASET_WEBSITE,
            "release_2_recommended_url": RELEASE_2_URL,
            "dataset_ready": True,
            "n_stimuli": len(stimuli),
            "subjects": list(subjects),
            "rois": list(rois),
            "tr": tr,
            "trial_limit": trial_limit,
            "trial_offset": trial_offset,
            "validation_fraction": validation_fraction,
            "hash_features": hash_features,
            "alpha_grid": list(alpha_grid),
            "permutations": permutations,
            "bootstraps": bootstraps,
            "seed": spec.seed,
            "p_value_note": (
                "Permutation p-values are exploratory and uncorrected for "
                "multiple ROI/model comparisons."
            ),
            "source_caveat": (
                "This adapter uses BOLD5000 Release 1.0 processed ROI vectors "
                "and stimulus name/label metadata, not Release 2.0 GLM outputs "
                "or raw pixel-image features."
            ),
            "bold5000_benchmark": _benchmark_summary(comparison_rows),
        }
        return BackendResult(
            status="completed",
            responses=responses,
            events=events,
            metrics=metrics,
            outputs=outputs,
        )


def _trial_rows(trials: list[TrialRecord]) -> list[dict[str, object]]:
    return [
        {
            "subject": trial.subject,
            "trial_index": trial.trial_index,
            "image_filename": trial.filename,
            "normalized_image_filename": trial.normalized_filename,
            "source_family": trial.source_family,
            "label": trial.label,
            "tokens": " ".join(trial.tokens),
            "repeated": trial.repeated,
        }
        for trial in trials
    ]


def _download_provenance(
    observed: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for name, expected in DOWNLOADS.items():
        actual = observed.get(name, {})
        rows[name] = {
            "file_id": expected["file_id"],
            "url": expected["url"],
            "expected_md5": expected["md5"],
            "observed_md5": actual.get("md5"),
            "expected_size": expected["size"],
            "observed_size": actual.get("size"),
        }
    return rows


def _split_indices(
    n_trials: int, validation_fraction: float, rng: np.random.Generator
) -> Any:
    validation_count = max(2, min(n_trials - 2, round(n_trials * validation_fraction)))
    indices = np.arange(n_trials)
    rng.shuffle(indices)
    validation = np.sort(indices[:validation_count])
    train = np.sort(indices[validation_count:])
    return _Split(train=train, validation=validation)


class _Split:
    def __init__(self, train: np.ndarray, validation: np.ndarray) -> None:
        self.train = train
        self.validation = validation


def _feature_sets(
    trials: list[TrialRecord], *, hash_features: int
) -> dict[str, tuple[np.ndarray, list[str]]]:
    sources = sorted({trial.source_family for trial in trials})
    source_matrix = np.zeros((len(trials), len(sources) + 1), dtype=np.float32)
    for row, trial in enumerate(trials):
        source_matrix[row, sources.index(trial.source_family)] = 1.0
        source_matrix[row, -1] = 1.0 if trial.repeated else 0.0
    source_names = [f"source:{source}" for source in sources] + ["repeated"]

    hash_matrix = np.zeros((len(trials), hash_features), dtype=np.float32)
    for row, trial in enumerate(trials):
        for token in trial.tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            column = int.from_bytes(digest[:4], "little") % hash_features
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            hash_matrix[row, column] += sign
    hash_names = [f"token_hash:{index:02d}" for index in range(hash_features)]
    return {
        "source_family": (source_matrix, source_names),
        "token_hash": (hash_matrix, hash_names),
        "source_plus_token_hash": (
            np.concatenate([source_matrix, hash_matrix], axis=1),
            [*source_names, *hash_names],
        ),
    }


def _score_roi(
    y: np.ndarray,
    split: _Split,
    feature_sets: dict[str, tuple[np.ndarray, list[str]]],
    *,
    alpha_grid: tuple[float, ...],
    permutations: int,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    y = np.nan_to_num(y.astype(np.float32), copy=False)
    y_train = y[split.train]
    y_val = y[split.validation]
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_scale = y_train.std(axis=0, keepdims=True)
    y_scale[y_scale == 0] = 1.0
    y_train_z = (y_train - y_mean) / y_scale
    y_val_z = (y_val - y_mean) / y_scale
    baseline_prediction = np.zeros_like(y_val_z)
    baseline = _score_prediction(y_val_z, baseline_prediction)
    scores = [
        {
            "model": "mean_baseline",
            "alpha": 0.0,
            "feature_count": 1,
            "prediction": baseline_prediction,
            **baseline,
            "bootstrap_low": 0.0,
            "bootstrap_high": 0.0,
            "permutation_p": 1.0,
            "feature_names": ["intercept"],
            "weights": np.zeros((1, y.shape[1]), dtype=np.float32),
        }
    ]
    permutation_records: list[dict[str, object]] = []
    for model_name, (x, feature_names) in feature_sets.items():
        x_train, x_val, x_mean, x_scale = _standardize_x(x, split)
        best_score: dict[str, Any] | None = None
        for alpha in alpha_grid:
            weights = _ridge_fit(x_train, y_train_z, alpha)
            prediction = _with_intercept(x_val, x_mean, x_scale) @ weights
            scored = _score_prediction(y_val_z, prediction)
            row = {
                "model": model_name,
                "alpha": alpha,
                "feature_count": len(feature_names),
                "prediction": prediction,
                "weights": weights,
                "feature_names": ["intercept", *feature_names],
                **scored,
            }
            if best_score is None or row["pearson_mean"] > best_score["pearson_mean"]:
                best_score = row
        assert best_score is not None
        bootstrap = _bootstrap_ci(y_val_z, best_score["prediction"], bootstraps, rng)
        permuted = _permutation_scores(
            x,
            y_train_z,
            y_val_z,
            split,
            float(best_score["alpha"]),
            int(permutations),
            rng,
        )
        p_value = (
            sum(value >= best_score["pearson_mean"] for value in permuted) + 1
        ) / (len(permuted) + 1)
        best_score["bootstrap_low"] = bootstrap[0]
        best_score["bootstrap_high"] = bootstrap[1]
        best_score["permutation_p"] = p_value
        scores.append(best_score)
        permutation_records.extend(
            {
                "model": model_name,
                "permutation_index": index,
                "pearson_mean": value,
            }
            for index, value in enumerate(permuted)
        )
    best = max(
        scores[1:] if len(scores) > 1 else scores, key=lambda item: item["pearson_mean"]
    )
    return {
        "best": best,
        "baseline": baseline,
        "scores": scores,
        "permutations": permutation_records,
    }


def _standardize_x(
    x: np.ndarray, split: _Split
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = x[split.train].astype(np.float32)
    val = x[split.validation].astype(np.float32)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return (train - mean) / scale, (val - mean) / scale, mean, scale


def _with_intercept(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    del mean, scale
    intercept = np.ones((x.shape[0], 1), dtype=np.float32)
    return np.concatenate([intercept, x], axis=1)


def _ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    design = np.concatenate(
        [np.ones((x_train.shape[0], 1), dtype=np.float32), x_train], axis=1
    )
    penalty = np.eye(design.shape[1], dtype=np.float32) * np.float32(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def _score_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    correlations = [
        _correlation(y_true[:, column], y_pred[:, column])
        for column in range(y_true.shape[1])
    ]
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "pearson_mean": float(np.mean(correlations)) if correlations else 0.0,
        "pearson_median": float(np.median(correlations)) if correlations else 0.0,
        "r2": 1.0 - residual / total if total > 0 else 0.0,
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_centered = left.astype(np.float64) - float(np.mean(left))
    right_centered = right.astype(np.float64) - float(np.mean(right))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if bootstraps <= 0 or y_true.shape[0] < 2:
        value = _score_prediction(y_true, y_pred)["pearson_mean"]
        return value, value
    values: list[float] = []
    for _ in range(bootstraps):
        rows = rng.integers(0, y_true.shape[0], y_true.shape[0])
        values.append(_score_prediction(y_true[rows], y_pred[rows])["pearson_mean"])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _permutation_scores(
    x: np.ndarray,
    y_train_z: np.ndarray,
    y_val_z: np.ndarray,
    split: _Split,
    alpha: float,
    permutations: int,
    rng: np.random.Generator,
) -> list[float]:
    scores: list[float] = []
    for _ in range(permutations):
        shuffled = np.array(split.train)
        rng.shuffle(shuffled)
        x_perm = np.array(x)
        x_perm[split.train] = x_perm[shuffled]
        x_train, x_val, _, _ = _standardize_x(x_perm, split)
        weights = _ridge_fit(x_train, y_train_z, alpha)
        prediction = (
            _with_intercept(x_val, np.empty((1, 0)), np.empty((1, 0))) @ weights
        )
        scores.append(_score_prediction(y_val_z, prediction)["pearson_mean"])
    return scores


def _score_rows(
    subject: str, roi: str, tr: str, scores: list[dict[str, Any]]
) -> list[dict[str, object]]:
    return [
        {
            "subject": subject,
            "roi": roi,
            "tr": tr,
            "model": score["model"],
            "alpha": score["alpha"],
            "feature_count": score["feature_count"],
            "pearson_mean": score["pearson_mean"],
            "pearson_median": score["pearson_median"],
            "r2": score["r2"],
            "bootstrap_low": score.get("bootstrap_low", ""),
            "bootstrap_high": score.get("bootstrap_high", ""),
            "permutation_p": score.get("permutation_p", ""),
        }
        for score in scores
    ]


def _comparison_row(
    subject: str, roi: str, tr: str, result: dict[str, Any]
) -> dict[str, object]:
    best = result["best"]
    baseline = result["scores"][0]
    return {
        "subject": subject,
        "roi": roi,
        "tr": tr,
        "best_model": best["model"],
        "best_alpha": best["alpha"],
        "best_pearson_mean": best["pearson_mean"],
        "best_r2": best["r2"],
        "mean_baseline_pearson": baseline["pearson_mean"],
        "improvement_over_mean": best["pearson_mean"] - baseline["pearson_mean"],
        "bootstrap_low": best.get("bootstrap_low", ""),
        "bootstrap_high": best.get("bootstrap_high", ""),
        "permutation_p": best.get("permutation_p", ""),
    }


def _permutation_rows(
    subject: str, roi: str, tr: str, rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    return [{**row, "subject": subject, "roi": roi, "tr": tr} for row in rows]


def _weight_rows(
    subject: str, roi: str, tr: str, best: dict[str, Any], *, max_rows: int
) -> list[dict[str, object]]:
    weights = np.asarray(best["weights"], dtype=np.float32)
    names = list(best["feature_names"])
    magnitudes = np.mean(np.abs(weights), axis=1)
    order = np.argsort(magnitudes)[::-1][:max_rows]
    return [
        {
            "subject": subject,
            "roi": roi,
            "tr": tr,
            "model": best["model"],
            "feature": names[int(index)],
            "mean_abs_weight": float(magnitudes[int(index)]),
        }
        for index in order
    ]


def _write_benchmark_outputs(
    run_dir: Path,
    *,
    trial_rows: list[dict[str, object]],
    score_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    permutation_rows: list[dict[str, object]],
    weight_rows: list[dict[str, object]],
    provenance: dict[str, object],
) -> list[dict[str, object]]:
    tables_dir = run_dir / "outputs" / "tables"
    return [
        write_csv_rows(tables_dir / "bold5000_trials.csv", trial_rows),
        write_csv_rows(tables_dir / "bold5000_roi_scores.csv", score_rows),
        write_csv_rows(tables_dir / "bold5000_model_comparison.csv", comparison_rows),
        write_csv_rows(
            tables_dir / "bold5000_permutation_scores.csv", permutation_rows
        ),
        write_csv_rows(tables_dir / "bold5000_feature_weights.csv", weight_rows),
        write_json_rows(tables_dir / "bold5000_provenance.json", provenance),
    ]


def _benchmark_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {
            "status": "no_scores",
            "best_roi": None,
            "best_model": None,
            "best_pearson_mean": None,
        }
    best = max(rows, key=lambda row: _finite_float(row.get("best_pearson_mean")))
    mean_improvement = float(
        np.mean([_finite_float(row.get("improvement_over_mean")) for row in rows])
    )
    return {
        "status": "completed",
        "n_roi_results": len(rows),
        "best_subject": best["subject"],
        "best_roi": best["roi"],
        "best_model": best["best_model"],
        "best_pearson_mean": best["best_pearson_mean"],
        "best_r2": best["best_r2"],
        "mean_improvement_over_mean": mean_improvement,
        "n_nominally_significant": sum(
            1
            for row in rows
            if _finite_float(row.get("permutation_p", math.inf)) <= 0.05
        ),
    }


def _finite_float(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.inf
    return number if math.isfinite(number) else math.inf
