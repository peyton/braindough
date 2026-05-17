"""Executive summary generation for Braindough experiment runs."""

from __future__ import annotations

import csv
import json
import textwrap
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from braindough.storage import BraindoughPaths

SUMMARY_SCHEMA_VERSION = "braindough.executive_summary.v1"
TARGET_RUNS = {
    ("fake", "smoke/fake-perturbation-optimization"),
    ("tribe-v2", "local/tribe-v2-perturbation-optimization"),
    ("bold5000-ridge", "local/bold5000-roi-encoding"),
}


@dataclass(frozen=True)
class RunBundle:
    """Loaded summary data for one run."""

    run_id: str
    backend: str
    experiment_id: str
    status: str
    blocker: str | None
    created_at: str | None
    completed_at: str | None
    n_stimuli: int
    n_responses: int
    metrics: dict[str, Any]
    tables: dict[str, Any]

    @property
    def evidence_label(self) -> str:
        if self.backend == "fake":
            return "software-validation"
        if self.backend == "tribe-v2":
            return "bounded-model-prediction"
        if self.backend == "bold5000-ridge":
            return "measured-public-data"
        return "artifact-supported"

    def to_payload(self) -> dict[str, Any]:
        """Return path-neutral JSON payload for this run."""

        return {
            "run_id": self.run_id,
            "backend": self.backend,
            "experiment_id": self.experiment_id,
            "status": self.status,
            "blocker": self.blocker,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "n_stimuli": self.n_stimuli,
            "n_responses": self.n_responses,
            "evidence_label": self.evidence_label,
            "suite_summary": self.metrics.get("suite_summary", {}),
            "mean_abs_activation": self.metrics.get("mean_abs_activation"),
            "top_activation_stimuli": self.metrics.get("top_activation_stimuli", []),
            "optimization": self.metrics.get("optimization", {}),
            "bold5000_benchmark": self.metrics.get("bold5000_benchmark", {}),
            "bold5000_context": {
                "dataset_release": self.metrics.get("dataset_release"),
                "dataset_release_label": self.metrics.get("dataset_release_label"),
                "subjects": self.metrics.get("subjects", []),
                "rois": self.metrics.get("rois", []),
                "tr": self.metrics.get("tr"),
                "trial_limit": self.metrics.get("trial_limit"),
                "trial_offset": self.metrics.get("trial_offset"),
                "validation_fraction": self.metrics.get("validation_fraction"),
                "seed": self.metrics.get("seed"),
                "permutations": self.metrics.get("permutations"),
                "bootstraps": self.metrics.get("bootstraps"),
                "p_value_note": self.metrics.get("p_value_note"),
                "source_caveat": self.metrics.get("source_caveat"),
            },
            "latent_components": self.metrics.get("latent_components", {}),
            "n_optimizer_candidates": self.metrics.get("n_optimizer_candidates", 0),
            "n_perturbation_comparisons": self.metrics.get(
                "n_perturbation_comparisons", 0
            ),
            "response_shape": {
                "timesteps": self.metrics.get("response_timesteps"),
                "vertices": self.metrics.get("response_vertices"),
            },
            "prediction_budget": {
                "attempted_predictions": self.metrics.get("attempted_predictions"),
                "max_predictions": self.metrics.get("max_predictions"),
                "max_predictions_per_suite": self.metrics.get(
                    "max_predictions_per_suite"
                ),
            },
        }


def discover_latest_run_dirs(home: str | Path | None = None) -> list[Path]:
    """Discover latest fake, TRIBE, and BOLD5000 summary target runs."""

    paths = BraindoughPaths.discover(home=home)
    if not paths.runs_root.is_dir():
        return []

    latest: dict[tuple[str, str], tuple[str, Path]] = {}
    for manifest_path in paths.runs_root.rglob("manifest.json"):
        try:
            manifest = _read_json(manifest_path)
        except (json.JSONDecodeError, OSError):
            continue
        backend = str(manifest.get("backend", {}).get("name", ""))
        experiment_id = str(manifest.get("config", {}).get("experiment_id", ""))
        key = (backend, experiment_id)
        if key not in TARGET_RUNS:
            continue
        sort_key = str(
            manifest.get("completed_at")
            or manifest.get("created_at")
            or manifest.get("run_id")
            or manifest_path.parent.name
        )
        existing = latest.get(key)
        if existing is None or sort_key > existing[0]:
            latest[key] = (sort_key, manifest_path.parent)

    ordered_keys = [
        ("fake", "smoke/fake-perturbation-optimization"),
        ("tribe-v2", "local/tribe-v2-perturbation-optimization"),
        ("bold5000-ridge", "local/bold5000-roi-encoding"),
    ]
    return [latest[key][1] for key in ordered_keys if key in latest]


def write_executive_summary(
    run_dirs: Sequence[str | Path] | None = None,
    output_dir: str | Path | None = None,
    home: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write executive summary PDF, Markdown, JSON, sources, and figures."""

    selected_run_dirs = [Path(path) for path in run_dirs or []]
    if not selected_run_dirs:
        selected_run_dirs = discover_latest_run_dirs(home=home)
    if not selected_run_dirs:
        raise ValueError(
            "no executive-summary target runs found; run just run-fake-optimization, "
            "just run-bold5000-real, or pass --run-dir"
        )

    paths = BraindoughPaths.discover(home=home)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = (
        Path(output_dir)
        if output_dir is not None
        else paths.home
        / "reports"
        / f"{datetime.now(UTC):%Y}"
        / f"{datetime.now(UTC):%m}"
        / f"{timestamp}-executive-summary"
    )
    destination.mkdir(parents=True, exist_ok=True)
    figures_dir = destination / "figures"
    figures_dir.mkdir(exist_ok=True)

    bundles = [_load_run_bundle(path) for path in selected_run_dirs]
    sources = _literature_sources()
    sources.extend(
        _research_sources(_resolve_repo_root(repo_root) / "docs" / "research")
    )
    charts = _write_figures(figures_dir, bundles)
    payload = _build_payload(bundles, charts, sources)

    json_path = destination / "executive-summary.json"
    sources_path = destination / "sources.json"
    md_path = destination / "executive-summary.md"
    pdf_path = destination / "executive-summary.pdf"

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    sources_path.write_text(
        json.dumps({"sources": sources}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_summary(payload), encoding="utf-8")
    _write_pdf(pdf_path, payload, charts)

    return {
        "summary_json": json_path,
        "sources_json": sources_path,
        "markdown": md_path,
        "pdf": pdf_path,
        "output_dir": destination,
    }


def _load_run_bundle(run_dir: Path) -> RunBundle:
    manifest = _read_json(run_dir / "manifest.json")
    metrics = _read_json(run_dir / "metrics.json")
    config = manifest.get("config", {})
    backend = manifest.get("backend", {})
    return RunBundle(
        run_id=str(manifest.get("run_id", run_dir.name)),
        backend=str(backend.get("name", metrics.get("backend", "unknown"))),
        experiment_id=str(config.get("experiment_id", "unknown")),
        status=str(manifest.get("status", "unknown")),
        blocker=manifest.get("blocker"),
        created_at=manifest.get("created_at"),
        completed_at=manifest.get("completed_at"),
        n_stimuli=int(metrics.get("n_stimuli", len(manifest.get("inputs", [])))),
        n_responses=int(metrics.get("n_responses", 0)),
        metrics=metrics,
        tables=_read_tables(run_dir),
    )


def _build_payload(
    bundles: Sequence[RunBundle],
    charts: Sequence[dict[str, Any]],
    sources: Sequence[dict[str, str]],
) -> dict[str, Any]:
    runs = [bundle.to_payload() for bundle in bundles]
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "audience": "research_leads",
        "scope": _summary_scope(bundles),
        "disclaimer": (
            "TRIBE values in this report are model-predicted responses. "
            "BOLD5000 values, when present, are Release 1.0 measured public "
            "ROI responses evaluated through bounded local benchmark code. "
            "BOLD5000 Release 2.0 is recommended by the dataset authors for "
            "new functional analyses and is not evaluated here."
        ),
        "runs": runs,
        "key_findings": _key_findings(bundles),
        "limitations": _limitations(),
        "future_directions": _future_directions(),
        "charts": list(charts),
        "sources": list(sources),
    }


def _summary_scope(bundles: Sequence[RunBundle]) -> str:
    if bundles and all(bundle.backend == "bold5000-ridge" for bundle in bundles):
        return (
            "BOLD5000 Release 1.0 real-data ROI encoding benchmark using public "
            "measured fMRI responses, stimulus-name metadata, and local "
            "Braindough artifact machinery"
        )
    return "Braindough program plus latest perturbation/optimization runs"


def _key_findings(bundles: Sequence[RunBundle]) -> list[dict[str, str]]:
    by_backend = {bundle.backend: bundle for bundle in bundles}
    fake = by_backend.get("fake")
    tribe = by_backend.get("tribe-v2")
    bold_runs = [bundle for bundle in bundles if bundle.backend == "bold5000-ridge"]
    findings: list[dict[str, str]] = []
    if fake is not None:
        findings.append(
            {
                "title": "Fake backend validates the full artifact path",
                "evidence_strength": "artifact-supported",
                "claim": (
                    "The fake backend exercises perturbation, counterfactual, "
                    "latent, and optimization suites through durable artifacts, "
                    "tables, reports, and figures; it validates software behavior "
                    f"rather than biology. Latest fake run: {fake.n_responses}/"
                    f"{fake.n_stimuli} responses."
                ),
            }
        )
    elif not bold_runs:
        findings.append(
            {
                "title": "Fake backend run is not present in this summary",
                "evidence_strength": "not-yet-demonstrated",
                "claim": (
                    "No fake perturbation/optimization run was loaded, so this "
                    "summary cannot claim CI-style schema or full-suite software "
                    "coverage for the selected artifacts."
                ),
            }
        )
    if tribe is not None:
        findings.append(
            {
                "title": "TRIBE execution is feasible but intentionally bounded",
                "evidence_strength": "bounded-model-prediction",
                "claim": (
                    "The local TRIBE perturbation/optimization run produced model "
                    "predictions under prediction caps, but it did not evaluate "
                    f"every generated stimulus. Latest TRIBE run: "
                    f"{tribe.n_responses}/{tribe.n_stimuli} responses."
                ),
            }
        )
    for bold in bold_runs:
        benchmark = bold.metrics.get("bold5000_benchmark", {})
        best_roi = benchmark.get("best_roi", "unknown")
        best_model = benchmark.get("best_model", "unknown")
        best_r = benchmark.get("best_pearson_mean")
        significant = benchmark.get("n_nominally_significant", 0)
        context = _bold5000_run_context(bold)
        findings.append(
            {
                "title": "BOLD5000 Release 1.0 adds measured ROI response evidence",
                "evidence_strength": "measured-public-data",
                "claim": (
                    "The BOLD5000 ridge benchmark evaluates Release 1.0 processed "
                    "ROI matrices, not fake responses or TRIBE predictions. In "
                    f"the selected run, the best metadata model was {best_model} "
                    f"for {best_roi} with mean voxel Pearson r {best_r}; "
                    f"nominal exploratory uncorrected ROI results: {significant}. "
                    f"{context}"
                ),
            }
        )
        findings.append(
            {
                "title": "Metadata-only features set a real-data lower bound",
                "evidence_strength": "measured-public-data",
                "claim": (
                    "The bounded BOLD5000 run is intentionally modest: it uses "
                    "presentation metadata and labels because the direct stimuli "
                    "archive does not include raw image pixels. The small "
                    "non-significant effect is useful as a reproducible negative "
                    "control before adding COCO/ImageNet/SUN pixel features. "
                    "BOLD5000 Release 2.0 is the author-recommended source for "
                    "future functional analyses and is not evaluated here."
                ),
            }
        )
    if tribe is None and not bold_runs:
        findings.append(
            {
                "title": "TRIBE execution is not represented in selected runs",
                "evidence_strength": "not-yet-demonstrated",
                "claim": (
                    "No tribe-v2 perturbation/optimization run was loaded, so this "
                    "summary makes no local TRIBE execution, latent-decomposition, "
                    "or optimizer-budget claim for the selected artifacts."
                ),
            }
        )
    if fake is not None and tribe is not None:
        findings.append(
            {
                "title": "Latent decomposition needs more real-model samples",
                "evidence_strength": "artifact-supported",
                "claim": (
                    "The fake run computes latent components because it has enough "
                    "responses; the bounded TRIBE run records insufficient samples "
                    "for decomposition instead of overclaiming a component model."
                ),
            }
        )
        findings.append(
            {
                "title": "Optimization traces currently show pipeline mechanics",
                "evidence_strength": "artifact-supported",
                "claim": (
                    "Discrete optimization records objectives, candidates, and "
                    "stopping reasons. In the bounded TRIBE run, the stopping "
                    "reason reflects prediction budget rather than convergence."
                ),
            }
        )
    elif fake is not None:
        findings.append(
            {
                "title": "Optimization and decomposition are software-path evidence",
                "evidence_strength": "artifact-supported",
                "claim": (
                    "The fake run records objectives, candidates, stopping reasons, "
                    "and latent tables. Without a TRIBE run in this summary, those "
                    "outputs should be interpreted as pipeline validation only."
                ),
            }
        )
    elif tribe is not None:
        findings.append(
            {
                "title": "TRIBE-only evidence is bounded model-prediction evidence",
                "evidence_strength": "bounded-model-prediction",
                "claim": (
                    "The selected TRIBE run provides model-predicted responses, but "
                    "without a paired fake run this summary does not assess full "
                    "CI-style suite coverage."
                ),
            }
        )
    findings.append(
        {
            "title": "Scientific value depends on benchmark alignment",
            "evidence_strength": "literature-supported",
            "claim": (
                "Natural-scene benchmarks, semantic maps, counterfactual editing, "
                "and closed-loop optimization all support the research direction, "
                "but empirical validation against measured neural datasets remains "
                "the central next step."
            ),
        }
    )
    return findings


def _bold5000_run_context(bundle: RunBundle) -> str:
    subjects = ", ".join(str(item) for item in bundle.metrics.get("subjects", []))
    rois = ", ".join(str(item) for item in bundle.metrics.get("rois", []))
    release = bundle.metrics.get("dataset_release_label", "BOLD5000 Release 1.0")
    return (
        f"Run context: {release}; subjects {subjects or 'unknown'}; "
        f"ROIs {rois or 'unknown'}; trial limit {bundle.metrics.get('trial_limit')}; "
        f"trial offset {bundle.metrics.get('trial_offset', 0)}; validation fraction "
        f"{bundle.metrics.get('validation_fraction')}; seed {bundle.metrics.get('seed')}; "
        f"{bundle.metrics.get('permutations')} permutations; "
        f"{bundle.metrics.get('bootstraps')} bootstraps; p-values are exploratory "
        "and uncorrected for multiple ROI/model comparisons."
    )


def _limitations() -> list[str]:
    return [
        "The BOLD5000 v1 benchmark uses Release 1.0 processed ROI matrices plus metadata/labels, not Release 2.0 GLM outputs or raw image pixels.",
        "BOLD5000 Release 2.0 is recommended by the dataset authors for new functional analyses and remains future adapter scope.",
        "BOLD5000 permutation p-values are exploratory and uncorrected for multiple ROI/model comparisons.",
        "BOLD5000 source labels are coarse and should be treated as lower-bound features, not state-of-the-art encoders.",
        "TRIBE outputs here are model-predicted responses, not measured human data.",
        "Fake-backend results validate schemas, reports, and deterministic control flow only.",
        "Bounded local TRIBE runs are too small for stable effect-size or optimization claims.",
        "Current perturbations are generated controls, not validated perceptual minimal edits.",
        "Closed-loop optimization can exploit model priors unless held-out benchmarks are used.",
        "Virtual lesions are stimulus-factor lesions, not internal model-layer ablations.",
        "Focused-ultrasound bridge runs are synthetic protocol/provenance proxies, not acoustic simulations, safety evaluations, real sonication, or causal human neuromodulation evidence.",
        "Audiovisual conclusions require controlled congruent and incongruent stimulus sets.",
        "Run artifacts need empirical benchmark adapters before biological claims are warranted.",
    ]


def _future_directions() -> list[dict[str, str]]:
    return [
        {
            "phase": "Phase 0",
            "title": "Artifact and provenance hardening",
            "next_step": (
                "Keep fake-backend CI green, validate executive summaries, and track "
                "research metadata for every external synthesis."
            ),
        },
        {
            "phase": "Phase 1",
            "title": "Bounded TRIBE replication",
            "next_step": (
                "Run the same perturbation library with larger but capped TRIBE "
                "budgets to estimate stability across seeds and stimulus families."
            ),
        },
        {
            "phase": "Phase 2",
            "title": "Natural-scene benchmark adapter",
            "next_step": (
                "Add COCO/ImageNet/SUN pixel retrieval for BOLD5000 and then "
                "NSD/Algonauts-style loaders for licensed local subsets."
            ),
        },
        {
            "phase": "Phase 3",
            "title": "Counterfactual and semantic-map probes",
            "next_step": (
                "Score paired edits and semantic contrasts with explicit minimality, "
                "response-delta, and category-selectivity metrics."
            ),
        },
        {
            "phase": "Phase 4",
            "title": "Closed-loop safety rails",
            "next_step": (
                "Constrain stimulus search with held-out controls, diversity penalties, "
                "and preregistered stopping criteria before longer optimization runs."
            ),
        },
        {
            "phase": "Phase 5",
            "title": "Focused-ultrasound protocol bridge",
            "next_step": (
                "Use protocol/sham metadata and explicit claim-scope checks before "
                "considering any separate validated acoustic-simulation adapter."
            ),
        },
    ]


def _write_figures(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    charts.extend(_bold5000_charts(figures_dir, bundles))
    charts.append(_suite_coverage_chart(figures_dir, bundles))
    charts.append(_mean_abs_by_suite_chart(figures_dir, bundles))
    charts.append(_optimizer_trace_chart(figures_dir, bundles))
    charts.append(_capability_chart(figures_dir, bundles))
    return charts


def _bold5000_charts(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        if bundle.backend != "bold5000-ridge":
            continue
        for row in bundle.tables.get("bold5000_model_comparison", []):
            comparison_rows.append({**row, "run_id": bundle.run_id})
        for row in bundle.tables.get("bold5000_roi_scores", []):
            score_rows.append({**row, "run_id": bundle.run_id})
    if not comparison_rows:
        return []
    return [
        _bold5000_roi_score_chart(figures_dir, comparison_rows),
        _bold5000_model_comparison_chart(figures_dir, score_rows),
    ]


def _bold5000_roi_score_chart(
    figures_dir: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    labels = [f"{row.get('subject')} {row.get('roi')}" for row in rows]
    values = [_float_value(row.get("best_pearson_mean"), 0.0) for row in rows]
    lows = [
        _float_value(row.get("bootstrap_low"), value)
        for row, value in zip(rows, values, strict=False)
    ]
    highs = [
        _float_value(row.get("bootstrap_high"), value)
        for row, value in zip(rows, values, strict=False)
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(range(len(values)), values, color="#3867d6")
    ax.errorbar(
        range(len(values)),
        values,
        yerr=[
            [max(0.0, value - low) for value, low in zip(values, lows, strict=False)],
            [
                max(0.0, high - value)
                for value, high in zip(values, highs, strict=False)
            ],
        ],
        fmt="none",
        ecolor="#2f3640",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="#2f3640", linewidth=0.8)
    ax.set_title("BOLD5000 validation correlation by ROI")
    ax.set_ylabel("mean voxel Pearson r")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "bold5000_roi_scores.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "BOLD5000 validation correlation by ROI",
        "path": f"figures/{path.name}",
        "purpose": "Shows held-out metadata-to-ROI prediction scores for measured BOLD5000 Release 1.0 ROI responses.",
        "data_source": "outputs/tables/bold5000_model_comparison.csv",
        "interpretation": "Values near zero indicate that coarse metadata has little predictive power on the bounded split.",
        "caveat": "This is not a Release 2.0, pixel-feature, CLIP, DINO, TRIBE, NSD, or hidden-leaderboard benchmark.",
    }


def _bold5000_model_comparison_chart(
    figures_dir: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    model_names = sorted(
        {
            str(row.get("model"))
            for row in rows
            if row.get("model") and row.get("model") != "mean_baseline"
        }
    )
    means = []
    for model in model_names:
        model_values = [
            _float_value(row.get("pearson_mean"), 0.0)
            for row in rows
            if row.get("model") == model
        ]
        means.append(sum(model_values) / len(model_values) if model_values else 0.0)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if model_names:
        ax.bar(range(len(model_names)), means, color="#20bf6b")
        ax.axhline(0.0, color="#2f3640", linewidth=0.8)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(
            [_labelize(model) for model in model_names], rotation=25, ha="right"
        )
    else:
        ax.text(0.5, 0.5, "No metadata model scores available", ha="center")
        ax.set_axis_off()
    ax.set_title("BOLD5000 metadata model comparison")
    ax.set_ylabel("mean ROI Pearson r")
    fig.tight_layout()
    path = figures_dir / "bold5000_model_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "BOLD5000 metadata model comparison",
        "path": f"figures/{path.name}",
        "purpose": "Compares source-family, token-hash, and combined metadata baselines across ROIs.",
        "data_source": "outputs/tables/bold5000_roi_scores.csv",
        "interpretation": "The best simple model becomes the reproducible lower bound for later pixel encoders.",
        "caveat": "Coarse labels are not expected to match modern visual encoding literature.",
    }


def _suite_coverage_chart(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> dict[str, Any]:
    suites = _suite_names(bundles)
    backends = [bundle.backend for bundle in bundles]
    bold_only = bool(bundles) and all(
        bundle.backend == "bold5000-ridge" for bundle in bundles
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.8 / max(1, len(bundles))
    offsets = [(-0.4 + width / 2) + idx * width for idx in range(len(bundles))]
    for offset, bundle in zip(offsets, bundles, strict=False):
        values = []
        for suite in suites:
            summary = bundle.metrics.get("suite_summary", {}).get(suite, {})
            stimuli = float(summary.get("stimuli", 0) or 0)
            responses = float(summary.get("responses", 0) or 0)
            values.append(responses / stimuli if stimuli else 0.0)
        ax.bar(
            [index + offset for index in range(len(suites))],
            values,
            width=width,
            label=bundle.backend,
        )
    ax.set_title("Suite response coverage by backend")
    ax.set_ylabel("responses / stimuli")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels([_labelize(suite) for suite in suites], rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "suite_response_coverage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "Suite response coverage by backend",
        "path": f"figures/{path.name}",
        "purpose": "Shows which suites produced predictions under each backend.",
        "data_source": "metrics.json suite_summary",
        "interpretation": (
            "BOLD5000 response counts are ROI/model prediction arrays, not one array per trial."
            if bold_only
            else "Fake should be complete; bounded TRIBE should show capped coverage."
        ),
        "caveat": "Coverage is a runtime budget signal, not a biological effect.",
        "backends": backends,
    }


def _mean_abs_by_suite_chart(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> dict[str, Any]:
    suites = _suite_names(bundles)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.8 / max(1, len(bundles))
    offsets = [(-0.4 + width / 2) + idx * width for idx in range(len(bundles))]
    for offset, bundle in zip(offsets, bundles, strict=False):
        by_suite = _mean_abs_by_suite(bundle)
        ax.bar(
            [index + offset for index in range(len(suites))],
            [by_suite.get(suite, 0.0) for suite in suites],
            width=width,
            label=bundle.backend,
        )
    ax.set_title("Mean absolute predicted response by suite")
    ax.set_ylabel("mean |predicted response|")
    ax.set_xticks(range(len(suites)))
    ax.set_xticklabels([_labelize(suite) for suite in suites], rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "mean_abs_by_suite.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "Mean absolute predicted response by suite",
        "path": f"figures/{path.name}",
        "purpose": "Compares coarse response magnitude summaries across suites.",
        "data_source": "metrics.json mean_abs_activation_by_stimulus",
        "interpretation": "Useful for spotting gross shifts that merit follow-up.",
        "caveat": "Magnitude is model-scale dependent and not directly comparable to measured fMRI.",
    }


def _optimizer_trace_chart(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plotted = False
    for bundle in bundles:
        rows = bundle.tables.get("optimization_history", [])
        evaluated_rows = [
            row
            for row in rows
            if row.get("status", "evaluated") == "evaluated"
            and _float_or_none(row.get("objective_score")) is not None
            and _float_or_none(row.get("best_score_so_far")) is not None
        ]
        if not evaluated_rows:
            continue
        plotted = True
        steps = [
            int(row.get("step", index)) for index, row in enumerate(evaluated_rows)
        ]
        best = [
            _float_value(row.get("best_score_so_far"), 0.0) for row in evaluated_rows
        ]
        score = [
            _float_value(row.get("objective_score"), 0.0) for row in evaluated_rows
        ]
        ax.plot(
            steps, score, marker="o", linewidth=1.2, label=f"{bundle.backend} score"
        )
        ax.plot(steps, best, marker="s", linewidth=1.2, label=f"{bundle.backend} best")
    if not plotted:
        ax.text(0.5, 0.5, "No optimizer history available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.set_title("Discrete optimizer trace")
        ax.set_xlabel("candidate step")
        ax.set_ylabel("objective score")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "optimizer_trace.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "Discrete optimizer trace",
        "path": f"figures/{path.name}",
        "purpose": "Shows candidate scoring and best-so-far behavior.",
        "data_source": "outputs/tables/optimization_history.jsonl",
        "interpretation": "Use stopping reason with trace shape before claiming convergence.",
        "caveat": "Bounded TRIBE traces can stop because of prediction budget.",
    }


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any, default: float) -> float:
    parsed = _float_or_none(value)
    return default if parsed is None else parsed


def _capability_chart(
    figures_dir: Path, bundles: Sequence[RunBundle]
) -> dict[str, Any]:
    bold_only = bool(bundles) and all(
        bundle.backend == "bold5000-ridge" for bundle in bundles
    )
    metrics = [
        ("Stimuli", "n_stimuli"),
        ("Responses", "n_responses"),
        ("Optimizer candidates", "n_optimizer_candidates"),
        ("Perturbation pairs", "n_perturbation_comparisons"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    width = 0.8 / max(1, len(bundles))
    offsets = [(-0.4 + width / 2) + idx * width for idx in range(len(bundles))]
    for offset, bundle in zip(offsets, bundles, strict=False):
        values = []
        payload = bundle.to_payload()
        for _, key in metrics:
            values.append(float(payload.get(key, bundle.metrics.get(key, 0)) or 0))
        ax.bar(
            [index + offset for index in range(len(metrics))],
            values,
            width=width,
            label=bundle.backend,
        )
    ax.set_title("Artifact capability comparison")
    ax.set_ylabel("count")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([label for label, _ in metrics], rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "artifact_capability_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {
        "title": "Artifact capability comparison",
        "path": f"figures/{path.name}",
        "purpose": (
            "Shows BOLD5000 run size, ROI response arrays, and absence of optimizer-only outputs."
            if bold_only
            else "Contrasts fake CI coverage with bounded local TRIBE coverage."
        ),
        "data_source": "manifest.json and metrics.json",
        "interpretation": (
            "The benchmark is an encoding evaluation, so optimizer and perturbation counts should be zero."
            if bold_only
            else "Large fake counts validate software paths; TRIBE counts reflect local budget."
        ),
        "caveat": (
            "Counts summarize artifact shape, not scientific effect size."
            if bold_only
            else "Count parity is not expected until TRIBE budgets are expanded."
        ),
    }


def _write_pdf(
    pdf_path: Path, payload: dict[str, Any], charts: Sequence[dict[str, Any]]
) -> None:
    with PdfPages(pdf_path) as pdf:
        _write_text_page(
            pdf,
            "Braindough Executive Summary",
            [
                payload["scope"],
                payload["disclaimer"],
                "Audience: research leads and senior technical stakeholders.",
                f"Generated: {payload['generated_at']}",
            ],
        )
        _write_text_page(
            pdf,
            "Key Findings",
            [
                f"{item['title']} ({item['evidence_strength']}): {item['claim']}"
                for item in payload["key_findings"]
            ],
        )
        _write_runs_page(pdf, payload["runs"])
        for chunk in _chunks(list(charts), 2):
            _write_chart_page(pdf, pdf_path.parent, chunk)
        _write_text_page(pdf, "Limitations", payload["limitations"])
        _write_text_page(
            pdf,
            "Future Directions",
            [
                f"{item['phase']} - {item['title']}: {item['next_step']}"
                for item in payload["future_directions"]
            ],
        )
        _write_text_page(
            pdf,
            "Literature Grounding",
            [
                f"{item['title']}: {item['relevance']} Source: {item['url']}"
                for item in payload["sources"][:12]
            ],
            footer="Full source metadata is in sources.json.",
        )


def _write_text_page(
    pdf: PdfPages, title: str, paragraphs: Sequence[str], footer: str | None = None
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    y = 0.94
    fig.text(0.07, y, title, fontsize=22, fontweight="bold", va="top")
    y -= 0.06
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=92)
        for line in lines:
            fig.text(0.08, y, line, fontsize=10.5, va="top")
            y -= 0.028
        y -= 0.018
        if y < 0.1:
            fig.text(0.08, y, "Continued in generated Markdown summary.", fontsize=9)
            break
    if footer:
        fig.text(0.08, 0.045, footer, fontsize=8.5, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def _write_runs_page(pdf: PdfPages, runs: Sequence[dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.94, "Run Coverage", fontsize=22, fontweight="bold", va="top")
    columns = ["Backend", "Responses", "Stimuli", "Optimization", "Latent"]
    rows = []
    for run in runs:
        optimization = run.get("optimization", {})
        latent = run.get("latent_components", {})
        rows.append(
            [
                run.get("backend", ""),
                str(run.get("n_responses", 0)),
                str(run.get("n_stimuli", 0)),
                str(optimization.get("stopping_reason", "n/a")),
                str(latent.get("status", "n/a")),
            ]
        )
    ax = fig.add_axes((0.07, 0.35, 0.86, 0.45))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    fig.text(
        0.08,
        0.25,
        (
            "Interpretation: BOLD5000 response counts are compact ROI/model "
            "prediction arrays used for artifact compatibility; scientific "
            "interpretation comes from the ROI benchmark tables."
            if runs and all(run.get("backend") == "bold5000-ridge" for run in runs)
            else "Interpretation: complete fake coverage validates software contracts; "
            "bounded TRIBE coverage demonstrates local model execution under caps."
        ),
        fontsize=10.5,
        va="top",
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _write_chart_page(
    pdf: PdfPages, output_dir: Path, charts: Sequence[dict[str, Any]]
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.94, "Executive Charts", fontsize=22, fontweight="bold", va="top")
    y_positions = [0.54, 0.12]
    for chart, bottom in zip(charts, y_positions, strict=False):
        path = output_dir / chart["path"]
        ax = fig.add_axes((0.08, bottom + 0.12, 0.84, 0.25))
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        fig.text(0.08, bottom + 0.08, chart["title"], fontsize=12, fontweight="bold")
        fig.text(
            0.08,
            bottom + 0.045,
            f"Purpose: {chart['purpose']} Caveat: {chart['caveat']}",
            fontsize=8.8,
            wrap=True,
        )
    pdf.savefig(fig)
    plt.close(fig)


def _markdown_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# Braindough Executive Summary",
        "",
        f"Audience: `{payload['audience']}`",
        "",
        f"> {payload['disclaimer']}",
        "",
        "## Key Findings",
        "",
    ]
    for item in payload["key_findings"]:
        lines.append(
            f"- **{item['title']}** (`{item['evidence_strength']}`): {item['claim']}"
        )
    lines.extend(["", "## Runs", ""])
    for run in payload["runs"]:
        lines.append(
            "- "
            f"`{run['backend']}` `{run['run_id']}`: "
            f"{run['n_responses']}/{run['n_stimuli']} responses, "
            f"status `{run['status']}`, evidence `{run['evidence_label']}`."
        )
    lines.extend(["", "## Charts", ""])
    for chart in payload["charts"]:
        lines.append(f"- `{chart['path']}` - {chart['purpose']} {chart['caveat']}")
    lines.extend(["", "## Limitations", ""])
    lines.extend([f"- {item}" for item in payload["limitations"]])
    lines.extend(["", "## Future Directions", ""])
    lines.extend(
        [
            f"- **{item['phase']}: {item['title']}** - {item['next_step']}"
            for item in payload["future_directions"]
        ]
    )
    lines.extend(["", "## Sources", ""])
    for source in payload["sources"]:
        lines.append(f"- [{source['title']}]({source['url']}) - {source['relevance']}")
    lines.append("")
    return "\n".join(lines)


def _literature_sources() -> list[dict[str, str]]:
    return [
        {
            "id": "bold5000_download",
            "title": "BOLD5000 Dataset Download",
            "url": "https://bold5000-dataset.github.io/website/download.html",
            "kind": "dataset",
            "relevance": "Primary source for Release 1.0 and Release 2.0 data availability.",
        },
        {
            "id": "bold5000_terms",
            "title": "BOLD5000 Terms of Use",
            "url": "https://bold5000-dataset.github.io/website/terms.html",
            "kind": "dataset_terms",
            "relevance": "Primary source for fMRI data licensing and stimulus-image exclusions.",
        },
        {
            "id": "bold5000_release_2_code",
            "title": "BOLD5000 Release 2.0 code",
            "url": "https://github.com/BOLD5000-dataset/BOLD5000",
            "kind": "dataset_code",
            "relevance": "Primary source noting Release 2.0 is recommended for functional analyses.",
        },
        {
            "id": "tribe_v2_model_card",
            "title": "facebook/tribev2 model card",
            "url": "https://huggingface.co/facebook/tribev2",
            "kind": "model_card",
            "relevance": "Primary public checkpoint and usage reference for TRIBE v2.",
        },
        {
            "id": "tribe_v2_meta_paper_page",
            "title": "A foundation model of vision, audition, and language for in-silico neuroscience",
            "url": "https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/",
            "kind": "publication_page",
            "relevance": "Primary Meta research page for the TRIBE v2 program.",
        },
        {
            "id": "natural_scenes_dataset",
            "title": "Natural Scenes Dataset",
            "url": "https://naturalscenesdataset.org/",
            "kind": "dataset",
            "relevance": "Large-scale natural-image fMRI benchmark target for future adapters.",
        },
        {
            "id": "algonauts_2023",
            "title": "Algonauts Project 2023 Challenge",
            "url": "https://algonautsproject.com/2023/challenge.html",
            "kind": "benchmark",
            "relevance": "Natural-scene brain-response prediction benchmark framing.",
        },
        {
            "id": "neurogen",
            "title": "NeuroGen: activation maximization for visual cortex",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8845078/",
            "kind": "paper",
            "relevance": "Grounding for closed-loop and optimized-stimulus experiments.",
        },
        {
            "id": "semantic_maps",
            "title": "Natural speech reveals the semantic maps that tile human cerebral cortex",
            "url": "https://www.nature.com/articles/nature17637",
            "kind": "paper",
            "relevance": "Grounding for semantic-map probes and category-selectivity roadmaps.",
        },
        {
            "id": "mcgurk_macdonald",
            "title": "Hearing lips and seeing voices",
            "url": "https://www.nature.com/articles/264746a0",
            "kind": "paper",
            "relevance": "Classic audiovisual-integration motivation for multimodal probes.",
        },
        {
            "id": "counterfactual_explanations",
            "title": "Counterfactual explanations without opening the black box",
            "url": "https://arxiv.org/abs/1711.00399",
            "kind": "paper",
            "relevance": "Minimal-edit counterfactual framing for paired stimulus changes.",
        },
        {
            "id": "feature_visualization",
            "title": "Feature Visualization",
            "url": "https://distill.pub/2017/feature-visualization/",
            "kind": "technical_article",
            "relevance": "Activation-maximization and interpretability caveats for optimizers.",
        },
        {
            "id": "itrusst_reporting",
            "title": "ITRUSST Consensus on Standardised Reporting for Transcranial Ultrasound Stimulation",
            "url": "https://arxiv.org/abs/2402.10027",
            "kind": "paper",
            "relevance": "Reporting-field anchor for focused-ultrasound protocol metadata and explicit out-of-scope acoustic fields.",
        },
        {
            "id": "lifu_human_primer",
            "title": "Low-intensity focused ultrasound for human neuromodulation",
            "url": "https://www.nature.com/articles/s43586-024-00368-6",
            "kind": "paper",
            "relevance": "Human LIFU methods and limitation anchor for protocol-bridge wording.",
        },
    ]


def _research_sources(research_dir: Path) -> list[dict[str, str]]:
    if not research_dir.is_dir():
        return []
    sources: list[dict[str, str]] = []
    for metadata_path in sorted(research_dir.glob("*.metadata.json")):
        try:
            metadata = _read_json(metadata_path)
        except (json.JSONDecodeError, OSError):
            continue
        research_id = metadata.get("id")
        if not isinstance(research_id, str):
            continue
        if "executive" not in research_id and "project" not in research_id:
            continue
        source = metadata.get("source", {})
        url = source.get("url") if isinstance(source, dict) else None
        sources.append(
            {
                "id": f"research_capture_{research_id}",
                "title": str(metadata.get("title", research_id)),
                "url": str(url or "docs/research"),
                "kind": "research_capture",
                "relevance": (
                    f"ChatGPT research capture status: {metadata.get('status', 'unknown')}."
                ),
            }
        )
    return sources


def _resolve_repo_root(repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root)
    for start in (Path.cwd(), Path(__file__).resolve()):
        root = _find_repo_root(start)
        if root is not None:
            return root
    return Path.cwd()


def _find_repo_root(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if (candidate / "docs" / "research").is_dir() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate
    return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tables(root: Path) -> dict[str, Any]:
    tables_dir = root / "outputs" / "tables"
    if not tables_dir.is_dir():
        return {}
    tables: dict[str, Any] = {}
    for path in sorted(tables_dir.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
    history_path = tables_dir / "optimization_history.jsonl"
    if history_path.is_file():
        tables["optimization_history"] = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    objectives_path = tables_dir / "objectives.json"
    if objectives_path.is_file():
        tables["objectives"] = _read_json(objectives_path)
    return tables


def _suite_names(bundles: Sequence[RunBundle]) -> list[str]:
    suites: set[str] = set()
    for bundle in bundles:
        suites.update(bundle.metrics.get("suite_summary", {}).keys())
    return sorted(suites)


def _mean_abs_by_suite(bundle: RunBundle) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for stimulus_id, value in bundle.metrics.get(
        "mean_abs_activation_by_stimulus", {}
    ).items():
        suite = str(stimulus_id).split(":", maxsplit=1)[0]
        values.setdefault(suite, []).append(float(value))
    return {
        suite: sum(suite_values) / len(suite_values)
        for suite, suite_values in values.items()
        if suite_values
    }


def _labelize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _chunks(
    items: Sequence[dict[str, Any]], size: int
) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
