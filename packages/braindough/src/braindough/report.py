"""Human-readable report generation."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from braindough.analysis import similarity_matrix


def write_report(run_dir: str | Path) -> tuple[Path, ...]:
    """Write Markdown, HTML, PDF, and figures for a run directory."""

    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    metrics = _read_json(root / "metrics.json")
    next_experiments = _read_json(root / "next_experiments.json")
    responses = _read_responses(root / "outputs" / "responses.npz")
    tables = _read_tables(root)
    figures = _write_figures(root, manifest, responses, tables)
    markdown = _markdown_report(manifest, metrics, next_experiments, figures, tables)
    md_path = root / "report.md"
    html_path = root / "report.html"
    pdf_path = root / "executive_summary.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_html_report(markdown), encoding="utf-8")
    _write_pdf(pdf_path, manifest, metrics, figures, tables)
    return md_path, html_path, pdf_path, *figures


def _write_figures(
    root: Path,
    manifest: dict[str, Any],
    responses: dict[str, np.ndarray],
    tables: dict[str, Any],
) -> list[Path]:
    figures: list[Path] = []
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)
    if not responses:
        return figures

    ids, matrix = similarity_matrix(responses)
    fig, ax = plt.subplots(figsize=(max(5, len(ids) * 0.35), 4))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Response similarity")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([_short_id(item) for item in ids], rotation=90, fontsize=7)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([_short_id(item) for item in ids], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = figures_dir / "response_similarity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    activations = {
        key: float(np.mean(np.abs(value))) for key, value in sorted(responses.items())
    }
    fig, ax = plt.subplots(figsize=(max(5, len(activations) * 0.35), 4))
    ax.bar(range(len(activations)), list(activations.values()), color="#4b7bec")
    ax.set_title("Mean absolute activation")
    ax.set_xticks(range(len(activations)))
    ax.set_xticklabels(
        [_short_id(item) for item in activations], rotation=90, fontsize=7
    )
    ax.set_ylabel("mean |response|")
    fig.tight_layout()
    path = figures_dir / "mean_abs_activation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)
    figures.extend(_write_bold5000_figures(root, manifest, tables))
    perturbation_rows = tables.get("perturbation_comparisons", [])
    completed_perturbation_rows = [
        row
        for row in perturbation_rows
        if _float_or_none(row.get("mean_abs_delta")) is not None
    ]
    if completed_perturbation_rows:
        labels = [
            _short_id(str(row.get("stimulus_id", "")))
            for row in completed_perturbation_rows[
                : min(20, len(completed_perturbation_rows))
            ]
        ]
        deltas = [
            _float_value(row.get("mean_abs_delta"), 0.0)
            for row in completed_perturbation_rows[
                : min(20, len(completed_perturbation_rows))
            ]
        ]
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.4), 4))
        ax.bar(range(len(deltas)), deltas, color="#20bf6b")
        ax.axhline(0.0, color="#2f3640", linewidth=0.8)
        ax.set_title("Perturbation response deltas")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("delta mean |response|")
        fig.tight_layout()
        path = figures_dir / "perturbation_deltas.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)

    optimization_rows = tables.get("optimization_history", [])
    if optimization_rows or _manifest_has_suite(
        manifest, "discrete_stimulus_optimizer"
    ):
        evaluated_rows = [
            row for row in optimization_rows if row.get("status") == "evaluated"
        ]
    else:
        evaluated_rows = []
    if evaluated_rows:
        steps = [int(row.get("step", 0)) for row in evaluated_rows]
        scores = [
            _float_value(row.get("objective_score"), 0.0) for row in evaluated_rows
        ]
        best = [
            _float_value(row.get("best_score_so_far"), 0.0) for row in evaluated_rows
        ]
        fig, ax = plt.subplots(figsize=(max(5, len(steps) * 0.35), 4))
        ax.plot(steps, scores, marker="o", label="candidate score")
        ax.plot(steps, best, marker="s", label="best so far")
        ax.set_title("Discrete optimizer trace")
        ax.set_xlabel("step")
        ax.set_ylabel("objective score")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "optimization_trace.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    elif optimization_rows:
        figures.append(
            _write_no_data_figure(
                figures_dir / "optimization_trace.png",
                "Discrete optimizer trace",
                "No evaluated optimizer candidates were available.",
            )
        )
    if optimization_rows or _manifest_has_suite(
        manifest, "discrete_stimulus_optimizer"
    ):
        figures.extend(_write_optimizer_figures(root, manifest, optimization_rows))
    figures.extend(_write_lesion_figures(root, manifest, tables))
    figures.extend(_write_counterfactual_figures(root, manifest, tables))
    figures.extend(_write_focused_ultrasound_figures(root, manifest, tables))
    return figures


def _write_bold5000_figures(
    root: Path, manifest: dict[str, Any], tables: dict[str, Any]
) -> list[Path]:
    figures: list[Path] = []
    if not _manifest_has_suite(manifest, "bold5000_roi_encoding"):
        return figures
    figures_dir = root / "figures"
    rows = tables.get("bold5000_model_comparison", [])
    if not rows:
        figures.append(
            _write_no_data_figure(
                figures_dir / "bold5000_roi_scores.png",
                "BOLD5000 ROI scores",
                "No BOLD5000 ROI scores were available.",
            )
        )
        figures.append(
            _write_no_data_figure(
                figures_dir / "bold5000_model_comparison.png",
                "BOLD5000 model comparison",
                "No BOLD5000 model comparison rows were available.",
            )
        )
        return figures

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
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.55), 4))
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
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path)

    score_rows = tables.get("bold5000_roi_scores", [])
    model_names = sorted(
        {
            str(row.get("model"))
            for row in score_rows
            if row.get("model") and row.get("model") != "mean_baseline"
        }
    )
    if model_names:
        means = [
            np.mean(
                [
                    _float_value(row.get("pearson_mean"), 0.0)
                    for row in score_rows
                    if row.get("model") == model
                ]
            )
            for model in model_names
        ]
        fig, ax = plt.subplots(figsize=(max(5, len(model_names) * 1.2), 4))
        ax.bar(range(len(model_names)), means, color="#20bf6b")
        ax.axhline(0.0, color="#2f3640", linewidth=0.8)
        ax.set_title("BOLD5000 metadata model comparison")
        ax.set_ylabel("mean ROI Pearson r")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(
            [_labelize(model) for model in model_names], rotation=25, ha="right"
        )
        fig.tight_layout()
        path = figures_dir / "bold5000_model_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "bold5000_model_comparison.png",
                "BOLD5000 model comparison",
                "No non-baseline BOLD5000 model rows were available.",
            )
        )
    return figures


def _write_lesion_figures(
    root: Path, manifest: dict[str, Any], tables: dict[str, Any]
) -> list[Path]:
    figures: list[Path] = []
    if not _manifest_has_suite(manifest, "virtual_lesion_lab"):
        return figures
    rows = [
        row
        for row in tables.get("lesion_comparisons", [])
        if row.get("complete_pair") in {"True", True}
    ]
    figures_dir = root / "figures"
    contact = _write_contact_sheet(
        root,
        manifest,
        suite="virtual_lesion_lab",
        filename="virtual_lesion_contact_sheet.png",
        title="Virtual lesion stimuli",
    )
    if contact:
        figures.append(contact)
    if rows:
        labels = [_short_id(str(row.get("stimulus_id", ""))) for row in rows[:20]]
        deltas = [_float_value(row.get("mean_abs_delta"), 0.0) for row in rows[:20]]
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.42), 4))
        ax.bar(range(len(deltas)), deltas, color="#eb3b5a")
        ax.axhline(0.0, color="#2f3640", linewidth=0.8)
        ax.set_title("Virtual lesion scoreboard")
        ax.set_ylabel("delta mean |response|")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        fig.tight_layout()
        path = figures_dir / "lesion_scoreboard.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)

    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "lesion_scoreboard.png",
                "Virtual lesion scoreboard",
                "No complete lesion pairs were available.",
            )
        )

    strength_rows = [
        row for row in rows if _float_or_none(row.get("strength")) is not None
    ]
    if strength_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [_float_value(row.get("strength"), 0.0) for row in strength_rows],
            [_float_value(row.get("mean_abs_delta"), 0.0) for row in strength_rows],
            color="#8854d0",
        )
        ax.axhline(0.0, color="#2f3640", linewidth=0.8)
        ax.set_title("Lesion dose response")
        ax.set_xlabel("lesion strength")
        ax.set_ylabel("delta mean |response|")
        fig.tight_layout()
        path = figures_dir / "lesion_dose_response.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "lesion_dose_response.png",
                "Lesion dose response",
                "No complete lesion pairs with strength metadata were available.",
            )
        )
    return figures


def _write_optimizer_figures(
    root: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> list[Path]:
    figures: list[Path] = []
    figures_dir = root / "figures"
    contact = _write_contact_sheet(
        root,
        manifest,
        suite="discrete_stimulus_optimizer",
        filename="optimizer_candidate_contact_sheet.png",
        title="Optimizer candidates",
    )
    if contact:
        figures.append(contact)
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    if evaluated:
        steps = [int(row.get("step", 0)) for row in evaluated]
        mean_abs = [
            _float_value(row.get("mean_abs_activation"), 0.0) for row in evaluated
        ]
        penalty = [
            _float_value(row.get("similarity_penalty"), 0.0) for row in evaluated
        ]
        scores = [_float_value(row.get("objective_score"), 0.0) for row in evaluated]
        fig, ax = plt.subplots(figsize=(max(5, len(steps) * 0.35), 4))
        ax.plot(steps, mean_abs, marker="o", label="mean |response|")
        ax.plot(steps, penalty, marker="s", label="similarity penalty")
        ax.plot(steps, scores, marker="^", label="objective")
        ax.set_title("Optimizer score components")
        ax.set_xlabel("step")
        ax.legend()
        fig.tight_layout()
        path = figures_dir / "optimization_score_components.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "optimization_score_components.png",
                "Optimizer score components",
                "No evaluated optimizer candidates were available.",
            )
        )
    return figures


def _write_counterfactual_figures(
    root: Path, manifest: dict[str, Any], tables: dict[str, Any]
) -> list[Path]:
    figures: list[Path] = []
    if not _manifest_has_suite(manifest, "counterfactual_editing_workbench"):
        return figures
    rows = [
        row
        for row in tables.get("counterfactual_pairs", [])
        if row.get("complete_pair") in {"True", True}
    ]
    figures_dir = root / "figures"
    contact = _write_contact_sheet(
        root,
        manifest,
        suite="counterfactual_editing_workbench",
        filename="counterfactual_delta_grid.png",
        title="Counterfactual edits",
    )
    if contact:
        figures.append(contact)
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [_float_value(row.get("changed_pixel_fraction"), 0.0) for row in rows],
            [_float_value(row.get("normalized_l2_delta"), 0.0) for row in rows],
            color="#0fb9b1",
        )
        ax.set_title("Counterfactual edit magnitude vs response delta")
        ax.set_xlabel("changed pixel fraction")
        ax.set_ylabel("normalized L2 response delta")
        fig.tight_layout()
        path = figures_dir / "counterfactual_tradeoff.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "counterfactual_tradeoff.png",
                "Counterfactual edit magnitude vs response delta",
                "No complete counterfactual pairs were available.",
            )
        )
    return figures


def _write_focused_ultrasound_figures(
    root: Path, manifest: dict[str, Any], tables: dict[str, Any]
) -> list[Path]:
    figures: list[Path] = []
    if not _manifest_has_suite(manifest, "focused_ultrasound_bridge"):
        return figures
    rows = [
        row
        for row in tables.get("focused_ultrasound_comparisons", [])
        if row.get("complete_pair") in {"True", True}
    ]
    figures_dir = root / "figures"
    contact = _write_contact_sheet(
        root,
        manifest,
        suite="focused_ultrasound_bridge",
        filename="focused_ultrasound_contact_sheet.png",
        title="Focused ultrasound protocol proxies",
    )
    if contact:
        figures.append(contact)

    if rows:
        labels = [
            f"{row.get('target_label')} {row.get('protocol_id')}" for row in rows[:20]
        ]
        deltas = [
            _float_value(row.get("normalized_l2_delta"), 0.0) for row in rows[:20]
        ]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), 4))
        ax.bar(range(len(deltas)), deltas, color="#2d98da")
        ax.set_title("Focused ultrasound proxy response deltas")
        ax.set_ylabel("normalized L2 response delta")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        fig.tight_layout()
        path = figures_dir / "focused_ultrasound_protocol_effects.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)

        dose_rows = [
            row
            for row in rows
            if _float_or_none(row.get("software_dose_index")) is not None
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            [_float_value(row.get("software_dose_index"), 0.0) for row in dose_rows],
            [_float_value(row.get("normalized_l2_delta"), 0.0) for row in dose_rows],
            color="#20bf6b",
        )
        ax.set_title("Software dose proxy vs response delta")
        ax.set_xlabel("software dose index")
        ax.set_ylabel("normalized L2 response delta")
        fig.tight_layout()
        path = figures_dir / "focused_ultrasound_dose_proxy.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    else:
        figures.append(
            _write_no_data_figure(
                figures_dir / "focused_ultrasound_protocol_effects.png",
                "Focused ultrasound proxy response deltas",
                "No complete focused-ultrasound proxy pairs were available.",
            )
        )
        figures.append(
            _write_no_data_figure(
                figures_dir / "focused_ultrasound_dose_proxy.png",
                "Software dose proxy vs response delta",
                "No complete focused-ultrasound proxy pairs were available.",
            )
        )
    return figures


def _write_contact_sheet(
    root: Path,
    manifest: dict[str, Any],
    *,
    suite: str,
    filename: str,
    title: str,
) -> Path | None:
    image_paths: list[tuple[str, Path]] = []
    for item in manifest.get("inputs", []):
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict) or metadata.get("suite") != suite:
            continue
        source_image = metadata.get("source_image")
        if isinstance(source_image, str) and source_image:
            path = root / source_image
            if path.is_file():
                image_paths.append((_short_id(str(item.get("id", ""))), path))
    if not image_paths:
        return None
    image_paths = image_paths[:12]
    cols = min(4, len(image_paths))
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, (label, path) in zip(axes_array, image_paths, strict=False):
        ax.imshow(plt.imread(path))
        ax.set_title(label, fontsize=7)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output = root / "figures" / filename
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def _write_no_data_figure(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _manifest_has_suite(manifest: dict[str, Any], suite: str) -> bool:
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict) and metadata.get("suite") == suite:
            return True
    return False


def _markdown_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    next_experiments: list[dict[str, str]],
    figures: list[Path],
    tables: dict[str, Any],
) -> str:
    lines = [
        f"# Braindough Run {manifest.get('run_id', 'unknown')}",
        "",
        f"- Status: `{manifest.get('status')}`",
        f"- Backend: `{manifest.get('backend', {}).get('name')}`",
        f"- Experiment: `{manifest.get('config', {}).get('experiment_id')}`",
        f"- Stimuli: `{len(manifest.get('inputs', []))}`",
        f"- Responses: `{metrics.get('n_responses', 0)}`",
    ]
    if manifest.get("blocker"):
        lines.append(f"- Blocker: {manifest['blocker']}")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, indent=2, sort_keys=True),
            "```",
            "",
            "## Figures",
            "",
        ]
    )
    if figures:
        lines.extend(
            [f"- `{figure.relative_to(figure.parents[1])}`" for figure in figures]
        )
    else:
        lines.append("- No response figures were generated.")

    lines.extend(["", "## Perturbation And Optimization", ""])
    perturbation_rows = tables.get("perturbation_comparisons", [])
    optimization_rows = tables.get("optimization_history", [])
    objectives = tables.get("objectives", {})
    completed_perturbations = [
        row
        for row in perturbation_rows
        if _float_or_none(row.get("mean_abs_delta")) is not None
    ]
    if completed_perturbations:
        top_delta = max(
            completed_perturbations,
            key=lambda row: abs(_float_value(row.get("mean_abs_delta"), 0.0)),
        )
        lines.append(
            "- Largest perturbation delta: "
            f"`{top_delta.get('stimulus_id')}` vs `{top_delta.get('parent_id')}` "
            f"({_float_value(top_delta.get('mean_abs_delta'), 0.0):.6f})."
        )
    else:
        lines.append("- No completed perturbation pairs were available.")
    evaluated_optimizer_rows = [
        row for row in optimization_rows if row.get("status") == "evaluated"
    ]
    if evaluated_optimizer_rows:
        top_rows = sorted(
            evaluated_optimizer_rows,
            key=lambda row: _float_value(row.get("objective_score"), 0.0),
            reverse=True,
        )[:3]
        lines.append(
            "- Top optimizer candidates: "
            + ", ".join(
                f"`{row.get('stimulus_id')}` "
                f"({_float_value(row.get('objective_score'), 0.0):.6f})"
                for row in top_rows
            )
            + "."
        )
        if objectives:
            lines.append(
                "- Optimizer stopping reason: "
                f"`{objectives.get('stopping_reason', 'unknown')}`."
            )
    else:
        lines.append("- No optimizer candidates were scored.")

    lines.extend(["", "## Virtual Lesion Lab", ""])
    lesion_rows = tables.get("lesion_comparisons", [])
    complete_lesions = [
        row for row in lesion_rows if row.get("complete_pair") in {"True", True}
    ]
    lines.append(
        f"- Lesion pairs: `{len(complete_lesions)}` complete of `{len(lesion_rows)}` planned."
    )
    if complete_lesions:
        top_lesion = max(
            complete_lesions,
            key=lambda row: abs(_float_value(row.get("normalized_l2_delta"), 0.0)),
        )
        lines.append(
            "- Largest descriptive lesion effect: "
            f"`{top_lesion.get('stimulus_id')}` "
            f"(normalized L2 `{_float_value(top_lesion.get('normalized_l2_delta'), 0.0):.6f}`)."
        )
    lines.append(
        "- Interpretation: model-predicted descriptive sensitivity, not causal human-neuroscience evidence."
    )

    lines.extend(["", "## Discrete Stimulus Optimizer", ""])
    catalog_rows = tables.get("candidate_catalog", [])
    lines.append(
        f"- Candidate catalog: `{len(catalog_rows)}` generated, "
        f"`{len(evaluated_optimizer_rows)}` evaluated."
    )
    if objectives:
        lines.append(
            "- Objective version: "
            f"`{objectives.get('objective_version', 'unknown')}`; best candidate "
            f"`{objectives.get('best_candidate_id')}`."
        )

    lines.extend(["", "## Counterfactual Editing", ""])
    counterfactual_rows = tables.get("counterfactual_pairs", [])
    complete_counterfactuals = [
        row for row in counterfactual_rows if row.get("complete_pair") in {"True", True}
    ]
    lines.append(
        f"- Counterfactual pairs: `{len(complete_counterfactuals)}` complete of "
        f"`{len(counterfactual_rows)}` planned."
    )
    if complete_counterfactuals:
        top_counterfactual = max(
            complete_counterfactuals,
            key=lambda row: abs(_float_value(row.get("normalized_l2_delta"), 0.0)),
        )
        lines.append(
            "- Largest counterfactual response delta: "
            f"`{top_counterfactual.get('stimulus_id')}` "
            f"(normalized L2 `{_float_value(top_counterfactual.get('normalized_l2_delta'), 0.0):.6f}`)."
        )

    lines.extend(["", "## Focused Ultrasound Bridge", ""])
    fus_rows = tables.get("focused_ultrasound_comparisons", [])
    fus_complete = [
        row for row in fus_rows if row.get("complete_pair") in {"True", True}
    ]
    protocol_rows = tables.get("focused_ultrasound_protocols", [])
    lines.append(
        f"- Protocol proxies: `{len(protocol_rows)}` stimuli, "
        f"`{len(fus_complete)}` complete paired comparisons."
    )
    if fus_complete:
        top_fus = max(
            fus_complete,
            key=lambda row: abs(_float_value(row.get("normalized_l2_delta"), 0.0)),
        )
        lines.append(
            "- Largest focused-ultrasound proxy delta: "
            f"`{top_fus.get('stimulus_id')}` "
            f"(normalized L2 `{_float_value(top_fus.get('normalized_l2_delta'), 0.0):.6f}`)."
        )
    lines.append(
        "- Interpretation: synthetic protocol/provenance bridge only; no acoustic "
        "propagation, real sonication, clinical effect, or causal neuromodulation claim."
    )

    lines.extend(["", "## Latent Components", ""])
    latent_rows = tables.get("latent_components", [])
    loading_rows = tables.get("latent_loadings", [])
    if latent_rows:
        first = latent_rows[0]
        if first.get("status") == "insufficient_samples":
            lines.append(
                "- Component decomposition was not run: "
                f"{first.get('message', 'insufficient samples')}"
            )
        else:
            best_component = max(
                latent_rows,
                key=lambda row: float(row.get("variance_ratio", 0.0)),
            )
            lines.append(
                "- Components computed: "
                f"`{len(latent_rows)}`; top component "
                f"`{best_component.get('component_id')}` explains "
                f"{float(best_component.get('variance_ratio', 0.0)):.4f} "
                "of response variance."
            )
            lines.append(f"- Component loadings rows: `{len(loading_rows)}`.")
    else:
        lines.append("- No latent component table was generated.")

    lines.extend(["", "## BOLD5000 Real-Data Benchmark", ""])
    bold_rows = tables.get("bold5000_model_comparison", [])
    if bold_rows:
        best_bold = max(
            bold_rows,
            key=lambda row: _float_value(row.get("best_pearson_mean"), 0.0),
        )
        subjects = ", ".join(str(item) for item in metrics.get("subjects", []))
        rois = ", ".join(str(item) for item in metrics.get("rois", []))
        release = metrics.get("dataset_release_label", "BOLD5000 Release 1.0")
        lines.append(
            "- Best ROI/model: "
            f"`{best_bold.get('subject')} {best_bold.get('roi')}` with "
            f"`{best_bold.get('best_model')}` "
            f"(mean voxel Pearson r `{_float_value(best_bold.get('best_pearson_mean'), 0.0):.6f}`, "
            f"R2 `{_float_value(best_bold.get('best_r2'), 0.0):.6f}`, "
            f"exploratory uncorrected permutation p `{best_bold.get('permutation_p')}`)."
        )
        benchmark = metrics.get("bold5000_benchmark", {})
        if isinstance(benchmark, dict) and benchmark.get("max_statistic_p") is not None:
            lines.append(
                "- Grid-level max-statistic diagnostic: "
                f"p `{benchmark.get('max_statistic_p')}` over "
                f"`{benchmark.get('max_statistic_permutations')}` permutations. "
                "This is exploratory and is reported to avoid over-interpreting "
                "the selected best ROI/model."
            )
        lines.append(
            f"- ROI rows: `{len(bold_rows)}`; trial rows: "
            f"`{len(tables.get('bold5000_trials', []))}`."
        )
        lines.append(
            "- Run context: "
            f"`{release}` processed ROI vectors; subjects `{subjects}`; ROIs `{rois}`; "
            f"trial limit `{metrics.get('trial_limit')}`; trial offset "
            f"`{metrics.get('trial_offset', 0)}`; trial selection "
            f"`{metrics.get('trial_selection', 'first')}`; split strategy "
            f"`{metrics.get('split_strategy', 'random')}`; validation fraction "
            f"`{metrics.get('validation_fraction')}`; seed `{metrics.get('seed')}`; "
            f"permutations `{metrics.get('permutations')}`; bootstraps "
            f"`{metrics.get('bootstraps')}`."
        )
        lines.append(
            "- Scope: BOLD5000 Release 1.0 ROI responses plus stimulus filenames, "
            "source families, and labels. Release 2.0 is recommended by the dataset "
            "authors for new functional analyses and is not evaluated by this adapter. "
            "Raw pixel-image models are not part of this v1 run."
        )
    else:
        lines.append("- No BOLD5000 benchmark table was generated.")

    lines.extend(["", "## Next Experiments", ""])
    for item in next_experiments:
        lines.append(f"- **{item['title']}** (`{item['id']}`): {item['rationale']}")
    lines.append("")
    return "\n".join(lines)


def _html_report(markdown: str) -> str:
    body = html.escape(markdown)
    return (
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8"><title>Braindough Report</title>'
        "<style>body{font:16px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "max-width:960px;margin:40px auto;padding:0 20px}pre{background:#f6f8fa;"
        "padding:16px;overflow:auto}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}"
        "</style></head><body><pre>"
        f"{body}"
        "</pre></body></html>\n"
    )


def _write_pdf(
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    figures: list[Path],
    tables: dict[str, Any],
) -> None:
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = [
            "Braindough Executive Summary",
            "",
            f"Run: {manifest.get('run_id', 'unknown')}",
            f"Status: {manifest.get('status', 'unknown')}",
            f"Backend: {manifest.get('backend', {}).get('name', 'unknown')}",
            f"Experiment: {manifest.get('config', {}).get('experiment_id', 'unknown')}",
            f"Responses: {metrics.get('n_responses', 0)}",
            "",
            "Key findings",
            f"- Lesion comparisons: {len(tables.get('lesion_comparisons', []))}",
            f"- Optimizer candidates: {len(tables.get('candidate_catalog', []))}",
            f"- Counterfactual pairs: {len(tables.get('counterfactual_pairs', []))}",
            f"- BOLD5000 ROI rows: {len(tables.get('bold5000_model_comparison', []))}",
            "",
            "TRIBE/fake suites are model-predicted response summaries.",
            "BOLD5000 suites use Release 1.0 measured public ROI response matrices.",
            "Release 2.0 is recommended by BOLD5000 authors for new functional analyses.",
            "The BOLD5000 v1 benchmark uses metadata/labels, not raw pixels.",
            "Permutation p-values are exploratory and uncorrected.",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
        )
        pdf.savefig(fig)
        plt.close(fig)

        for figure_path in figures[:8]:
            if not figure_path.is_file():
                continue
            fig, ax = plt.subplots(figsize=(8.5, 6))
            ax.imshow(plt.imread(figure_path))
            ax.axis("off")
            ax.set_title(figure_path.name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def _read_responses(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


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
    for path in sorted(tables_dir.glob("*.jsonl")):
        tables[path.stem] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    for path in sorted(tables_dir.glob("*.json")):
        tables[path.stem] = _read_json(path)
    return tables


def _short_id(value: str) -> str:
    parts = value.split(":")
    return ":".join(parts[-2:]) if len(parts) > 2 else value


def _labelize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


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
