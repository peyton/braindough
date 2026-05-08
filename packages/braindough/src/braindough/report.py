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
            "",
            "These are model-predicted response summaries for descriptive",
            "sensitivity analysis. They are not human measurements or causal claims.",
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
    objectives_path = tables_dir / "objectives.json"
    if objectives_path.is_file():
        tables["objectives"] = _read_json(objectives_path)
    return tables


def _short_id(value: str) -> str:
    parts = value.split(":")
    return ":".join(parts[-2:]) if len(parts) > 2 else value


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
