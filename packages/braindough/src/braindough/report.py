"""Human-readable report generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from braindough.analysis import similarity_matrix


def write_report(run_dir: str | Path) -> tuple[Path, Path]:
    """Write Markdown, HTML, and figures for a run directory."""

    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    metrics = _read_json(root / "metrics.json")
    next_experiments = _read_json(root / "next_experiments.json")
    responses = _read_responses(root / "outputs" / "responses.npz")
    figures = _write_figures(root, responses)
    markdown = _markdown_report(manifest, metrics, next_experiments, figures)
    md_path = root / "report.md"
    html_path = root / "report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_html_report(markdown), encoding="utf-8")
    return md_path, html_path


def _write_figures(root: Path, responses: dict[str, np.ndarray]) -> list[Path]:
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
    return figures


def _markdown_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    next_experiments: list[dict[str, str]],
    figures: list[Path],
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


def _read_responses(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _short_id(value: str) -> str:
    parts = value.split(":")
    return ":".join(parts[-2:]) if len(parts) > 2 else value
