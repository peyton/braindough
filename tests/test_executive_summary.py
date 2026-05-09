import json
from pathlib import Path
from typing import Any

from braindough.cli import main
from braindough.config import is_absolute_local_path
from braindough.executive_summary import (
    SUMMARY_SCHEMA_VERSION,
    discover_latest_run_dirs,
    write_executive_summary,
)


def test_discover_latest_run_dirs_prefers_latest_target_backend(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    older = _write_run(
        home,
        "20260508T010000Z-smoke-fake-perturbation-optimization-fake-old",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
        completed_at="2026-05-08T01:00:00Z",
    )
    newer = _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake-new",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
        completed_at="2026-05-08T02:00:00Z",
    )
    tribe = _write_run(
        home,
        "20260508T030000Z-local-tribe-v2-perturbation-optimization-tribe-v2",
        backend="tribe-v2",
        experiment_id="local/tribe-v2-perturbation-optimization",
        completed_at="2026-05-08T03:00:00Z",
        n_stimuli=4,
        n_responses=1,
    )

    discovered = discover_latest_run_dirs(home=home)

    assert older not in discovered
    assert discovered == [newer, tribe]


def test_discover_latest_run_dirs_includes_bold5000_runs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bold = _write_run(
        home,
        "20260509T010000Z-local-bold5000-roi-encoding-bold5000-ridge",
        backend="bold5000-ridge",
        experiment_id="local/bold5000-roi-encoding",
        n_stimuli=384,
        n_responses=6,
    )

    discovered = discover_latest_run_dirs(home=home)

    assert discovered == [bold]


def test_write_executive_summary_outputs_pdf_json_sources_and_figures(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    fake = _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
    )
    tribe = _write_run(
        home,
        "20260508T030000Z-local-tribe-v2-perturbation-optimization-tribe-v2",
        backend="tribe-v2",
        experiment_id="local/tribe-v2-perturbation-optimization",
        n_stimuli=4,
        n_responses=1,
    )
    output = tmp_path / "summary"

    paths = write_executive_summary(
        run_dirs=[fake, tribe], output_dir=output, home=home
    )

    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert len(summary["runs"]) == 2
    assert any("TRIBE" in item["title"] for item in summary["key_findings"])
    assert not _contains_absolute_local_path(summary)
    assert paths["pdf"].read_bytes().startswith(b"%PDF")
    assert paths["pdf"].read_bytes().count(b"/Type /Page") >= 3
    assert (
        paths["markdown"]
        .read_text(encoding="utf-8")
        .startswith("# Braindough Executive Summary")
    )
    for chart in summary["charts"]:
        assert (output / chart["path"]).is_file()
    source_ids = {source["id"] for source in summary["sources"]}
    assert "tribe_v2_model_card" in source_ids
    assert "natural_scenes_dataset" in source_ids


def test_write_executive_summary_discovers_runs_when_not_explicit(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
    )

    paths = write_executive_summary(output_dir=tmp_path / "summary", home=home)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    assert [run["backend"] for run in summary["runs"]] == ["fake"]
    findings = " ".join(item["claim"] for item in summary["key_findings"])
    assert "The local TRIBE perturbation/optimization run produced" not in findings
    assert "No tribe-v2 perturbation/optimization run was loaded" in findings


def test_write_executive_summary_handles_bold5000_only_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_run(
        home,
        "20260509T010000Z-local-bold5000-roi-encoding-bold5000-ridge",
        backend="bold5000-ridge",
        experiment_id="local/bold5000-roi-encoding",
        n_stimuli=384,
        n_responses=6,
    )

    paths = write_executive_summary(output_dir=tmp_path / "summary", home=home)
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert [run["backend"] for run in summary["runs"]] == ["bold5000-ridge"]
    assert "BOLD5000 Release 1.0" in summary["scope"]
    assert "Release 2.0" in markdown
    assert "exploratory and uncorrected" in markdown
    chart_paths = {chart["path"] for chart in summary["charts"]}
    assert "figures/bold5000_roi_scores.png" in chart_paths
    assert "figures/bold5000_model_comparison.png" in chart_paths
    source_ids = {source["id"] for source in summary["sources"]}
    assert "bold5000_download" in source_ids
    assert "bold5000_terms" in source_ids


def test_write_executive_summary_finds_research_metadata_outside_cwd(
    tmp_path: Path, monkeypatch: Any
) -> None:
    home = tmp_path / "home"
    run = _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
    )
    monkeypatch.chdir(tmp_path)

    paths = write_executive_summary(
        run_dirs=[run], output_dir=tmp_path / "summary", home=home
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    source_ids = {source["id"] for source in summary["sources"]}
    assert "research_capture_executive_summary_deep_research" in source_ids


def test_cli_executive_summary_with_explicit_run_dir(
    tmp_path: Path, capsys: Any
) -> None:
    home = tmp_path / "home"
    run = _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
    )
    output_dir = tmp_path / "summary"

    assert (
        main(
            [
                "executive-summary",
                "--run-dir",
                str(run),
                "--output-dir",
                str(output_dir),
                "--home",
                str(home),
            ]
        )
        == 0
    )

    output = json.loads(capsys.readouterr().out)
    assert Path(output["pdf"]).is_file()
    assert Path(output["summary_json"]).is_file()


def test_executive_summary_handles_skipped_optimizer_rows(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _write_run(
        home,
        "20260508T020000Z-smoke-fake-perturbation-optimization-fake",
        backend="fake",
        experiment_id="smoke/fake-perturbation-optimization",
        include_skipped_optimizer_row=True,
    )

    paths = write_executive_summary(
        run_dirs=[run], output_dir=tmp_path / "summary", home=home
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    optimizer_chart = next(
        chart
        for chart in summary["charts"]
        if chart["path"] == "figures/optimizer_trace.png"
    )
    assert (tmp_path / "summary" / optimizer_chart["path"]).is_file()


def test_executive_summary_handles_backend_error_optimizer_rows(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    run = _write_run(
        home,
        "20260508T030000Z-local-tribe-v2-perturbation-optimization-tribe-v2",
        backend="tribe-v2",
        experiment_id="local/tribe-v2-perturbation-optimization",
        include_backend_error_optimizer_row=True,
    )

    paths = write_executive_summary(
        run_dirs=[run], output_dir=tmp_path / "summary", home=home
    )
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

    optimizer_chart = next(
        chart
        for chart in summary["charts"]
        if chart["path"] == "figures/optimizer_trace.png"
    )
    assert (tmp_path / "summary" / optimizer_chart["path"]).is_file()


def _write_run(
    home: Path,
    run_id: str,
    *,
    backend: str,
    experiment_id: str,
    completed_at: str = "2026-05-08T02:00:00Z",
    n_stimuli: int = 6,
    n_responses: int = 6,
    include_skipped_optimizer_row: bool = False,
    include_backend_error_optimizer_row: bool = False,
) -> Path:
    run_dir = home / "runs" / "2026" / "05" / run_id
    table_dir = run_dir / "outputs" / "tables"
    table_dir.mkdir(parents=True)
    suite_responses = max(1, n_responses // 2)
    suite_summary = {
        "latent_network_ica_explorer": {
            "stimuli": max(1, n_stimuli // 2),
            "responses": min(suite_responses, n_responses),
        },
        "discrete_stimulus_optimizer": {
            "stimuli": max(1, n_stimuli - max(1, n_stimuli // 2)),
            "responses": max(0, n_responses - min(suite_responses, n_responses)),
        },
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "created_at": "2026-05-08T01:00:00Z",
        "completed_at": completed_at,
        "config": {"experiment_id": experiment_id},
        "backend": {"name": backend},
        "inputs": [{"id": f"stimulus:{index}"} for index in range(n_stimuli)],
        "outputs": [],
        "blocker": None,
    }
    metrics = {
        "schema_version": "braindough.artifact.v1",
        "backend": backend,
        "n_stimuli": n_stimuli,
        "n_responses": n_responses,
        "suite_summary": suite_summary,
        "mean_abs_activation": 0.12,
        "mean_abs_activation_by_stimulus": {
            "latent_network_ica_explorer:base": 0.12,
            "discrete_stimulus_optimizer:candidate_00": 0.18,
        },
        "optimization": {
            "best_candidate_id": "discrete_stimulus_optimizer:candidate_00",
            "best_score": 0.18,
            "best_mean_abs_activation": 0.18,
            "n_candidates": max(1, n_responses // 2),
            "objective": "mean_abs_activation_minus_similarity_penalty",
            "stopping_reason": (
                "prediction_budget_reached"
                if backend == "tribe-v2"
                else "candidate_budget_exhausted"
            ),
        },
        "latent_components": {
            "status": "insufficient_samples" if backend == "tribe-v2" else "computed",
            "n_components": 0 if backend == "tribe-v2" else 2,
        },
        "n_optimizer_candidates": max(1, n_responses // 2),
        "n_perturbation_comparisons": 0 if backend == "tribe-v2" else 2,
        "top_activation_stimuli": [
            {
                "stimulus_id": "discrete_stimulus_optimizer:candidate_00",
                "mean_abs_activation": 0.18,
            }
        ],
        "attempted_predictions": n_responses if backend == "tribe-v2" else None,
        "max_predictions": n_responses if backend == "tribe-v2" else None,
    }
    if backend == "bold5000-ridge":
        metrics.update(
            {
                "suite_summary": {
                    "bold5000_roi_encoding": {
                        "stimuli": n_stimuli,
                        "responses": n_responses,
                    }
                },
                "subjects": ["CSI1"],
                "rois": ["RHEarlyVis", "LHEarlyVis"],
                "tr": "TR1",
                "trial_limit": n_stimuli,
                "trial_offset": 0,
                "validation_fraction": 0.25,
                "seed": 20260509,
                "permutations": 16,
                "bootstraps": 64,
                "dataset_release": "release-1.0",
                "dataset_release_label": "BOLD5000 Release 1.0",
                "p_value_note": (
                    "Permutation p-values are exploratory and uncorrected for "
                    "multiple ROI/model comparisons."
                ),
                "source_caveat": (
                    "This adapter uses BOLD5000 Release 1.0 processed ROI "
                    "vectors and stimulus name/label metadata."
                ),
                "bold5000_benchmark": {
                    "status": "completed",
                    "n_roi_results": 2,
                    "best_subject": "CSI1",
                    "best_roi": "RHEarlyVis",
                    "best_model": "source_family",
                    "best_pearson_mean": 0.031,
                    "best_r2": -0.02,
                    "mean_improvement_over_mean": 0.01,
                    "n_nominally_significant": 0,
                },
                "mean_abs_activation_by_stimulus": {
                    "bold5000_roi_encoding:CSI1:RHEarlyVis:source_family": 0.03
                },
                "n_optimizer_candidates": 0,
                "n_perturbation_comparisons": 0,
            }
        )
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if backend == "bold5000-ridge":
        (table_dir / "bold5000_model_comparison.csv").write_text(
            "\n".join(
                [
                    "subject,roi,tr,best_model,best_alpha,best_pearson_mean,best_r2,mean_baseline_pearson,improvement_over_mean,bootstrap_low,bootstrap_high,permutation_p",
                    "CSI1,RHEarlyVis,TR1,source_family,1.0,0.031,-0.02,0.0,0.01,-0.01,0.06,0.25",
                    "CSI1,LHEarlyVis,TR1,token_hash,10.0,0.02,-0.03,0.0,0.005,-0.02,0.05,0.50",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (table_dir / "bold5000_roi_scores.csv").write_text(
            "\n".join(
                [
                    "subject,roi,tr,model,alpha,feature_count,pearson_mean,pearson_median,r2,bootstrap_low,bootstrap_high,permutation_p",
                    "CSI1,RHEarlyVis,TR1,source_family,1.0,4,0.031,0.02,-0.02,-0.01,0.06,0.25",
                    "CSI1,RHEarlyVis,TR1,token_hash,10.0,16,0.01,0.0,-0.04,-0.02,0.03,0.75",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    if include_backend_error_optimizer_row:
        second_row = {
            "step": 1,
            "stimulus_id": "discrete_stimulus_optimizer:candidate_01",
            "status": "backend_error",
            "objective_score": "",
            "best_score_so_far": "",
        }
    elif include_skipped_optimizer_row:
        second_row = {
            "step": 1,
            "stimulus_id": "discrete_stimulus_optimizer:candidate_01",
            "status": "skipped_prediction_budget",
            "objective_score": "",
            "best_score_so_far": "",
        }
    else:
        second_row = {
            "step": 1,
            "stimulus_id": "discrete_stimulus_optimizer:candidate_01",
            "status": "evaluated",
            "objective_score": 0.14,
            "best_score_so_far": 0.18,
        }
    history_rows = [
        {
            "step": 0,
            "stimulus_id": "discrete_stimulus_optimizer:candidate_00",
            "status": "evaluated",
            "objective_score": 0.18,
            "best_score_so_far": 0.18,
        },
        second_row,
    ]
    (table_dir / "optimization_history.jsonl").write_text(
        "\n".join(json.dumps(row) for row in history_rows) + "\n", encoding="utf-8"
    )
    (table_dir / "objectives.json").write_text(
        json.dumps(metrics["optimization"]), encoding="utf-8"
    )
    return run_dir


def _contains_absolute_local_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_local_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_local_path(child) for child in value)
    if isinstance(value, str):
        return is_absolute_local_path(value)
    return False
