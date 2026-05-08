import json
from pathlib import Path

import numpy as np
from PIL import Image

from braindough.artifacts import RunArtifact, create_fixture_artifact, validate_artifact
from braindough.config import ExperimentSpec, is_absolute_local_path
from braindough.report import write_report
from braindough.runner import run_experiment
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths, make_run_id


def test_fake_run_writes_valid_artifact(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_first_suite.yaml",
        home=tmp_path / "braindough-home",
    )

    assert not validate_artifact(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    responses = np.load(run_dir / "outputs" / "responses.npz")

    assert manifest["status"] == "completed"
    assert manifest["created_at"]
    assert manifest["completed_at"]
    assert manifest["workspace"]["worktree_id"]
    assert metrics["n_responses"] == len(manifest["inputs"])
    assert len(responses.files) == len(manifest["inputs"])
    assert all("sha256" in output for output in manifest["outputs"])
    assert all("media_type" in item for item in manifest["inputs"])
    assert not _contains_absolute_path(manifest["inputs"])
    assert (run_dir / "figures" / "response_similarity.png").is_file()
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "executive_summary.pdf").is_file()
    assert (run_dir / "next_experiments.json").is_file()
    assert (run_dir / "outputs" / "tables" / "stimuli.csv").is_file()
    assert (run_dir / "outputs" / "tables" / "response_metrics.csv").is_file()


def test_user_image_specs_write_path_neutral_config(tmp_path: Path) -> None:
    image_path = tmp_path / "external-input.png"
    Image.new("RGB", (32, 32), (20, 40, 80)).save(image_path)
    spec_path = tmp_path / "absolute-image-spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "id: local/absolute-image",
                "title: Absolute image path",
                "backend: fake",
                "seed: 7",
                "suites:",
                "  - image_activation",
                "stimuli:",
                "  images:",
                f"    - {image_path}",
                "backend_config:",
                "  vertices: 8",
                "  timesteps: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = run_experiment(spec_path, home=tmp_path / "braindough-home")

    assert not validate_artifact(run_dir)
    config_lock = json.loads((run_dir / "config.lock.json").read_text("utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert not _contains_absolute_path(config_lock["experiment"])
    image_token = config_lock["experiment"]["stimuli"]["images"][0]
    assert image_token["local_path_name"] == "external-input.png"
    assert len(image_token["local_path_sha256"]) == 64
    assert (
        manifest["config"]["config_sha256"] == config_lock["experiment_config_sha256"]
    )


def test_fake_backend_is_deterministic(tmp_path: Path) -> None:
    first = run_experiment(
        "experiments/smoke/fake_first_suite.yaml",
        home=tmp_path / "first",
    )
    second = run_experiment(
        "experiments/smoke/fake_first_suite.yaml",
        home=tmp_path / "second",
    )

    first_npz = np.load(first / "outputs" / "responses.npz")
    second_npz = np.load(second / "outputs" / "responses.npz")
    key = sorted(first_npz.files)[0]

    assert np.array_equal(first_npz[key], second_npz[key])


def test_fake_perturbation_optimization_run_writes_tables(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_perturbation_optimization.yaml",
        home=tmp_path / "braindough-home",
    )

    assert not validate_artifact(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    table_dir = run_dir / "outputs" / "tables"
    history = (table_dir / "optimization_history.jsonl").read_text(encoding="utf-8")
    objectives = json.loads((table_dir / "objectives.json").read_text("utf-8"))

    assert manifest["status"] == "completed"
    assert any(output["id"] == "table:objectives" for output in manifest["outputs"])
    assert metrics["n_optimizer_candidates"] == 12
    assert metrics["n_perturbation_comparisons"] > 0
    assert metrics["latent_components"]["status"] == "computed"
    assert len(history.splitlines()) == 12
    assert objectives["stopping_reason"] == "candidate_budget_exhausted"
    assert objectives["objective_version"] == "discrete_optimizer.v2"
    assert (table_dir / "candidate_catalog.jsonl").is_file()
    assert (table_dir / "lesion_comparisons.csv").is_file()
    assert (table_dir / "counterfactual_pairs.csv").is_file()
    assert (run_dir / "outputs" / "deltas.npz").is_file()
    assert (run_dir / "executive_summary.pdf").is_file()
    assert (run_dir / "figures" / "perturbation_deltas.png").is_file()
    assert (run_dir / "figures" / "optimization_trace.png").is_file()
    assert "## Latent Components" in (run_dir / "report.md").read_text(encoding="utf-8")


def test_focused_fake_experiment_runs_write_suite_artifacts(tmp_path: Path) -> None:
    specs = [
        (
            "experiments/smoke/fake_virtual_lesion_lab.yaml",
            "lesion_comparisons.csv",
            "lesion_scoreboard.png",
        ),
        (
            "experiments/smoke/fake_discrete_stimulus_optimizer.yaml",
            "candidate_catalog.jsonl",
            "optimization_score_components.png",
        ),
        (
            "experiments/smoke/fake_counterfactual_editing_workbench.yaml",
            "counterfactual_pairs.csv",
            "counterfactual_tradeoff.png",
        ),
    ]

    for spec, table, figure in specs:
        run_dir = run_experiment(spec, home=tmp_path / spec.replace("/", "-"))

        assert not validate_artifact(run_dir)
        assert (run_dir / "outputs" / "tables" / table).is_file()
        assert (run_dir / "figures" / figure).is_file()
        assert (run_dir / "executive_summary.pdf").is_file()


def test_validate_requires_suite_specific_outputs(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_virtual_lesion_lab.yaml",
        home=tmp_path / "braindough-home",
    )
    table_path = run_dir / "outputs" / "tables" / "lesion_manifest.csv"
    table_path.unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = [
        output
        for output in manifest["outputs"]
        if output.get("path") != "outputs/tables/lesion_manifest.csv"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_artifact(run_dir)

    assert "missing outputs/tables/lesion_manifest.csv" in errors
    assert (
        "completed artifacts must include outputs/tables/lesion_manifest.csv output"
        in errors
    )


def test_validate_requires_suite_specific_figures_in_manifest(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_virtual_lesion_lab.yaml",
        home=tmp_path / "braindough-home",
    )
    figure_path = run_dir / "figures" / "lesion_scoreboard.png"
    figure_path.unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any(
        output.get("path") == "figures/lesion_scoreboard.png"
        and output.get("media_type") == "image/png"
        for output in manifest["outputs"]
    )
    manifest["outputs"] = [
        output
        for output in manifest["outputs"]
        if output.get("path") != "figures/lesion_scoreboard.png"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_artifact(run_dir)

    assert "missing figures/lesion_scoreboard.png" in errors
    assert (
        "completed artifacts must include figures/lesion_scoreboard.png output"
        in errors
    )


def test_validate_requires_shared_figures_and_optimizer_trace(
    tmp_path: Path,
) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_discrete_stimulus_optimizer.yaml",
        home=tmp_path / "braindough-home",
    )
    for relative_path in [
        "figures/response_similarity.png",
        "figures/mean_abs_activation.png",
        "figures/optimization_trace.png",
    ]:
        (run_dir / relative_path).unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    removed = {
        "figures/response_similarity.png",
        "figures/mean_abs_activation.png",
        "figures/optimization_trace.png",
    }
    manifest["outputs"] = [
        output for output in manifest["outputs"] if output.get("path") not in removed
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_artifact(run_dir)

    assert "missing figures/response_similarity.png" in errors
    assert "missing figures/mean_abs_activation.png" in errors
    assert "missing figures/optimization_trace.png" in errors
    assert (
        "completed artifacts must include figures/optimization_trace.png output"
        in errors
    )


def test_validate_verifies_checksum_coverage(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_counterfactual_editing_workbench.yaml",
        home=tmp_path / "braindough-home",
    )
    checksums = run_dir / "checksums.sha256"
    checksums.write_text(
        "\n".join(
            line
            for line in checksums.read_text(encoding="utf-8").splitlines()
            if "executive_summary.pdf" not in line
        )
        + "\n",
        encoding="utf-8",
    )

    errors = validate_artifact(run_dir)

    assert "checksums.sha256 missing executive_summary.pdf" in errors


def test_delta_arrays_write_empty_index_for_incomplete_pair_suite(
    tmp_path: Path,
) -> None:
    paths = BraindoughPaths.discover(home=tmp_path / "home")
    paths.init()
    spec = ExperimentSpec(
        experiment_id="test/partial-lesion",
        title="Partial lesion",
        backend="fake",
        seed=1,
        suites=("virtual_lesion_lab",),
    )
    artifact = RunArtifact(
        run_id="partial-lesion",
        run_dir=tmp_path / "partial-lesion",
        spec=spec,
        paths=paths,
    )
    artifact.prepare()
    parent = _stimulus(
        "virtual_lesion_lab:base_00:baseline",
        "virtual_lesion_lab",
        {"role": "lesion_source"},
    )
    child = _stimulus(
        "virtual_lesion_lab:base_00:low_contrast",
        "virtual_lesion_lab",
        {"parent_id": parent.stimulus_id, "lesion_type": "low_contrast"},
    )

    outputs = artifact.write_delta_arrays([parent, child], {})
    deltas = np.load(artifact.outputs_dir / "deltas.npz")

    assert [output["path"] for output in outputs] == [
        "outputs/deltas.npz",
        "outputs/deltas.index.jsonl",
    ]
    assert deltas.files == []
    assert (artifact.outputs_dir / "deltas.index.jsonl").read_text("utf-8") == ""


def test_report_handles_incomplete_perturbation_rows(tmp_path: Path) -> None:
    run_dir = run_experiment(
        "experiments/smoke/fake_virtual_lesion_lab.yaml",
        home=tmp_path / "braindough-home",
    )
    table_path = run_dir / "outputs" / "tables" / "perturbation_comparisons.csv"
    text = table_path.read_text(encoding="utf-8")
    header, first_row, *rest = text.splitlines()
    values = first_row.split(",")
    fields = header.split(",")
    row = dict(zip(fields, values, strict=True))
    row["complete_pair"] = "False"
    row["mean_abs_delta"] = ""
    row["missing_reason"] = "missing_child_response"
    table_path.write_text(
        "\n".join(
            [
                header,
                ",".join(row.get(field, "") for field in fields),
                *rest,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    paths = write_report(run_dir)

    assert paths[2].is_file()


def test_zero_strength_fake_lesion_is_parent_like(tmp_path: Path) -> None:
    spec_path = tmp_path / "zero-strength.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "id: smoke/zero-strength-lesion",
                "title: Zero strength lesion",
                "backend: fake",
                "seed: 7",
                "suites:",
                "  - virtual_lesion_lab",
                "stimuli:",
                "  virtual_lesion_base_count: 1",
                "  virtual_lesion_types: [low_contrast]",
                "  lesion_strengths: [0.0]",
                "backend_config:",
                "  vertices: 8",
                "  timesteps: 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = run_experiment(spec_path, home=tmp_path / "braindough-home")
    responses = np.load(run_dir / "outputs" / "responses.npz")
    baseline_id = next(key for key in responses.files if key.endswith(":baseline"))
    lesion_id = next(key for key in responses.files if key.endswith(":low_contrast"))

    assert np.array_equal(responses[baseline_id], responses[lesion_id])


def test_validate_rejects_absolute_paths_in_table_outputs(tmp_path: Path) -> None:
    path = create_fixture_artifact()
    table_dir = path / "outputs" / "tables"
    table_dir.mkdir(parents=True)
    table_path = table_dir / "bad.csv"
    table_path.write_text("path\n/tmp/secret.png\n", encoding="utf-8")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"].append(
        {
            "id": "table:bad",
            "path": "outputs/tables/bad.csv",
            "sha256": "0" * 64,
            "media_type": "text/csv",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = validate_artifact(path)
    assert "output contains absolute path metadata: outputs/tables/bad.csv" in errors


def test_validate_rejects_absolute_paths_in_metrics_events_and_reports() -> None:
    path = create_fixture_artifact()
    (path / "metrics.json").write_text(
        json.dumps({"schema_version": "braindough.artifact.v1", "note": "/tmp/secret"}),
        encoding="utf-8",
    )
    (path / "events.ndjson").write_text(
        json.dumps({"event": "bad", "path": "/Users/peyton/secret"}) + "\n",
        encoding="utf-8",
    )
    (path / "report.md").write_text("bad /Volumes/secret\n", encoding="utf-8")
    (path / "report.html").write_text(
        "bad C:\\Users\\peyton\\secret\n", encoding="utf-8"
    )

    errors = validate_artifact(path)
    assert "metrics contains absolute path" in errors
    assert "events.ndjson contains absolute path" in errors
    assert "report.md contains absolute path" in errors
    assert "report.html contains absolute path" in errors


def test_validate_rejects_incomplete_completed_artifact() -> None:
    path = create_fixture_artifact()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["created_at"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert "manifest missing created_at" in validate_artifact(path)


def test_validate_rejects_absolute_paths_in_config_lock(tmp_path: Path) -> None:
    path = create_fixture_artifact()
    config_path = path / "config.lock.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"]["stimuli"] = {"images": [str(tmp_path / "input.png")]}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert "config.lock experiment contains absolute path" in validate_artifact(path)


def test_validate_rejects_windows_absolute_paths_in_config_lock() -> None:
    path = create_fixture_artifact()
    config_path = path / "config.lock.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["experiment"]["stimuli"] = {"images": [r"C:\Users\peyton\secret\input.png"]}
    config_path.write_text(json.dumps(config), encoding="utf-8")

    assert "config.lock experiment contains absolute path" in validate_artifact(path)


def test_run_ids_do_not_collide_within_same_second() -> None:
    ids = {make_run_id("smoke/fake-first-suite", "fake") for _ in range(25)}

    assert len(ids) == 25


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


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(child) for child in value)
    if isinstance(value, str):
        return is_absolute_local_path(value)
    return False
