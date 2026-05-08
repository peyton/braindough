import json
from pathlib import Path

import numpy as np
from PIL import Image

from braindough.artifacts import create_fixture_artifact, validate_artifact
from braindough.config import is_absolute_local_path
from braindough.runner import run_experiment
from braindough.storage import make_run_id


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
    assert (run_dir / "figures" / "perturbation_deltas.png").is_file()
    assert (run_dir / "figures" / "optimization_trace.png").is_file()
    assert "## Latent Components" in (run_dir / "report.md").read_text(encoding="utf-8")


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


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(child) for child in value)
    if isinstance(value, str):
        return is_absolute_local_path(value)
    return False
