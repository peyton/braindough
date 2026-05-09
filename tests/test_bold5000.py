import json
from pathlib import Path

import numpy as np

from braindough.artifacts import validate_artifact
from braindough.cli import main
from braindough.datasets.bold5000 import BOLD5000Dataset, create_fixture_dataset
from braindough.runner import run_experiment
from braindough.stimuli import generate_stimuli


def test_bold5000_fixture_doctor_and_stimuli(tmp_path: Path) -> None:
    dataset = create_fixture_dataset(tmp_path / "bold5000")

    doctor = dataset.doctor()
    stimuli = generate_stimuli(
        suites=("bold5000_roi_encoding",),
        output_dir=tmp_path / "stimuli",
        seed=1,
        config={
            "dataset_root": str(dataset.root),
            "subjects": ["CSI1"],
            "trial_limit": 5,
        },
    )

    assert doctor.ready
    assert len(stimuli) == 5
    assert all(stimulus.path.is_file() for stimulus in stimuli)
    assert {stimulus.metadata["source_family"] for stimulus in stimuli} == {
        "coco",
        "imagenet",
        "scene",
    }


def test_bold5000_fixture_run_writes_valid_benchmark_artifact(
    tmp_path: Path,
) -> None:
    dataset = create_fixture_dataset(tmp_path / "bold5000")
    spec = tmp_path / "bold5000.yaml"
    spec.write_text(
        "\n".join(
            [
                "id: test/bold5000-roi-encoding",
                "title: BOLD5000 fixture benchmark",
                "backend: bold5000-ridge",
                "seed: 11",
                "suites:",
                "  - bold5000_roi_encoding",
                "stimuli:",
                f"  dataset_root: {dataset.root}",
                "  subjects: [CSI1]",
                "  trial_limit: 12",
                "backend_config:",
                f"  dataset_root: {dataset.root}",
                "  subjects: [CSI1]",
                "  rois: [LHEarlyVis, LHPPA]",
                "  trial_limit: 12",
                "  validation_fraction: 0.25",
                "  hash_features: 8",
                "  alpha_grid: [0.1, 1.0]",
                "  permutations: 4",
                "  bootstraps: 8",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = run_experiment(spec, home=tmp_path / "home")

    assert not validate_artifact(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (run_dir / "outputs" / "tables" / "bold5000_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    responses = np.load(run_dir / "outputs" / "responses.npz")
    report = (run_dir / "report.md").read_text(encoding="utf-8")

    assert manifest["status"] == "completed"
    assert metrics["bold5000_benchmark"]["status"] == "completed"
    assert metrics["bold5000_benchmark"]["n_roi_results"] == 2
    assert metrics["dataset_release"] == "release-1.0"
    assert metrics["permutations"] == 4
    assert metrics["bootstraps"] == 8
    assert provenance["dataset_release"] == "release-1.0"
    assert provenance["terms"].endswith("/terms.html")
    assert provenance["downloads"]["BOLD5000_ROIs.zip"]["file_id"] == "12965447"
    assert responses.files
    assert (run_dir / "outputs" / "tables" / "bold5000_trials.csv").is_file()
    assert (run_dir / "outputs" / "tables" / "bold5000_roi_scores.csv").is_file()
    assert (run_dir / "outputs" / "tables" / "bold5000_model_comparison.csv").is_file()
    assert (run_dir / "figures" / "bold5000_roi_scores.png").is_file()
    assert "BOLD5000 Real-Data Benchmark" in report
    assert "Release 1.0" in report
    assert "exploratory uncorrected" in report
    assert "Release 2.0" in report


def test_cli_bold5000_doctor_with_fixture(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    dataset = create_fixture_dataset(tmp_path / "bold5000")

    assert main(["datasets", "bold5000", "doctor", "--root", str(dataset.root)]) == 0

    output = capsys.readouterr().out
    assert '"ready": true' in output


def test_cli_bold5000_doctor_missing_dataset_is_informational(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    missing_root = tmp_path / "missing"

    assert main(["datasets", "bold5000", "doctor", "--root", str(missing_root)]) == 0

    output = capsys.readouterr().out
    assert '"ready": false' in output
    assert "missing or invalid archives" in output


def test_bold5000_missing_dataset_stimulus_records_blocker(tmp_path: Path) -> None:
    dataset = BOLD5000Dataset(tmp_path / "missing")

    stimuli = generate_stimuli(
        suites=("bold5000_roi_encoding",),
        output_dir=tmp_path / "stimuli",
        seed=1,
        config={"dataset_root": str(dataset.root), "trial_limit": 4},
    )

    assert len(stimuli) == 1
    assert stimuli[0].kind == "dataset_missing"
    assert "missing or invalid archives" in str(stimuli[0].metadata["blocker"])
