from pathlib import Path

import pytest

from braindough.config import (
    discover_experiment_specs,
    is_absolute_local_path,
    load_experiment_spec,
)


def test_load_smoke_spec() -> None:
    spec = load_experiment_spec("experiments/smoke/fake_first_suite.yaml")

    assert spec.experiment_id == "smoke/fake-first-suite"
    assert spec.backend == "fake"
    assert "image_activation" in spec.suites
    assert spec.seed == 20260508


def test_invalid_spec_requires_suites(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: bad\ntitle: Bad\nbackend: fake\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required"):
        load_experiment_spec(path)


def test_discover_experiment_specs() -> None:
    specs = discover_experiment_specs()

    assert Path("experiments/smoke/fake_first_suite.yaml") in specs
    assert Path("experiments/local/tribe_v2_first_suite.yaml") in specs


def test_experiment_identity_redacts_absolute_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    checkpoint_path = tmp_path / "checkpoint.pt"
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "id: local/path-redaction",
                "title: Path redaction",
                "backend: fake",
                "suites:",
                "  - image_activation",
                "stimuli:",
                "  images:",
                f"    - {image_path}",
                "backend_config:",
                f"  checkpoint: {checkpoint_path}",
                "output:",
                f"  scratch: {tmp_path / 'scratch'}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_experiment_spec(spec_path)
    config = spec.to_dict()
    image_token = config["stimuli"]["images"][0]
    checkpoint_token = config["backend_config"]["checkpoint"]
    scratch_token = config["output"]["scratch"]

    assert image_token["local_path_name"] == "source.png"
    assert len(image_token["local_path_sha256"]) == 64
    assert checkpoint_token["local_path_name"] == "checkpoint.pt"
    assert len(checkpoint_token["local_path_sha256"]) == 64
    assert scratch_token["local_path_name"] == "scratch"
    assert len(scratch_token["local_path_sha256"]) == 64


def test_experiment_identity_redacts_windows_absolute_paths(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    windows_path = r"C:\Users\peyton\secret\source.png"
    spec_path.write_text(
        "\n".join(
            [
                "id: local/windows-path-redaction",
                "title: Windows path redaction",
                "backend: fake",
                "suites:",
                "  - image_activation",
                "stimuli:",
                "  images:",
                f"    - '{windows_path}'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_experiment_spec(spec_path).to_dict()
    token = config["stimuli"]["images"][0]

    assert is_absolute_local_path(windows_path)
    assert token["local_path_name"] == "source.png"
    assert len(token["local_path_sha256"]) == 64
    assert windows_path not in str(config)


def test_redacted_local_paths_keep_distinct_identity(tmp_path: Path) -> None:
    first = tmp_path / "first" / "model.pt"
    second = tmp_path / "second" / "model.pt"

    first_spec = _write_checkpoint_spec(tmp_path / "first.yaml", first)
    second_spec = _write_checkpoint_spec(tmp_path / "second.yaml", second)

    first_token = load_experiment_spec(first_spec).to_dict()["backend_config"][
        "checkpoint"
    ]
    second_token = load_experiment_spec(second_spec).to_dict()["backend_config"][
        "checkpoint"
    ]

    assert first_token["local_path_name"] == second_token["local_path_name"]
    assert first_token["local_path_sha256"] != second_token["local_path_sha256"]


def _write_checkpoint_spec(path: Path, checkpoint: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "id: local/path-identity",
                "title: Path identity",
                "backend: fake",
                "suites:",
                "  - image_activation",
                "backend_config:",
                f"  checkpoint: {checkpoint}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path
