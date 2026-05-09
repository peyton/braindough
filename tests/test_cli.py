from pathlib import Path

from braindough.cli import main


def test_cli_experiments_list(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["experiments", "list"]) == 0

    output = capsys.readouterr().out
    assert "smoke/fake-first-suite" in output
    assert "smoke/fake-perturbation-optimization" in output
    assert "smoke/fake-virtual-lesion-lab" in output
    assert "smoke/fake-discrete-stimulus-optimizer" in output
    assert "smoke/fake-counterfactual-editing-workbench" in output
    assert "local/tribe-v2-first-suite" in output
    assert "local/tribe-v2-perturbation-optimization" in output
    assert "local/tribe-v2-virtual-lesion-lab" in output
    assert "local/tribe-v2-discrete-stimulus-optimizer" in output
    assert "local/tribe-v2-counterfactual-editing-workbench" in output
    assert "local/bold5000-roi-encoding" in output


def test_cli_validate_fixture(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["validate", "--fixture"]) == 0

    output = capsys.readouterr().out
    assert '"valid": true' in output


def test_cli_run_fake(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("BRAINDOUGH_HOME", str(tmp_path / "home"))

    assert main(["run", "experiments/smoke/fake_first_suite.yaml"]) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
