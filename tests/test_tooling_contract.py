from pathlib import Path


def test_ci_uses_locked_mise_and_just_ci() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert 'install_args: "--locked"' in workflow
    assert "just ci" in workflow
    assert "BRAINDOUGH_BACKEND: fake" in workflow


def test_justfile_exposes_required_targets() -> None:
    justfile = Path("justfile").read_text(encoding="utf-8")
    required = [
        "bootstrap:",
        "fmt:",
        "fmt-check:",
        "lint:",
        "typecheck:",
        "test:",
        "check:",
        "ci:",
        "doctor:",
        "storage-init:",
        "storage-doctor:",
        "run-fake:",
        "run-fake-optimization:",
        "run-tribe:",
        "run-tribe-optimization:",
        "artifact-validate",
        "research-validate:",
        "report RUN_DIR:",
        "executive-summary RUN_DIRS='' OUTPUT_DIR='':",
    ]

    for target in required:
        assert target in justfile


def test_mise_experimental_lockfile_enabled() -> None:
    config = Path("mise.toml").read_text(encoding="utf-8")

    assert "experimental = true" in config
    assert "lockfile = true" in config
