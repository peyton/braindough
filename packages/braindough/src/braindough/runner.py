"""Experiment runner orchestration."""

from __future__ import annotations

from pathlib import Path

from braindough.artifacts import RunArtifact
from braindough.backends import get_backend
from braindough.config import load_experiment_spec
from braindough.report import write_report
from braindough.stimuli import generate_stimuli
from braindough.storage import BraindoughPaths, make_run_id, sha256_file


def run_experiment(spec_path: str | Path, home: str | Path | None = None) -> Path:
    """Run an experiment spec and return its artifact directory."""

    spec = load_experiment_spec(spec_path)
    paths = BraindoughPaths.discover(home=home)
    paths.init()
    run_id = make_run_id(spec.experiment_id, spec.backend)
    run_dir = paths.run_dir(run_id)
    artifact = RunArtifact(run_id=run_id, run_dir=run_dir, spec=spec, paths=paths)
    artifact.prepare()
    artifact.write_config_lock()

    stimuli = generate_stimuli(
        suites=spec.suites,
        output_dir=artifact.stimuli_dir,
        seed=spec.seed,
        config=spec.stimuli,
    )
    stimulus_events = [stimulus.to_event(run_dir) for stimulus in stimuli]
    backend = get_backend(spec.backend)
    result = backend.run(spec=spec, stimuli=stimuli, paths=paths, run_dir=run_dir)
    missing_statuses = missing_statuses_from_events(result.events)
    artifact.status = result.status
    artifact.blocker = result.blocker
    artifact.write_events(stimulus_events + result.events)
    outputs = (
        artifact.write_responses(result.responses)
        if result.status != "skipped" or result.responses
        else []
    )
    if result.status != "skipped" or result.responses:
        outputs.extend(
            artifact.write_tables(
                stimuli, result.responses, missing_statuses=missing_statuses
            )
        )
        outputs.extend(artifact.write_delta_arrays(stimuli, result.responses))
    metrics = artifact.write_metrics(
        result.metrics,
        result.responses,
        stimuli,
        missing_statuses=missing_statuses,
    )
    artifact.write_manifest(stimuli=stimuli, outputs=outputs, metrics=metrics)
    report_paths = write_report(run_dir)
    outputs.extend(artifact_report_outputs(run_dir, report_paths))
    artifact.write_manifest(stimuli=stimuli, outputs=outputs, metrics=metrics)
    artifact.write_checksums()
    return run_dir


def artifact_report_outputs(
    run_dir: Path, report_paths: tuple[Path, ...]
) -> list[dict[str, object]]:
    """Return manifest output rows for generated human-readable reports."""

    outputs: list[dict[str, object]] = []
    for path in report_paths:
        if not path.is_file():
            continue
        outputs.append(
            {
                "id": f"{_report_output_prefix(path)}:{path.name}",
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "media_type": _report_media_type(path),
            }
        )
    return outputs


def _report_output_prefix(path: Path) -> str:
    return "figure" if path.suffix == ".png" else "report"


def missing_statuses_from_events(events: list[dict[str, object]]) -> dict[str, str]:
    """Return per-stimulus missing response statuses inferred from backend events."""

    statuses: dict[str, str] = {}
    for event in events:
        if event.get("event") != "prediction_error":
            continue
        stimulus_id = event.get("stimulus_id")
        if isinstance(stimulus_id, str):
            statuses[stimulus_id] = "backend_error"
    return statuses


def _report_media_type(path: Path) -> str:
    if path.suffix == ".md":
        return "text/markdown"
    if path.suffix == ".html":
        return "text/html"
    if path.suffix == ".pdf":
        return "application/pdf"
    if path.suffix == ".png":
        return "image/png"
    return "application/octet-stream"
