"""Experiment runner orchestration."""

from __future__ import annotations

from pathlib import Path

from braindough.artifacts import RunArtifact
from braindough.backends import get_backend
from braindough.config import load_experiment_spec
from braindough.report import write_report
from braindough.stimuli import generate_stimuli
from braindough.storage import BraindoughPaths, make_run_id


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
    artifact.status = result.status
    artifact.blocker = result.blocker
    artifact.write_events(stimulus_events + result.events)
    outputs = (
        artifact.write_responses(result.responses)
        if result.status != "skipped" or result.responses
        else []
    )
    if result.status != "skipped" or result.responses:
        outputs.extend(artifact.write_tables(stimuli, result.responses))
    metrics = artifact.write_metrics(result.metrics, result.responses, stimuli)
    artifact.write_manifest(stimuli=stimuli, outputs=outputs, metrics=metrics)
    write_report(run_dir)
    artifact.write_checksums()
    return run_dir
