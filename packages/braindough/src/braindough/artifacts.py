"""Artifact schema and serialization."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from braindough import __version__
from braindough.analysis import (
    build_derived_tables,
    build_objectives_summary,
    derived_metrics,
    next_experiment_suggestions,
    response_metrics,
)
from braindough.config import ExperimentSpec, is_absolute_local_path
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths, sha256_file, sha256_text

SCHEMA_VERSION = "braindough.artifact.v1"
_LOCAL_PATH_FRAGMENT_RE = re.compile(
    r"(?:file://)?/(?:Users|Volumes|tmp|private/tmp|var/folders)[^\n\r\t\"'<>]*"
)
_WINDOWS_PATH_FRAGMENT_RE = re.compile(
    r"(?:[a-zA-Z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+|\\[^\\/]+)[^\n\r\t\"'<>]*"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class RunArtifact:
    """Mutable artifact state before finalization."""

    run_id: str
    run_dir: Path
    spec: ExperimentSpec
    paths: BraindoughPaths
    started_at: str = field(default_factory=_now)
    status: str = "running"
    blocker: str | None = None

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.run_dir / "figures"

    @property
    def stimuli_dir(self) -> Path:
        return self.run_dir / "stimuli"

    def prepare(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.stimuli_dir.mkdir(parents=True, exist_ok=True)

    def write_config_lock(self) -> None:
        git_sha, dirty = _git_state()
        config = self.spec.to_dict()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": config,
            "experiment_config_sha256": _json_sha256(config),
            "workspace": {
                "git_sha": git_sha,
                "dirty": dirty,
                "worktree_id": self.paths.worktree_id,
            },
            "package": {"braindough_version": __version__},
        }
        _write_json(self.run_dir / "config.lock.json", payload)

    def write_events(self, events: list[dict[str, Any]]) -> None:
        with (self.run_dir / "events.ndjson").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(
                    json.dumps(_sanitize_local_paths(event), sort_keys=True) + "\n"
                )

    def write_responses(self, responses: dict[str, np.ndarray]) -> list[dict[str, Any]]:
        if responses:
            arrays: dict[str, Any] = dict(responses)
            np.savez_compressed(self.outputs_dir / "responses.npz", **arrays)
        else:
            np.savez_compressed(self.outputs_dir / "responses.npz")
        responses_sha256 = sha256_file(self.outputs_dir / "responses.npz")

        outputs: list[dict[str, Any]] = []
        with (self.outputs_dir / "responses.index.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for key, value in sorted(responses.items()):
                item = {
                    "id": key,
                    "path": "outputs/responses.npz",
                    "array_key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": responses_sha256,
                }
                handle.write(json.dumps(item, sort_keys=True) + "\n")
                outputs.append(
                    {
                        "id": key,
                        "path": "outputs/responses.npz",
                        "sha256": responses_sha256,
                        "media_type": "application/vnd.numpy.npz",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                )
        return outputs

    def write_metrics(
        self,
        backend_metrics: dict[str, Any],
        responses: dict[str, np.ndarray],
        stimuli: list[Stimulus],
    ) -> dict[str, Any]:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "backend": self.spec.backend,
            **_sanitize_local_paths(backend_metrics),
            **response_metrics(responses),
            **derived_metrics(stimuli, responses),
        }
        _write_json(self.run_dir / "metrics.json", metrics)
        _write_json(
            self.run_dir / "next_experiments.json", next_experiment_suggestions(metrics)
        )
        return metrics

    def write_tables(
        self, stimuli: list[Stimulus], responses: dict[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        tables_dir = self.outputs_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tables = build_derived_tables(stimuli, responses)
        outputs: list[dict[str, Any]] = []

        csv_table_names = [
            "stimuli",
            "response_metrics",
            "perturbation_comparisons",
            "latent_components",
            "latent_loadings",
        ]
        for table_name in csv_table_names:
            path = tables_dir / f"{table_name}.csv"
            rows = tables.get(table_name, [])
            _write_csv(path, rows)
            outputs.append(_table_output(self.run_dir, path, table_name, len(rows)))

        optimization_rows = tables.get("optimization_history", [])
        history_path = tables_dir / "optimization_history.jsonl"
        with history_path.open("w", encoding="utf-8") as handle:
            for row in optimization_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        outputs.append(
            _table_output(
                self.run_dir,
                history_path,
                "optimization_history",
                len(optimization_rows),
            )
        )

        objectives_path = tables_dir / "objectives.json"
        _write_json(objectives_path, build_objectives_summary(optimization_rows))
        outputs.append(
            _table_output(
                self.run_dir,
                objectives_path,
                "objectives",
                1 if optimization_rows else 0,
            )
        )
        return outputs

    def write_manifest(
        self,
        stimuli: list[Stimulus],
        outputs: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> None:
        git_sha, dirty = _git_state()
        config = self.spec.to_dict()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "blocker": _sanitize_local_paths(self.blocker),
            "created_at": self.started_at,
            "completed_at": _now(),
            "backend": {
                "name": self.spec.backend,
                "version": __version__,
                "model_ref": self.spec.backend_config.get("model_ref"),
                "model_sha256": self.spec.backend_config.get("model_sha256"),
            },
            "workspace": {
                "git_sha": git_sha,
                "dirty": dirty,
                "worktree_id": self.paths.worktree_id,
            },
            "config": {
                "experiment_id": self.spec.experiment_id,
                "config_sha256": _json_sha256(config),
                "seed": self.spec.seed,
            },
            "inputs": [
                stimulus.to_manifest_input(self.run_dir) for stimulus in stimuli
            ],
            "outputs": outputs,
            "metrics": metrics,
        }
        _write_json(self.run_dir / "manifest.json", payload)

    def write_checksums(self) -> None:
        checksum_path = self.run_dir / "checksums.sha256"
        rows: list[str] = []
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != checksum_path.name:
                rows.append(f"{sha256_file(path)}  {path.relative_to(self.run_dir)}")
        checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def validate_artifact(run_dir: str | Path) -> list[str]:
    """Return validation errors for an artifact directory."""

    root = Path(run_dir)
    required = [
        "manifest.json",
        "config.lock.json",
        "metrics.json",
        "events.ndjson",
        "checksums.sha256",
        "report.md",
        "report.html",
        "next_experiments.json",
    ]
    errors = [f"missing {item}" for item in required if not (root / item).is_file()]
    if errors:
        return errors

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest has wrong schema_version")
    if status not in {"completed", "skipped", "failed"}:
        errors.append("manifest status must be completed, skipped, or failed")
    errors.extend(_validate_required_mapping(manifest, "manifest", _MANIFEST_FIELDS))
    if status in {"skipped", "failed"} and not manifest.get("blocker"):
        errors.append("skipped or failed artifacts must include blocker")
    if status == "completed":
        errors.extend(
            [
                f"missing {item}"
                for item in ["outputs/responses.npz", "outputs/responses.index.jsonl"]
                if not (root / item).is_file()
            ]
        )
        if not manifest.get("outputs"):
            errors.append("completed artifacts must include outputs")
    for output in manifest.get("outputs", []):
        path = output.get("path")
        if not isinstance(path, str) or Path(path).is_absolute():
            errors.append(f"output path is not relative: {path}")
            continue
        output_path = root / path
        if not output_path.is_file():
            errors.append(f"output path is missing: {path}")
        expected_sha = output.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            errors.append(f"output missing sha256: {output.get('id')}")
        elif output_path.is_file() and sha256_file(output_path) != expected_sha:
            errors.append(f"output sha256 mismatch: {path}")
        if not output.get("media_type"):
            errors.append(f"output missing media_type: {output.get('id')}")
        if output_path.is_file() and _output_contains_absolute_path(output_path):
            errors.append(f"output contains absolute path metadata: {path}")
    for item in manifest.get("inputs", []):
        uri = item.get("uri")
        if not isinstance(uri, str) or Path(uri).is_absolute():
            errors.append(f"input uri is not relative: {uri}")
        if not item.get("media_type"):
            errors.append(f"input missing media_type: {item.get('id')}")
        sha = item.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            errors.append(f"input missing sha256: {item.get('id')}")
        if _contains_absolute_path(item):
            errors.append(f"input contains absolute path metadata: {item.get('id')}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("schema_version") != SCHEMA_VERSION:
        errors.append("metrics has wrong schema_version")
    if _contains_absolute_path(metrics):
        errors.append("metrics contains absolute path")
    if _text_contains_absolute_path(
        (root / "events.ndjson").read_text(encoding="utf-8")
    ):
        errors.append("events.ndjson contains absolute path")
    if _text_contains_absolute_path((root / "report.md").read_text(encoding="utf-8")):
        errors.append("report.md contains absolute path")
    if _text_contains_absolute_path((root / "report.html").read_text(encoding="utf-8")):
        errors.append("report.html contains absolute path")
    config_lock = json.loads((root / "config.lock.json").read_text(encoding="utf-8"))
    if config_lock.get("schema_version") != SCHEMA_VERSION:
        errors.append("config.lock has wrong schema_version")
    experiment_config = config_lock.get("experiment")
    if not isinstance(experiment_config, dict):
        errors.append("config.lock experiment must be a mapping")
    else:
        if _contains_absolute_path(experiment_config):
            errors.append("config.lock experiment contains absolute path")
        config_sha = _json_sha256(experiment_config)
        if config_lock.get("experiment_config_sha256") != config_sha:
            errors.append("config.lock experiment_config_sha256 mismatch")
        if manifest.get("config", {}).get("config_sha256") != config_sha:
            errors.append("manifest config_sha256 mismatch")
    checksums = (root / "checksums.sha256").read_text(encoding="utf-8")
    if "manifest.json" not in checksums:
        errors.append("checksums.sha256 does not include manifest.json")
    return errors


def create_fixture_artifact() -> Path:
    """Create a temporary minimal valid artifact for tooling-contract checks."""

    tmp = Path(tempfile.mkdtemp(prefix="braindough-artifact-"))
    outputs = tmp / "outputs"
    outputs.mkdir()
    (tmp / "figures").mkdir()
    fixture_config = {"experiment_id": "fixture"}
    fixture_config_sha = _json_sha256(fixture_config)
    response = np.zeros((1, 4), dtype=np.float32)
    np.savez_compressed(outputs / "responses.npz", fixture=response)
    response_sha = sha256_file(outputs / "responses.npz")
    (outputs / "responses.index.jsonl").write_text(
        json.dumps(
            {
                "id": "fixture",
                "path": "outputs/responses.npz",
                "array_key": "fixture",
                "shape": [1, 4],
                "dtype": "float32",
                "sha256": response_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": "fixture",
            "status": "completed",
            "blocker": None,
            "created_at": "2026-05-08T00:00:00+00:00",
            "completed_at": "2026-05-08T00:00:01+00:00",
            "backend": {"name": "fake", "version": "0.1.0"},
            "workspace": {"git_sha": None, "dirty": False, "worktree_id": "fixture"},
            "config": {
                "experiment_id": "fixture",
                "config_sha256": fixture_config_sha,
                "seed": 0,
            },
            "inputs": [
                {
                    "id": "fixture-input",
                    "kind": "video",
                    "media_type": "video/mp4",
                    "uri": "stimuli/fixture.mp4",
                    "sha256": "1" * 64,
                    "license": "generated-unlicense",
                    "metadata": {},
                }
            ],
            "outputs": [
                {
                    "id": "fixture",
                    "path": "outputs/responses.npz",
                    "sha256": response_sha,
                    "media_type": "application/vnd.numpy.npz",
                    "shape": [1, 4],
                    "dtype": "float32",
                }
            ],
            "metrics": {"schema_version": SCHEMA_VERSION, "n_responses": 1},
        },
    )
    _write_json(
        tmp / "config.lock.json",
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": fixture_config,
            "experiment_config_sha256": fixture_config_sha,
            "workspace": {"git_sha": None, "dirty": False, "worktree_id": "fixture"},
            "package": {"braindough_version": __version__},
        },
    )
    _write_json(
        tmp / "metrics.json", {"schema_version": SCHEMA_VERSION, "n_responses": 1}
    )
    _write_json(tmp / "next_experiments.json", [])
    (tmp / "events.ndjson").write_text("", encoding="utf-8")
    (tmp / "report.md").write_text("# Fixture\n", encoding="utf-8")
    (tmp / "report.html").write_text("<h1>Fixture</h1>\n", encoding="utf-8")
    checksum_rows = []
    for path in sorted(tmp.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            checksum_rows.append(f"{sha256_file(path)}  {path.relative_to(tmp)}")
    (tmp / "checksums.sha256").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )
    return tmp


def cleanup_fixture_artifact(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> str | int | float | bool:
    if isinstance(value, str | int | float | bool):
        return value
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True)


def _table_output(
    run_dir: Path, path: Path, table_name: str, row_count: int
) -> dict[str, Any]:
    return {
        "id": f"table:{table_name}",
        "path": str(path.relative_to(run_dir)),
        "sha256": sha256_file(path),
        "media_type": _table_media_type(path),
        "rows": row_count,
    }


def _table_media_type(path: Path) -> str:
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".jsonl":
        return "application/x-ndjson"
    if path.suffix == ".json":
        return "application/json"
    return "application/octet-stream"


def _json_sha256(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True))


_MANIFEST_FIELDS = [
    "schema_version",
    "run_id",
    "status",
    "created_at",
    "completed_at",
    "backend",
    "workspace",
    "config",
    "inputs",
    "outputs",
    "metrics",
]


def _validate_required_mapping(
    payload: dict[str, Any], label: str, fields: list[str]
) -> list[str]:
    return [f"{label} missing {field}" for field in fields if field not in payload]


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(child) for child in value)
    if isinstance(value, str):
        return is_absolute_local_path(value) or _text_contains_absolute_path(value)
    return False


def _sanitize_local_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_local_paths(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_local_paths(child) for child in value]
    if isinstance(value, tuple):
        return [_sanitize_local_paths(child) for child in value]
    if isinstance(value, Path):
        return _path_token(str(value))
    if isinstance(value, str):
        return _sanitize_local_path_string(value)
    return value


def _sanitize_local_path_string(value: str) -> str | dict[str, str]:
    if is_absolute_local_path(value):
        return _path_token(value)

    sanitized = _LOCAL_PATH_FRAGMENT_RE.sub(
        lambda match: _path_token_text(match.group(0)), value
    )
    return _WINDOWS_PATH_FRAGMENT_RE.sub(
        lambda match: _path_token_text(match.group(0)), sanitized
    )


def _path_token(path: str) -> dict[str, str]:
    stripped = path.removeprefix("file://").rstrip("/\\")
    normalized = stripped.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1] if normalized else "path"
    return {
        "local_path_name": name or "path",
        "local_path_sha256": sha256_text(normalized),
    }


def _path_token_text(path: str) -> str:
    token = _path_token(path)
    return f"[local-path:{token['local_path_name']}:{token['local_path_sha256'][:12]}]"


def _text_contains_absolute_path(value: str) -> bool:
    return bool(
        _LOCAL_PATH_FRAGMENT_RE.search(value) or _WINDOWS_PATH_FRAGMENT_RE.search(value)
    )


def _output_contains_absolute_path(path: Path) -> bool:
    if path.suffix == ".json":
        return _contains_absolute_path(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and _contains_absolute_path(json.loads(line)):
                return True
        return False
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if _contains_absolute_path(row):
                    return True
        return False
    return False


def _git_state() -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False
