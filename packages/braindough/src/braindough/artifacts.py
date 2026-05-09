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
    build_delta_arrays,
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
_NON_FILE_URL_RE = re.compile(r"\b(?!file://)[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")


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
        index_path = self.outputs_dir / "responses.index.jsonl"
        with index_path.open("w", encoding="utf-8") as handle:
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
        outputs.append(
            {
                "id": "responses:index",
                "path": str(index_path.relative_to(self.run_dir)),
                "sha256": sha256_file(index_path),
                "media_type": "application/x-ndjson",
                "rows": len(responses),
            }
        )
        return outputs

    def write_metrics(
        self,
        backend_metrics: dict[str, Any],
        responses: dict[str, np.ndarray],
        stimuli: list[Stimulus],
        missing_statuses: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        metrics = {
            "schema_version": SCHEMA_VERSION,
            "backend": self.spec.backend,
            **_sanitize_local_paths(backend_metrics),
            **response_metrics(responses),
            **derived_metrics(stimuli, responses, missing_statuses=missing_statuses),
        }
        _write_json(self.run_dir / "metrics.json", metrics)
        _write_json(
            self.run_dir / "next_experiments.json", next_experiment_suggestions(metrics)
        )
        return metrics

    def write_tables(
        self,
        stimuli: list[Stimulus],
        responses: dict[str, np.ndarray],
        missing_statuses: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        tables_dir = self.outputs_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        tables = build_derived_tables(
            stimuli, responses, missing_statuses=missing_statuses
        )
        outputs: list[dict[str, Any]] = []

        csv_table_names = [
            "stimuli",
            "response_metrics",
            "perturbation_comparisons",
            "lesion_manifest",
            "lesion_comparisons",
            "lesion_roi_summary",
            "top_delta_vertices",
            "latent_components",
            "latent_loadings",
            "counterfactual_pairs",
        ]
        for table_name in csv_table_names:
            path = tables_dir / f"{table_name}.csv"
            rows = tables.get(table_name, [])
            _write_csv(path, rows)
            outputs.append(_table_output(self.run_dir, path, table_name, len(rows)))

        jsonl_table_names = [
            "candidate_catalog",
            "optimization_history",
            "counterfactual_edits",
        ]
        for table_name in jsonl_table_names:
            rows = tables.get(table_name, [])
            path = tables_dir / f"{table_name}.jsonl"
            _write_jsonl(path, rows)
            outputs.append(_table_output(self.run_dir, path, table_name, len(rows)))

        objectives_path = tables_dir / "objectives.json"
        _write_json(
            objectives_path,
            build_objectives_summary(tables.get("optimization_history", [])),
        )
        outputs.append(
            _table_output(
                self.run_dir,
                objectives_path,
                "objectives",
                1 if tables.get("optimization_history", []) else 0,
            )
        )
        return outputs

    def write_delta_arrays(
        self, stimuli: list[Stimulus], responses: dict[str, np.ndarray]
    ) -> list[dict[str, Any]]:
        arrays, index_rows = build_delta_arrays(stimuli, responses)
        requires_delta_artifact = any(
            stimulus.suite in {"virtual_lesion_lab", "counterfactual_editing_workbench"}
            for stimulus in stimuli
        )
        if not arrays and not requires_delta_artifact:
            return []
        delta_path = self.outputs_dir / "deltas.npz"
        payload: dict[str, Any] = dict(arrays)
        np.savez_compressed(delta_path, **payload)
        delta_sha = sha256_file(delta_path)
        index_path = self.outputs_dir / "deltas.index.jsonl"
        _write_jsonl(index_path, index_rows)
        return [
            {
                "id": "deltas",
                "path": str(delta_path.relative_to(self.run_dir)),
                "sha256": delta_sha,
                "media_type": "application/vnd.numpy.npz",
                "arrays": len(arrays),
            },
            {
                "id": "deltas:index",
                "path": str(index_path.relative_to(self.run_dir)),
                "sha256": sha256_file(index_path),
                "media_type": "application/x-ndjson",
                "rows": len(index_rows),
            },
        ]

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
        required_outputs = _required_completed_outputs(manifest)
        errors.extend(
            [
                f"missing {item}"
                for item in sorted(required_outputs)
                if not (root / item).is_file()
            ]
        )
        if not manifest.get("outputs"):
            errors.append("completed artifacts must include outputs")
        output_paths = {item.get("path") for item in manifest.get("outputs", [])}
        for path in sorted(required_outputs):
            if path not in output_paths:
                errors.append(f"completed artifacts must include {path} output")
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
    errors.extend(_validate_checksums(root))
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
                },
                {
                    "id": "responses:index",
                    "path": "outputs/responses.index.jsonl",
                    "sha256": sha256_file(outputs / "responses.index.jsonl"),
                    "media_type": "application/x-ndjson",
                    "rows": 1,
                },
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
    (tmp / "executive_summary.pdf").write_bytes(
        b"%PDF-1.4\n% Braindough fixture PDF\n%%EOF\n"
    )
    (tmp / "figures" / "response_similarity.png").write_bytes(b"fixture figure\n")
    (tmp / "figures" / "mean_abs_activation.png").write_bytes(b"fixture figure\n")
    pdf_sha = sha256_file(tmp / "executive_summary.pdf")
    manifest = json.loads((tmp / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"].extend(
        [
            {
                "id": "report:report.md",
                "path": "report.md",
                "sha256": sha256_file(tmp / "report.md"),
                "media_type": "text/markdown",
            },
            {
                "id": "report:report.html",
                "path": "report.html",
                "sha256": sha256_file(tmp / "report.html"),
                "media_type": "text/html",
            },
            {
                "id": "report:executive_summary",
                "path": "executive_summary.pdf",
                "sha256": pdf_sha,
                "media_type": "application/pdf",
            },
            {
                "id": "figure:response_similarity.png",
                "path": "figures/response_similarity.png",
                "sha256": sha256_file(tmp / "figures" / "response_similarity.png"),
                "media_type": "image/png",
            },
            {
                "id": "figure:mean_abs_activation.png",
                "path": "figures/mean_abs_activation.png",
                "sha256": sha256_file(tmp / "figures" / "mean_abs_activation.png"),
                "media_type": "image/png",
            },
        ]
    )
    _write_json(tmp / "manifest.json", manifest)
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_sanitize_local_paths(row), sort_keys=True) + "\n")


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


def _required_completed_outputs(manifest: dict[str, Any]) -> set[str]:
    paths = {
        "outputs/responses.npz",
        "outputs/responses.index.jsonl",
        "report.md",
        "report.html",
        "executive_summary.pdf",
        "figures/response_similarity.png",
        "figures/mean_abs_activation.png",
    }
    suites = _manifest_suites(manifest)
    if "latent_network_ica_explorer" in suites:
        paths.update(
            {
                "outputs/tables/latent_components.csv",
                "outputs/tables/latent_loadings.csv",
            }
        )
    if "virtual_lesion_lab" in suites:
        paths.update(
            {
                "outputs/deltas.npz",
                "outputs/deltas.index.jsonl",
                "outputs/tables/lesion_manifest.csv",
                "outputs/tables/lesion_comparisons.csv",
                "outputs/tables/lesion_roi_summary.csv",
                "outputs/tables/top_delta_vertices.csv",
                "figures/virtual_lesion_contact_sheet.png",
                "figures/lesion_scoreboard.png",
                "figures/lesion_dose_response.png",
            }
        )
    if "discrete_stimulus_optimizer" in suites:
        paths.update(
            {
                "outputs/tables/candidate_catalog.jsonl",
                "outputs/tables/optimization_history.jsonl",
                "outputs/tables/objectives.json",
                "figures/optimization_trace.png",
                "figures/optimizer_candidate_contact_sheet.png",
                "figures/optimization_score_components.png",
            }
        )
    if "counterfactual_editing_workbench" in suites:
        paths.update(
            {
                "outputs/deltas.npz",
                "outputs/deltas.index.jsonl",
                "outputs/tables/counterfactual_edits.jsonl",
                "outputs/tables/counterfactual_pairs.csv",
                "figures/counterfactual_delta_grid.png",
                "figures/counterfactual_tradeoff.png",
            }
        )
    if "bold5000_roi_encoding" in suites:
        paths.update(
            {
                "outputs/tables/bold5000_trials.csv",
                "outputs/tables/bold5000_roi_scores.csv",
                "outputs/tables/bold5000_model_comparison.csv",
                "outputs/tables/bold5000_permutation_scores.csv",
                "outputs/tables/bold5000_feature_weights.csv",
                "outputs/tables/bold5000_provenance.json",
                "figures/bold5000_roi_scores.png",
                "figures/bold5000_model_comparison.png",
            }
        )
    return paths


def _manifest_suites(manifest: dict[str, Any]) -> set[str]:
    suites: set[str] = set()
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict) and isinstance(metadata.get("suite"), str):
            suites.add(metadata["suite"])
    return suites


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
    value = _NON_FILE_URL_RE.sub("", value)
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


def _validate_checksums(root: Path) -> list[str]:
    checksum_path = root / "checksums.sha256"
    errors: list[str] = []
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        match = re.match(r"^([0-9a-fA-F]{64})  (.+)$", line)
        if not match:
            errors.append(f"checksums.sha256 line {line_number} is malformed")
            continue
        digest, relative_path = match.groups()
        if Path(relative_path).is_absolute():
            errors.append(f"checksums.sha256 path is not relative: {relative_path}")
            continue
        entries[relative_path] = digest.lower()

    expected: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != checksum_path.name:
            expected[str(path.relative_to(root))] = sha256_file(path)

    for relative_path, expected_digest in expected.items():
        digest = entries.get(relative_path)
        if digest is None:
            errors.append(f"checksums.sha256 missing {relative_path}")
        elif digest != expected_digest:
            errors.append(f"checksums.sha256 mismatch {relative_path}")
    for relative_path in sorted(set(entries) - set(expected)):
        errors.append(f"checksums.sha256 references missing {relative_path}")
    return errors


def _git_state() -> tuple[str | None, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, False
