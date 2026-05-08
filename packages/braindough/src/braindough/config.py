"""Experiment configuration loading."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"^(?:[a-zA-Z]:[\\/]|\\\\[^\\/]+[\\/][^\\/]+|\\[^\\/]+)"
)


@dataclass(frozen=True)
class ExperimentSpec:
    """Normalized experiment specification."""

    experiment_id: str
    title: str
    backend: str
    seed: int
    suites: tuple[str, ...]
    stimuli: dict[str, Any] = field(default_factory=dict)
    backend_config: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def slug(self) -> str:
        return self.experiment_id.replace("/", "-").replace("_", "-")

    def to_dict(self) -> dict[str, Any]:
        """Return the path-independent experiment identity."""

        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "backend": self.backend,
            "seed": self.seed,
            "suites": list(self.suites),
            "stimuli": _path_neutral(self.stimuli),
            "backend_config": _path_neutral(self.backend_config),
            "output": _path_neutral(self.output),
        }


def load_experiment_spec(path: str | Path) -> ExperimentSpec:
    """Load and validate a YAML experiment spec."""

    spec_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment spec must be a mapping: {spec_path}")

    required = ["id", "title", "backend", "suites"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{spec_path} is missing required fields: {missing}")

    suites = raw["suites"]
    if (
        not isinstance(suites, list)
        or not suites
        or not all(isinstance(item, str) for item in suites)
    ):
        raise ValueError("'suites' must be a non-empty list of strings")

    seed = raw.get("seed", 0)
    if not isinstance(seed, int):
        raise ValueError("'seed' must be an integer")

    return ExperimentSpec(
        experiment_id=str(raw["id"]),
        title=str(raw["title"]),
        backend=str(raw["backend"]),
        seed=seed,
        suites=tuple(suites),
        stimuli=dict(raw.get("stimuli", {})),
        backend_config=dict(raw.get("backend_config", {})),
        output=dict(raw.get("output", {})),
        source_path=spec_path,
    )


def discover_experiment_specs(root: str | Path = "experiments") -> list[Path]:
    """Return all experiment YAML files in a stable order."""

    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(root_path.glob("**/*.yaml"))


def _path_neutral(value: Any) -> Any:
    """Return config identity data without machine-local absolute paths."""

    if isinstance(value, dict):
        return {str(key): _path_neutral(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_path_neutral(child) for child in value]
    if isinstance(value, Path):
        path = str(value)
        return _local_path_token(path) if is_absolute_local_path(path) else path
    if isinstance(value, str) and is_absolute_local_path(value):
        return _local_path_token(value)
    return value


def is_absolute_local_path(value: str) -> bool:
    """Return whether a string is a POSIX, Windows, UNC, or file URI path."""

    if value.startswith("file://"):
        return True
    try:
        if Path(value).expanduser().is_absolute():
            return True
    except RuntimeError:
        if Path(value).is_absolute():
            return True
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(value))


def _local_path_token(path: str) -> dict[str, str]:
    normalized = path.replace("\\", "/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return {
        "local_path_name": _local_path_name(path),
        "local_path_sha256": digest,
    }


def _local_path_name(path: str) -> str:
    stripped = path.removeprefix("file://").rstrip("/\\")
    parts = re.split(r"[\\/]+", stripped)
    return parts[-1] if parts and parts[-1] else "path"
