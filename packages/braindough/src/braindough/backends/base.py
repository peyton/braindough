"""Backend contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from braindough.config import ExperimentSpec
from braindough.stimuli import Stimulus
from braindough.storage import BraindoughPaths


@dataclass
class BackendResult:
    """Backend output before artifact serialization."""

    status: str
    responses: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    outputs: list[dict[str, object]] = field(default_factory=list)
    blocker: str | None = None


class Backend(Protocol):
    name: str

    def run(
        self,
        spec: ExperimentSpec,
        stimuli: list[Stimulus],
        paths: BraindoughPaths,
        run_dir: Path,
    ) -> BackendResult:
        """Run predictions for generated stimuli."""
        ...
