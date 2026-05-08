"""Storage layout for local-heavy experiment assets."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_HOME = Path("/Volumes/Virtual Machine HD/Projects/braindough")
TRIBE_V2_REVISION = "f894e783"
TRIBE_V2_SOURCE_REVISION = "72399081ed3f1040c4d996cefb2864a4c46f5b8e"


def default_home() -> Path:
    """Resolve the configured Braindough external storage home."""

    configured = os.environ.get("BRAINDOUGH_HOME")
    return Path(configured).expanduser() if configured else DEFAULT_HOME


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BraindoughPaths:
    """Resolved storage paths for one checkout."""

    home: Path
    workspace: Path

    @classmethod
    def discover(
        cls, home: str | Path | None = None, workspace: str | Path | None = None
    ) -> BraindoughPaths:
        return cls(
            home=(Path(home).expanduser() if home else default_home()).resolve(),
            workspace=(Path(workspace) if workspace else Path.cwd()).resolve(),
        )

    @property
    def worktree_id(self) -> str:
        return sha256_text(str(self.workspace))[:12]

    @property
    def shared(self) -> Path:
        return self.home / "shared"

    @property
    def hf_cache(self) -> Path:
        return self.shared / "hf-cache"

    @property
    def torch_cache(self) -> Path:
        return self.shared / "torch-cache"

    @property
    def tribe_model_dir(self) -> Path:
        return (
            self.shared
            / "models"
            / "tribe-v2"
            / f"facebook-tribev2-{TRIBE_V2_REVISION}"
        )

    @property
    def tribe_code_dir(self) -> Path:
        return self.shared / "code" / "tribe-v2" / TRIBE_V2_SOURCE_REVISION

    @property
    def blobs(self) -> Path:
        return self.home / "blobs" / "sha256"

    @property
    def worktree_root(self) -> Path:
        return self.home / "worktrees" / self.worktree_id

    @property
    def scratch(self) -> Path:
        return self.worktree_root / "scratch"

    @property
    def tmp(self) -> Path:
        return self.worktree_root / "tmp"

    @property
    def logs(self) -> Path:
        return self.worktree_root / "logs"

    @property
    def runs_root(self) -> Path:
        return self.home / "runs"

    def run_dir(self, run_id: str) -> Path:
        today = datetime.now(UTC)
        return self.runs_root / f"{today:%Y}" / f"{today:%m}" / run_id

    def init(self) -> list[Path]:
        """Create the stable storage directory layout."""

        paths = [
            self.tribe_model_dir,
            self.tribe_code_dir,
            self.hf_cache,
            self.torch_cache,
            self.blobs,
            self.scratch,
            self.tmp,
            self.logs,
            self.runs_root,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)
        return paths

    def env(self) -> dict[str, str]:
        """Environment variables used by heavy local backends."""

        return {
            "BRAINDOUGH_HOME": str(self.home),
            "HF_HOME": str(self.hf_cache),
            "TORCH_HOME": str(self.torch_cache),
            "BRAINDOUGH_WORKTREE_ID": self.worktree_id,
        }


def make_run_id(experiment_id: str, backend: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    slug = experiment_id.replace("/", "-").replace("_", "-")
    nonce = uuid.uuid4().hex[:8]
    suffix = sha256_text(f"{timestamp}:{experiment_id}:{backend}:{nonce}")[:8]
    return f"{timestamp}-{slug}-{backend}-{suffix}"
