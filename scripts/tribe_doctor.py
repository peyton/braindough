"""TRIBE v2 import/runtime doctor."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from braindough.storage import BraindoughPaths


def main() -> int:
    paths = BraindoughPaths.discover()
    candidates = [
        os.environ.get("BRAINDOUGH_TRIBE_REPO"),
        str(paths.tribe_code_dir),
        "/tmp/braindough-tribev2-inspect",
    ]
    repo = next(
        (
            Path(candidate).expanduser()
            for candidate in candidates
            if candidate and (Path(candidate).expanduser() / "tribev2").is_dir()
        ),
        None,
    )
    if repo is not None:
        sys.path.insert(0, str(repo))
    payload: dict[str, object] = {
        "repo": str(repo) if repo is not None else None,
        "storage": str(paths.home),
        "importable": False,
    }
    try:
        import tribev2  # type: ignore[import-not-found]  # noqa: F401

        payload["importable"] = True
    except Exception as exc:
        payload["error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["importable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
