"""Validate saved research capture metadata."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESEARCH_DIR = Path("docs/research")
VALID_STATUSES = {
    "blocked",
    "captured",
    "captured_partial",
    "completed",
    "in_progress",
    "launched",
}
REQUIRED_IDS = {
    "project_deep_research_report",
    "latent_network_ica_explorer",
    "virtual_lesion_lab",
    "discrete_stimulus_optimizer",
    "counterfactual_editing_workbench",
}


def main() -> int:
    errors = validate_research_dir(RESEARCH_DIR)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"valid": True}, indent=2))
    return 0


def validate_research_dir(path: Path) -> list[str]:
    if not path.is_dir():
        return [f"missing research directory: {path}"]

    errors: list[str] = []
    seen_ids: set[str] = set()
    metadata_paths = sorted(path.glob("*.metadata.json"))
    if not metadata_paths:
        return [f"no research metadata files found in {path}"]

    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{metadata_path} is invalid JSON: {exc}")
            continue
        errors.extend(_validate_metadata(metadata_path, metadata))
        research_id = metadata.get("id")
        if isinstance(research_id, str):
            seen_ids.add(research_id)
            markdown_path = metadata_path.with_name(
                metadata_path.name.removesuffix(".metadata.json") + ".md"
            )
            if not markdown_path.is_file():
                errors.append(f"{metadata_path} missing sibling markdown report")
            elif not markdown_path.read_text(encoding="utf-8").strip():
                errors.append(f"{markdown_path} is empty")

    missing_ids = sorted(REQUIRED_IDS - seen_ids)
    if missing_ids:
        errors.append(f"missing required research ids: {missing_ids}")
    return errors


def _validate_metadata(path: Path, metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["id", "title", "status", "source", "prompt", "created_at"]
    for field in required:
        if field not in metadata:
            errors.append(f"{path} missing {field}")
    if metadata.get("status") not in VALID_STATUSES:
        errors.append(f"{path} has invalid status: {metadata.get('status')}")
    if not isinstance(metadata.get("source"), dict):
        errors.append(f"{path} source must be a mapping")
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str) or len(prompt.strip()) < 40:
        errors.append(f"{path} prompt must be substantive")
    if not isinstance(metadata.get("created_at"), str) or not metadata["created_at"]:
        errors.append(f"{path} created_at must be a non-empty string")
    return errors


if __name__ == "__main__":
    sys.exit(main())
