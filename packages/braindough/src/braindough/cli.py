"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from pathlib import Path

from braindough.artifacts import (
    cleanup_fixture_artifact,
    create_fixture_artifact,
    validate_artifact,
)
from braindough.config import discover_experiment_specs, load_experiment_spec
from braindough.report import write_report
from braindough.runner import run_experiment
from braindough.storage import BraindoughPaths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="braindough")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor")

    storage = subparsers.add_parser("storage")
    storage_sub = storage.add_subparsers(dest="storage_command", required=True)
    storage_sub.add_parser("init")
    storage_sub.add_parser("doctor")

    run = subparsers.add_parser("run")
    run.add_argument("spec", type=Path)
    run.add_argument("--home", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("run_dir", nargs="?", type=Path)
    validate.add_argument("--fixture", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("run_dir", type=Path)

    experiments = subparsers.add_parser("experiments")
    experiments_sub = experiments.add_subparsers(
        dest="experiments_command", required=True
    )
    experiments_sub.add_parser("list")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "storage":
        if args.storage_command == "init":
            paths = BraindoughPaths.discover()
            created = paths.init()
            print(
                json.dumps(
                    {
                        "home": str(paths.home),
                        "created": [str(path) for path in created],
                    },
                    indent=2,
                )
            )
            return 0
        return _storage_doctor()
    if args.command == "run":
        run_dir = run_experiment(args.spec, home=args.home)
        print(run_dir)
        return 0
    if args.command == "validate":
        return _validate(args.run_dir, fixture=args.fixture)
    if args.command == "report":
        md_path, html_path = write_report(args.run_dir)
        print(json.dumps({"report": str(md_path), "html": str(html_path)}, indent=2))
        return 0
    if args.command == "experiments":
        for path in discover_experiment_specs():
            spec = load_experiment_spec(path)
            print(f"{path}\t{spec.experiment_id}\t{spec.backend}\t{spec.title}")
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def _doctor() -> int:
    paths = BraindoughPaths.discover()
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "storage_home": str(paths.home),
        "worktree_id": paths.worktree_id,
        "storage_exists": paths.home.exists(),
        "experiments": [str(path) for path in discover_experiment_specs()],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _storage_doctor() -> int:
    paths = BraindoughPaths.discover()
    expected = paths.init()
    payload = {
        "home": str(paths.home),
        "worktree_id": paths.worktree_id,
        "paths": {
            path.name: {"path": str(path), "exists": path.exists()} for path in expected
        },
        "env": paths.env(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate(run_dir: Path | None, fixture: bool) -> int:
    if fixture:
        path = create_fixture_artifact()
        try:
            errors = validate_artifact(path)
        finally:
            cleanup_fixture_artifact(path)
    else:
        if run_dir is None:
            raise SystemExit("validate requires RUN_DIR unless --fixture is used")
        errors = validate_artifact(run_dir)

    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    print(json.dumps({"valid": True}, indent=2))
    return 0
