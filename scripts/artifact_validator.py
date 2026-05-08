"""Artifact validator script wrapper."""

from braindough.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", "--fixture"]))
