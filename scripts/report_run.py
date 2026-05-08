"""Report generation script wrapper."""

from __future__ import annotations

import sys

from braindough.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["report", *sys.argv[1:]]))
