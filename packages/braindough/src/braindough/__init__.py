"""Braindough experiment harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("braindough")
except PackageNotFoundError:  # pragma: no cover - editable checkout fallback
    __version__ = "0.0.0"

__all__ = ["__version__"]
