"""Backend registry."""

from braindough.backends.base import Backend, BackendResult
from braindough.backends.fake import FakeBackend
from braindough.backends.tribe_v2 import TribeV2Backend

__all__ = ["Backend", "BackendResult", "FakeBackend", "TribeV2Backend", "get_backend"]


def get_backend(name: str) -> Backend:
    if name == "fake":
        return FakeBackend()
    if name == "tribe-v2":
        return TribeV2Backend()
    raise ValueError(f"Unknown backend: {name}")
