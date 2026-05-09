"""Backend registry."""

from braindough.backends.base import Backend, BackendResult
from braindough.backends.bold5000_ridge import Bold5000RidgeBackend
from braindough.backends.fake import FakeBackend
from braindough.backends.tribe_v2 import TribeV2Backend

__all__ = [
    "Backend",
    "BackendResult",
    "Bold5000RidgeBackend",
    "FakeBackend",
    "TribeV2Backend",
    "get_backend",
]


def get_backend(name: str) -> Backend:
    if name == "fake":
        return FakeBackend()
    if name == "bold5000-ridge":
        return Bold5000RidgeBackend()
    if name == "tribe-v2":
        return TribeV2Backend()
    raise ValueError(f"Unknown backend: {name}")
