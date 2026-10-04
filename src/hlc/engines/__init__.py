"""Move engines package."""

# Lazy import: maia3 and torch are heavy optional dependencies.
# Import Maia3Engine explicitly when needed rather than at package load time.
# This allows unit tests for guard.py / pacing.py to run without the full ML stack.


def __getattr__(name: str):
    if name == "Maia3Engine":
        from hlc.engines.maia3 import Maia3Engine  # noqa: PLC0415
        return Maia3Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Maia3Engine"]
