"""Adapters package for connecting HLC to test harnesses, Lichess, and manual CLIs."""

# All heavy adapters (HLCEngine, harness classes) are lazy — they pull in maia3+torch
# which are optional in lightweight / CI environments. Import them explicitly when needed.

_LAZY_MAP = {
    "HLCEngine": ("hlc.adapters.lichess_bot_engine", "HLCEngine"),
    "HarnessOpponent": ("hlc.adapters.local_harness", "HarnessOpponent"),
    "LocalHarness": ("hlc.adapters.local_harness", "LocalHarness"),
    "Maia3Opponent": ("hlc.adapters.local_harness", "Maia3Opponent"),
    "StockfishOpponent": ("hlc.adapters.local_harness", "StockfishOpponent"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP)
