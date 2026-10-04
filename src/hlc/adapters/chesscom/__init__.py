"""chess.com browser automation adapter for HLC."""

# All chesscom adapters are lazy — they require playwright which is not available
# in lightweight / CI test environments. Import them explicitly when needed.

_LAZY_MAP = {
    "ChessDotComBot": ("hlc.adapters.chesscom.bot", "ChessDotComBot"),
    "BoardReader": ("hlc.adapters.chesscom.board_reader", "BoardReader"),
    "ChessDotComBrowser": ("hlc.adapters.chesscom.browser", "ChessDotComBrowser"),
    "MoveExecutor": ("hlc.adapters.chesscom.move_executor", "MoveExecutor"),
}


def __getattr__(name: str):
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP)
