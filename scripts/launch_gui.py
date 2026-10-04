"""launch_gui.py — Launch the HLC Desktop Native Agent Dashboard or Bot Worker."""

import os
import sys

# Ensure root_dir, src/, scripts/, and _maia3_repo are on pythonpath
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in [
    root_dir,
    os.path.join(root_dir, "src"),
    os.path.join(root_dir, "scripts"),
    os.path.join(root_dir, "_maia3_repo"),
]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

if __name__ == "__main__":
    if "--bot-worker" in sys.argv:
        sys.argv.remove("--bot-worker")
        try:
            from scripts.chesscom_bot import main as bot_main
        except ModuleNotFoundError:
            from chesscom_bot import main as bot_main
        bot_main()
    else:
        from hlc.gui.app import main
        main()
