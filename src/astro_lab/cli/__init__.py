"""
AstroLab CLI - Command Line Interface
====================================

Provides command-line tools for training, evaluation, and visualization.
"""

import importlib
import sys

# Import existing CLI modules
from .download import main as download_main
from .optimize import main as optimize_main
from .preprocess import main as preprocess_main
from .train import main as train_main

__all__ = [
    "download_main",
    "preprocess_main",
    "train_main",
    "optimize_main",
]


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: astro-lab <command> [options]\nAvailable commands: train, download, optimize, preprocess"
        )
        sys.exit(1)
    cmd = sys.argv[1]
    args = sys.argv[2:]

    # Set sys.argv to the remaining arguments for the submodules
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + args

    try:
        if cmd == "train":
            mod = importlib.import_module("astro_lab.cli.train")
            mod.main()
        elif cmd == "download":
            mod = importlib.import_module("astro_lab.cli.download")
            mod.main()
        elif cmd == "optimize":
            mod = importlib.import_module("astro_lab.cli.optimize")
            mod.main()
        elif cmd == "preprocess":
            mod = importlib.import_module("astro_lab.cli.preprocess")
            mod.main()
        else:
            print(
                f"Unknown command: {cmd}\nAvailable commands: train, download, optimize, preprocess"
            )
            sys.exit(1)
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    main()
