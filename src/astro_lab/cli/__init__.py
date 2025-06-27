"""
AstroLab CLI - Command Line Interface
====================================

Provides command-line tools for training, evaluation, and visualization.
"""

import argparse
import sys
from typing import List, Optional

# Command registry - import lazily to speed up startup
COMMANDS = {
    "train": "astro_lab.cli.train",
    "download": "astro_lab.cli.download",
    "optimize": "astro_lab.cli.optimize",
    "preprocess": "astro_lab.cli.preprocess",
    # "process": "astro_lab.cli.process",  # Deprecated, use preprocess
}


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the AstroLab CLI.
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="astro-lab",
        description="AstroLab: Modern Astronomical Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  train       Train models on astronomical data
  download    Download survey data
  optimize    Optimize model hyperparameters
  preprocess  Preprocess survey data
        """,
    )
    
    parser.add_argument(
        "command",
        choices=COMMANDS.keys(),
        help="Command to run",
    )
    
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments for the command",
    )
    
    # Parse only the command
    args = parser.parse_args(argv or sys.argv[1:2])
    
    # Import and run the command module lazily
    try:
        import importlib
        module = importlib.import_module(COMMANDS[args.command])
        
        # Set argv for the subcommand
        sys.argv = [f"astro-lab {args.command}"] + (argv or sys.argv)[2:]
        
        # Run the command's main function
        return module.main()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


__all__ = ["main"]
