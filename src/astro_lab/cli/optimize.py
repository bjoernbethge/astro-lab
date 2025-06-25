"""
Optimization CLI module for AstroLab - Thin wrapper around training module.
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    """Main entry point for optimization command."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for astronomical models"
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Configuration file path"
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of optimization trials"
    )
    parser.add_argument(
        "--ui", action="store_true", help="Start MLflow UI after optimization"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Import only what we need
    from astro_lab.training import optimize_hyperparameters

    try:
        return optimize_hyperparameters(
            config_path=args.config,
            n_trials=args.trials,
            ui=args.ui,
            port=args.port,
        )
    except KeyboardInterrupt:
        logger.error("Optimization interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
