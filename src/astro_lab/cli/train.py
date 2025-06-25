"""
Training CLI module for AstroLab - Thin wrapper around training module.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for training command."""
    parser = argparse.ArgumentParser(
        prog="astro-lab train",
        description="Train astronomical machine learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file or quick training options (new syntax)
    config_group = parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file (YAML)",
    )
    config_group.add_argument(
        "--quick",
        "-q",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Quick training with dataset and model names",
    )

    # Backward compatibility for README examples
    parser.add_argument("--dataset", help="Dataset name (backward compatibility)")
    parser.add_argument("--model", help="Model name (backward compatibility)")

    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--strategy", choices=["auto", "ddp", "fsdp"], default="auto")
    parser.add_argument(
        "--precision", choices=["32", "16-mixed", "bf16-mixed"], default="16-mixed"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Import only what we need
    from astro_lab.training import train_from_config, train_quick

    # Determine training mode with backward compatibility
    if args.config:
        # Train from config file
        return train_from_config(
            config_path=args.config,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            devices=args.devices,
            precision=args.precision,
        )
    elif args.quick:
        # Quick training (new syntax)
        dataset, model = args.quick
        return train_quick(
            dataset=dataset,
            model=model,
            epochs=args.epochs or 10,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            learning_rate=args.learning_rate or 0.001,
            devices=args.devices,
            precision=args.precision,
            strategy=args.strategy,
        )
    elif args.dataset and args.model:
        # Backward compatibility with README examples
        return train_quick(
            dataset=args.dataset,
            model=args.model,
            epochs=args.epochs or 10,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            learning_rate=args.learning_rate or 0.001,
            devices=args.devices,
            precision=args.precision,
            strategy=args.strategy,
        )
    else:
        # No valid training option provided
        parser.error("Must provide either --config, --quick, or --dataset and --model")


if __name__ == "__main__":
    sys.exit(main())
