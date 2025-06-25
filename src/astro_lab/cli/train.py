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
    
    # Config file or quick training options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to configuration file (YAML)",
    )
    config_group.add_argument(
        "--quick", "-q",
        nargs=2,
        metavar=("DATASET", "MODEL"),
        help="Quick training with dataset and model names",
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--strategy", choices=["auto", "ddp", "fsdp"], default="auto")
    parser.add_argument("--precision", choices=["32", "16-mixed", "bf16-mixed"], default="16-mixed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Import only what we need
    from astro_lab.training import train_from_config, train_quick
    
    # Let the training functions handle all errors and return appropriate exit codes
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
    else:
        # Quick training
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

if __name__ == "__main__":
    sys.exit(main())
