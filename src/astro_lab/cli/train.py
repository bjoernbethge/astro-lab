#!/usr/bin/env python3
"""
AstroLab Training CLI - 2025 Edition
====================================

CLI for training astronomical ML models with state-of-the-art optimizations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

from astro_lab.training.astro_trainer import AstroTrainer


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train astronomical ML models with AstroLab 2025",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  astro-lab train -c my_experiment.yaml --verbose
  astro-lab train --dataset gaia --model astro_graph_gnn --epochs 50 --batch-size 32
  
  # Large-scale training with sampling
  astro-lab train --dataset gaia --sampling-strategy neighbor --neighbor-sizes 25 10 5
  astro-lab train --dataset nsa --sampling-strategy cluster --num-clusters 1500
  astro-lab train --dataset sdss --sampling-strategy saint --saint-coverage 50
  
  # Dynamic batching and compilation
  astro-lab train --dataset gaia --enable-dynamic-batching --compile-model
  astro-lab train -c config.yaml --compile-mode reduce-overhead --compile-dynamic
  
  # Distributed training
  astro-lab train --dataset gaia --devices 4 --partition-method metis --num-partitions 4
  
  # Development/debugging
  astro-lab train --dataset gaia --max-samples 1000 --overfit-batches 10
""",
    )
    
    # Basic training arguments
    parser.add_argument(
        "-c", "--config", type=Path, help="Path to YAML configuration file"
    )
    parser.add_argument("--dataset", type=str, help="Dataset/Survey name")
    parser.add_argument("--model", dest="model_type", type=str, help="Model type/name")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", "--lr", type=float, help="Learning rate")
    parser.add_argument("--devices", type=int, help="Number of GPUs/devices to use")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to resume from")
    parser.add_argument(
        "--overfit-batches", type=int, help="Overfit on N batches for debugging"
    )
    parser.add_argument("--max-samples", type=int, help="Maximum samples for debugging")
    parser.add_argument(
        "--precision", type=str, help="Training precision (e.g. 16-mixed)"
    )
    
    # Large-scale sampling arguments
    sampling_group = parser.add_argument_group("Large-scale sampling")
    sampling_group.add_argument(
        "--sampling-strategy",
        type=str,
        choices=["none", "neighbor", "cluster", "saint"],
        default="none",
        help="Sampling strategy for large graphs",
    )
    sampling_group.add_argument(
        "--neighbor-sizes",
        type=int,
        nargs="+",
        default=[25, 10],
        help="Number of neighbors to sample per hop (for neighbor sampling)",
    )
    sampling_group.add_argument(
        "--num-clusters",
        type=int,
        default=1500,
        help="Number of clusters for ClusterGCN",
    )
    sampling_group.add_argument(
        "--saint-coverage",
        type=int,
        default=50,
        help="Sample coverage for GraphSAINT",
    )
    sampling_group.add_argument(
        "--saint-walk-length",
        type=int,
        default=2,
        help="Random walk length for GraphSAINT",
    )
    
    # Dynamic batching arguments
    batching_group = parser.add_argument_group("Dynamic batching")
    batching_group.add_argument(
        "--enable-dynamic-batching",
        action="store_true",
        help="Enable dynamic batch size adjustment",
    )
    batching_group.add_argument(
        "--min-batch-size",
        type=int,
        default=1,
        help="Minimum batch size for dynamic batching",
    )
    batching_group.add_argument(
        "--max-batch-size",
        type=int,
        default=512,
        help="Maximum batch size for dynamic batching",
    )
    
    # Model compilation arguments
    compile_group = parser.add_argument_group("Model compilation")
    compile_group.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable torch.compile for faster training",
    )
    compile_group.add_argument(
        "--compile-mode",
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode",
    )
    compile_group.add_argument(
        "--compile-dynamic",
        action="store_true",
        default=True,
        help="Enable dynamic shapes in torch.compile",
    )
    
    # Distributed training arguments
    dist_group = parser.add_argument_group("Distributed training")
    dist_group.add_argument(
        "--partition-method",
        type=str,
        choices=["metis", "random"],
        help="Graph partitioning method for distributed training",
    )
    dist_group.add_argument(
        "--num-partitions",
        type=int,
        default=4,
        help="Number of partitions for distributed training",
    )
    
    # Other arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (optional)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    
    return parser


def load_config_from_file(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def main(args=None) -> int:
    parser = create_parser()
    parsed_args = parser.parse_args(args)
    logger = setup_logging(parsed_args.verbose)

    # Load config from YAML if provided
    config = {}
    if parsed_args.config:
        config = load_config_from_file(parsed_args.config)
        logger.info(f"Loaded configuration from {parsed_args.config}")

    # CLI overrides
    cli_overrides = {}
    
    # Basic training parameters
    if parsed_args.dataset:
        cli_overrides["survey"] = parsed_args.dataset
    if parsed_args.model_type:
        cli_overrides["model_type"] = parsed_args.model_type
    if parsed_args.epochs:
        cli_overrides["max_epochs"] = parsed_args.epochs
    if parsed_args.batch_size:
        cli_overrides["batch_size"] = parsed_args.batch_size
    if parsed_args.learning_rate:
        cli_overrides["learning_rate"] = parsed_args.learning_rate
    if parsed_args.devices:
        cli_overrides["devices"] = parsed_args.devices
    if parsed_args.checkpoint:
        cli_overrides["checkpoint"] = str(parsed_args.checkpoint)
    if parsed_args.overfit_batches:
        cli_overrides["overfit_batches"] = parsed_args.overfit_batches
    if parsed_args.max_samples:
        cli_overrides["max_samples"] = parsed_args.max_samples
    if parsed_args.precision:
        cli_overrides["precision"] = parsed_args.precision
        
    # Large-scale sampling parameters
    if parsed_args.sampling_strategy:
        cli_overrides["sampling_strategy"] = parsed_args.sampling_strategy
    if parsed_args.neighbor_sizes:
        cli_overrides["neighbor_sizes"] = parsed_args.neighbor_sizes
    if parsed_args.num_clusters:
        cli_overrides["num_clusters"] = parsed_args.num_clusters
    if parsed_args.saint_coverage:
        cli_overrides["saint_sample_coverage"] = parsed_args.saint_coverage
    if parsed_args.saint_walk_length:
        cli_overrides["saint_walk_length"] = parsed_args.saint_walk_length
        
    # Dynamic batching parameters
    if parsed_args.enable_dynamic_batching:
        cli_overrides["enable_dynamic_batching"] = True
    if parsed_args.min_batch_size:
        cli_overrides["min_batch_size"] = parsed_args.min_batch_size
    if parsed_args.max_batch_size:
        cli_overrides["max_batch_size"] = parsed_args.max_batch_size
        
    # Model compilation parameters
    if parsed_args.compile_model:
        cli_overrides["compile_model"] = True
    if parsed_args.compile_mode:
        cli_overrides["compile_mode"] = parsed_args.compile_mode
    if hasattr(parsed_args, "compile_dynamic"):
        cli_overrides["compile_dynamic"] = parsed_args.compile_dynamic
        
    # Distributed training parameters
    if parsed_args.partition_method:
        cli_overrides["partition_method"] = parsed_args.partition_method
    if parsed_args.num_partitions:
        cli_overrides["num_partitions"] = parsed_args.num_partitions
        
    # Output directory
    if parsed_args.output_dir is not None:
        cli_overrides["output_dir"] = str(parsed_args.output_dir)
    if "output_dir" not in cli_overrides and "output_dir" not in config:
        cli_overrides["output_dir"] = "results"

    # Merge config and CLI overrides
    config.update(cli_overrides)

    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration summary
    if logger.isEnabledFor(logging.INFO):
        logger.info("Configuration summary:")
        logger.info(f"  Survey: {config.get('survey', 'Not specified')}")
        logger.info(f"  Model: {config.get('model_type', 'Not specified')}")
        logger.info(f"  Sampling strategy: {config.get('sampling_strategy', 'none')}")
        logger.info(f"  Batch size: {config.get('batch_size', 32)}")
        logger.info(f"  Devices: {config.get('devices', 1)}")
        logger.info(f"  Compile model: {config.get('compile_model', False)}")

    try:
        # Initialize trainer
        logger.info("Initializing AstroTrainer...")
        trainer = AstroTrainer(config)

        # Setup and train
        logger.info("Starting training...")
        results = trainer.train()

        # Save results
        results_file = output_dir / "training_results.yaml"
        with open(results_file, "w") as f:
            yaml.dump(results, f, default_flow_style=False)

        logger.info(f"Training completed! Results saved to {results_file}")
        logger.info(f"Best validation accuracy: {results.get('best_val_acc', 0):.4f}")
        
        # Save final model if requested
        if config.get("save_final_model", True):
            model_path = output_dir / "final_model.pt"
            trainer.save_checkpoint(str(model_path))
            logger.info(f"Final model saved to {model_path}")
        
        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
