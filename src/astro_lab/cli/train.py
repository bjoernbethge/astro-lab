#!/usr/bin/env python3
"""
AstroLab Training CLI - Simple Command Interface
===============================================

Minimal CLI that only parses arguments and delegates to trainer classes.
All logging is handled by the trainer and lightning modules.
Updated for 2025 best practices including FSDP and modern optimization techniques.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Disable duplicate logging from Lightning BEFORE imports
os.environ["HYDRA_FULL_ERROR"] = "0"
os.environ["PYTORCH_LIGHTNING_SUPPRESS_LOGS"] = "1"
os.environ["PL_DISABLE_FORK"] = "1"

# Suppress all rank_zero logs
import warnings

warnings.filterwarnings("ignore")

# Suppress Lightning's rank_zero logging
import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Redirect stderr during imports to suppress duplicate messages
import contextlib
import io

# Suppress import-time messages
with contextlib.redirect_stderr(io.StringIO()):
    import torch
    import yaml
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        GradientAccumulationScheduler,
        LearningRateMonitor,
        ModelCheckpoint,
        StochasticWeightAveraging,
    )
    from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy


def train_from_config(config_path: str) -> None:
    """
    Train from configuration file - delegates to AstroTrainer.
    Updated with 2025 best practices.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load config without logging
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
    except Exception as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        sys.exit(1)

    # Import modules after logging setup
    from astro_lab.data.config import data_config
    from astro_lab.data.datamodule import AstroDataModule
    from astro_lab.models.factories import (
        create_asteroid_period_detector,
        create_gaia_classifier,
        create_galaxy_modeler,
        create_lightcurve_classifier,
        create_lsst_transient_detector,
        create_sdss_galaxy_model,
        create_temporal_graph_model,
    )
    from astro_lab.training.lightning_module import AstroLightningModule

    # Create a custom stderr that filters duplicate messages
    class FilteredStderr:
        def __init__(self):
            self.seen_messages = set()
            self.original_stderr = sys.stderr

        def write(self, msg):
            # Filter duplicate Lightning messages
            if msg.strip() and msg not in self.seen_messages:
                if any(
                    x in msg
                    for x in [
                        "GPU available",
                        "TPU available",
                        "HPU available",
                        "Using 16bit",
                    ]
                ):
                    self.seen_messages.add(msg)
                    return  # Don't print first occurrence either
                self.original_stderr.write(msg)

        def flush(self):
            self.original_stderr.flush()

    # Replace stderr temporarily
    filtered_stderr = FilteredStderr()
    old_stderr = sys.stderr
    sys.stderr = filtered_stderr

    try:
        # Setup experiment directories silently
        experiment_name = config.get("mlflow", {}).get(
            "experiment_name", "astro_training"
        )
        data_config.ensure_experiment_directories(experiment_name)

        # Extract data config
        data_config_section = config.get("data", {})
        dataset_name = data_config_section.get("dataset", "gaia")
        batch_size = data_config_section.get("batch_size", 32)
        max_samples = data_config_section.get("max_samples", 1000)
        k_neighbors = data_config_section.get("k_neighbors", 8)

        # Create datamodule with optimizations
        datamodule = AstroDataModule(
            survey=dataset_name,
            batch_size=batch_size,
            max_samples=max_samples,
            k_neighbors=k_neighbors,
            num_workers=data_config_section.get("num_workers", None),  # Auto-detect
            pin_memory=data_config_section.get("pin_memory", True),
            persistent_workers=data_config_section.get("persistent_workers", True),
            prefetch_factor=data_config_section.get("prefetch_factor", 2),
            # Laptop optimization parameters
            max_nodes_per_graph=data_config_section.get("max_nodes_per_graph", 1000),
            use_subgraph_sampling=data_config_section.get(
                "use_subgraph_sampling", True
            ),
        )

        # Setup datamodule to extract metadata
        datamodule.setup()

        # Extract model config
        model_config = config.get("model", {})
        model_type = model_config.get("type", "gaia_classifier")
        model_params = model_config.get("params", {})

        # Auto-detect classes if needed (silently)
        if "output_dim" not in model_params or model_params["output_dim"] is None:
            try:
                # Use datamodule metadata if available
                if hasattr(datamodule, "num_classes") and datamodule.num_classes:
                    num_classes = max(datamodule.num_classes, 2)
                    model_params["output_dim"] = num_classes
                    print(f"Auto-detected {num_classes} classes from data")
                else:
                    # Fallback: sample from dataloader
                    train_data = datamodule._main_data
                    if hasattr(train_data, "y"):
                        targets = train_data.y  # type: ignore[attr-defined]
                        num_classes = int(targets.max().item()) + 1
                        # Ensure at least 2 classes for classification
                        num_classes = max(num_classes, 2)
                        model_params["output_dim"] = num_classes
                        print(f"Auto-detected {num_classes} classes from data")
                    else:
                        model_params["output_dim"] = 4
                        print("Using default 4 classes")
            except Exception as e:
                model_params["output_dim"] = 4
                print(f"Using default 4 classes (error: {e})")

        # Create model
        if model_type == "gaia_classifier":
            model = create_gaia_classifier(
                num_classes=model_params.get("output_dim", 7),
                hidden_dim=model_params.get("hidden_dim", 128),
                device=model_params.get("device", None),
                **model_params,
            )
        elif model_type == "sdss_galaxy_classifier":
            model = create_sdss_galaxy_model(
                output_dim=model_params.get("output_dim", 5),
                hidden_dim=model_params.get("hidden_dim", 256),
                device=model_params.get("device", None),
                **model_params,
            )
        else:
            # Fallback: nutze create_gaia_classifier als Default
            model = create_gaia_classifier(**model_params)

        print(f"Created model: {type(model).__name__}")

        # Apply torch.compile with robust error handling BEFORE creating LightningModule
        if config.get("training", {}).get("use_compile", True):
            try:
                print("🔄 Attempting to compile model with torch.compile...")
                model = torch.compile(
                    model,
                    mode="reduce-overhead",  # More stable than "max-autotune"
                    backend="inductor",  # Use inductor backend for best performance
                    fullgraph=False,  # Allow fallback for unsupported operations
                    dynamic=True,  # Enable dynamic shapes
                )
                print("✅ Model compiled successfully with torch.compile")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
                print("🔄 Falling back to eager mode (no compilation)")
                try:
                    # Fallback to eager backend (no compilation, just Python)
                    model = torch.compile(
                        model,
                        backend="eager",
                        fullgraph=False,
                    )
                    print("✅ Model using eager backend (no compilation)")
                except Exception as fallback_error:
                    print(f"⚠️ Even eager compilation failed: {fallback_error}")
                    print("🔄 Using uncompiled model")
                    # Keep the original model without compilation
                    pass
        else:
            print("📝 Model training without torch.compile")

        # Extract training config
        training_config = config.get("training", {})

        # Stelle sicher, dass model eine Instanz von torch.nn.Module ist
        if callable(model):
            model = model()
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
            gradient_accumulation_steps=training_config.get(
                "gradient_accumulation_steps", 1
            ),
            gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
            gradient_clip_algorithm=training_config.get(
                "gradient_clip_algorithm", "norm"
            ),
            scheduler_type=training_config.get("scheduler_type", "cosine"),
            warmup_steps=training_config.get("warmup_steps", 0),
            use_compile=True,  # Always enable torch.compile
            use_ema=training_config.get("use_ema", False),
            ema_decay=training_config.get("ema_decay", 0.999),
            label_smoothing=training_config.get("label_smoothing", 0.0),
        )

        # Setup callbacks
        callbacks = []

        # Setup results structure in project root parallel to data/
        dataset_name = (
            datamodule.dataset_name
            if hasattr(datamodule, "dataset_name")
            else "unknown"
        )

        # Project root results structure: results/survey/models/, results/survey/statistics/, results/survey/visuals/
        project_root = Path.cwd()  # astro-lab/
        results_base = project_root / "results" / dataset_name
        models_dir = results_base / "models"
        statistics_dir = results_base / "statistics"
        visuals_dir = results_base / "visuals"

        # Create directories
        for dir_path in [models_dir, statistics_dir, visuals_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # ModelCheckpoint: speichert in results/survey/models/
        # Format: results/gaia/models/gaia_gcn.ckpt
        model_base_name = f"{dataset_name}_{model_params.get('conv_type', 'gcn')}"
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(models_dir),
                filename=f"{model_base_name}",  # gaia_gcn.ckpt
                monitor="val_loss",
                mode="min",
                save_top_k=1,  # Nur das beste
                save_last=False,  # Kein last model
                verbose=True,
            )
        )

        # EarlyStopping: stoppt, wenn sich val_loss nicht verbessert
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=training_config.get("patience", 10),
                mode="min",
                verbose=True,
            )
        )

        # LearningRateMonitor: loggt den Lernratenverlauf
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Stochastic Weight Averaging (optional)
        if training_config.get("use_swa", False):
            callbacks.append(
                StochasticWeightAveraging(
                    swa_lrs=training_config.get("swa_lr", 0.001),
                    swa_epoch_start=training_config.get("swa_epoch_start", 0.8),
                )
            )

        # Gradient accumulation scheduler (if needed)
        gradient_accumulation_schedule = training_config.get(
            "gradient_accumulation_schedule", None
        )
        # Also check in advanced section for backward compatibility
        if not gradient_accumulation_schedule and "advanced" in config:
            gradient_accumulation_schedule = config.get("advanced", {}).get(
                "gradient_accumulation_schedule", None
            )
        if gradient_accumulation_schedule:
            callbacks.append(
                GradientAccumulationScheduler(scheduling=gradient_accumulation_schedule)
            )

        # Setup MLflow logger if configured
        logger_instance = None
        if config.get("mlflow", {}).get("tracking_uri"):
            try:
                from lightning.pytorch.loggers import MLFlowLogger

                logger_instance = MLFlowLogger(
                    experiment_name=experiment_name,
                    tracking_uri=config["mlflow"]["tracking_uri"],
                    log_model=False,  # Disable model logging to reduce clutter
                )
            except:
                pass

        # Setup strategy (DDP, FSDP, etc.)
        strategy = training_config.get("strategy", "auto")
        if strategy == "fsdp":
            # FSDP strategy for large models
            from torch.distributed.fsdp import ShardingStrategy

            strategy = FSDPStrategy(
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=training_config.get("fsdp_cpu_offload", False),
                mixed_precision=training_config.get("mixed_precision", "bf16"),
            )
        elif strategy == "ddp":
            # DDP with optimizations
            strategy = DDPStrategy(
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        # Determine precision
        precision = training_config.get("precision", "16-mixed")
        if precision == "bf16":
            precision = "bf16-mixed"

        # Create and run trainer with minimal logging
        trainer = Trainer(
            max_epochs=training_config.get("max_epochs", 20),
            accelerator="auto",
            devices=training_config.get("devices", "auto"),
            strategy=strategy,
            precision=precision,
            callbacks=callbacks,
            logger=logger_instance,
            gradient_clip_val=None,  # Handled in LightningModule
            enable_progress_bar=True,
            enable_model_summary=True,  # Re-enable for debugging
            enable_checkpointing=True,
            log_every_n_steps=training_config.get("log_every_n_steps", 50),
            # Don't set accumulate_grad_batches here since we use manual optimization
            deterministic=training_config.get("deterministic", True),
            benchmark=training_config.get("benchmark", False),
            profiler=training_config.get("profiler", None),
            val_check_interval=training_config.get("val_check_interval", 1.0),
        )

        print(
            f"Starting training for {training_config.get('max_epochs', 20)} epochs..."
        )
        print(
            f"Strategy: {type(strategy).__name__ if hasattr(strategy, '__name__') else strategy}"
        )
        print(f"Precision: {precision}")
        print(
            f"Gradient accumulation: {training_config.get('gradient_accumulation_steps', 1)} steps"
        )

        # Train
        trainer.fit(lightning_module, datamodule=datamodule)

        print("Training completed!")

        # Test
        trainer.test(lightning_module, datamodule=datamodule)

        # Collect training statistics
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        # Get trainer metrics
        train_metrics = (
            trainer.logged_metrics if hasattr(trainer, "logged_metrics") else {}
        )

        # Create focused training summary with statistics
        training_summary = {
            "model": f"{dataset_name}_{model_params.get('conv_type', 'gcn')}",
            "dataset": dataset_name,
            "model_path": str(checkpoint_callback.best_model_path)
            if checkpoint_callback
            else None,
            # Training Statistics
            "training_stats": {
                "epochs_trained": training_config.get("max_epochs", 20),
                "best_val_loss": float(train_metrics.get("val_loss", 0.0)),
                "best_val_acc": float(train_metrics.get("val_acc", 0.0)),
                "final_train_loss": float(train_metrics.get("train_loss_epoch", 0.0)),
                "test_acc": float(train_metrics.get("test_acc", 0.0)),
                "test_loss": float(train_metrics.get("test_loss", 0.0)),
            },
            # Model Architecture
            "architecture": {
                "conv_type": model_params.get("conv_type", "gcn"),
                "hidden_dim": model_params.get("hidden_dim", 128),
                "num_layers": model_params.get("num_layers", 3),
                "dropout": model_params.get("dropout", 0.2),
                "activation": model_params.get("activation", "relu"),
            },
            # Key Training Parameters
            "hyperparameters": {
                "learning_rate": training_config.get("learning_rate", 0.001),
                "batch_size": training_config.get("batch_size", 1),
                "precision": training_config.get("precision", "16-mixed"),
                "max_samples": training_config.get("max_samples"),
                "gradient_accumulation": training_config.get(
                    "gradient_accumulation_steps", 1
                ),
            },
        }

        # Save training summary in statistics directory
        summary_path = statistics_dir / f"{model_base_name}_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(training_summary, f, default_flow_style=False, indent=2)

        print(f"🏆 Model saved: {models_dir / f'{model_base_name}.ckpt'}")
        print(f"📊 Statistics saved: {summary_path}")
        print(f"📈 Visuals directory: {visuals_dir}")
        print(f"📁 Results structure: {results_base}")

        # Cleanup
        import gc

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        if "--verbose" in sys.argv:
            import traceback

            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore stderr
        sys.stderr = old_stderr


def train_quick(
    dataset: str,
    model: str,
    epochs: int = 10,
    batch_size: int = 32,
    max_samples: int = 1000,
    learning_rate: float = 0.001,
    devices: int = 1,
    strategy: str = "auto",
    precision: str = "16-mixed",
    accumulate: int = 1,
) -> None:
    """
    Quick training without config file.
    Updated with 2025 best practices and advanced options.

    Args:
        dataset: Dataset name (gaia, sdss, nsa)
        model: Model type (gaia_classifier, etc.)
        epochs: Number of epochs
        batch_size: Batch size
        max_samples: Maximum number of samples
        learning_rate: Learning rate
        devices: Number of GPUs to use
        strategy: Training strategy (auto/ddp/fsdp)
        precision: Training precision
        accumulate: Gradient accumulation steps
    """
    from astro_lab.data.config import data_config

    # Create temporary config in proper temp directory
    temp_dir = data_config.cache_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / "quick_train_config.yaml"

    # Create comprehensive config with all parameters
    temp_config = {
        "mlflow": {
            "experiment_name": f"quick_{model}_{dataset}",
            "tracking_uri": "file:./data/experiments/mlruns",
            "tags": {
                "mode": "quick_training",
                "dataset": dataset,
                "model": model,
                "version": "v1.0",
            },
        },
        "model": {
            "type": model,
            "params": {
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "conv_type": "gcn",
                "use_batch_norm": True,
                "activation": "relu",
                "pooling": "mean",
            },
        },
        "data": {
            "dataset": dataset,
            "data_root": str(data_config.base_dir),  # Explicitly set data root
            "batch_size": batch_size,
            "max_samples": max_samples,
            "k_neighbors": 8,
            "num_workers": None,  # Auto-detect for laptop
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "drop_last": True,
            "use_distributed_sampler": True,
            "return_tensor": True,
            "split_ratios": [0.7, 0.15, 0.15],
            # Laptop optimization parameters
            "max_nodes_per_graph": 1000,  # Conservative for RTX 4070
            "use_subgraph_sampling": True,
        },
        "training": {
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": 0.0001,
            "patience": 10,
            "devices": devices,
            "strategy": strategy,
            "precision": precision,
            "gradient_accumulation_steps": accumulate,
            "gradient_clip_val": 1.0,
            "gradient_clip_algorithm": "norm",
            "scheduler_type": "cosine",
            "warmup_steps": 0,
            "use_compile": True,
            "use_ema": False,
            "ema_decay": 0.999,
            "label_smoothing": 0.0,
            "deterministic": True,
            "benchmark": False,
            "log_every_n_steps": 50,
            "val_check_interval": 1.0,
        },
        "callbacks": {
            "early_stopping": {"monitor": "val_loss", "patience": 10, "mode": "min"},
            "model_checkpoint": {"monitor": "val_loss", "save_top_k": 3, "mode": "min"},
            "lr_monitor": {"logging_interval": "step"},
        },
    }

    # Save temporary config
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f, default_flow_style=False, indent=2)

    try:
        print("🚀 Starting quick training:")
        print(f"   Dataset: {dataset}")
        print(f"   Model: {model}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Max samples: {max_samples}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Devices: {devices}")
        print(f"   Strategy: {strategy}")
        print(f"   Precision: {precision}")
        print(f"   Gradient accumulation: {accumulate}")

        # Run training
        train_from_config(str(temp_config_path))
    finally:
        # Always clean up temp file
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="AstroLab Training - Train astronomical ML models with state-of-the-art techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with config file
  astro-lab train --config config.yaml
  
  # Quick training with defaults
  astro-lab train --dataset gaia --model gaia_classifier --epochs 10
  
  # Train with FSDP for large models
  astro-lab train --dataset gaia --model large_gaia --strategy fsdp --precision bf16
  
  # Train with gradient accumulation
  astro-lab train --dataset gaia --model gaia_classifier --accumulate 4
""",
    )

    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--dataset", help="Dataset for quick training")
    parser.add_argument("--model", help="Model for quick training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-samples", type=int, default=1000, help="Maximum number of samples"
    )
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--strategy", default="auto", help="Training strategy (auto/ddp/fsdp)"
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        help="Training precision (32/16-mixed/bf16-mixed)",
    )
    parser.add_argument(
        "--accumulate", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.config:
        train_from_config(args.config)
    elif args.dataset and args.model:
        train_quick(
            args.dataset,
            args.model,
            args.epochs,
            args.batch_size,
            args.max_samples,
            args.learning_rate,
            args.devices,
            args.strategy,
            args.precision,
            args.accumulate,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
