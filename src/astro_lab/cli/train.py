#!/usr/bin/env python3
"""
AstroLab Training CLI - Simple Command Interface
===============================================

Minimal CLI that only parses arguments and delegates to trainer classes.
All logging is handled by the trainer and lightning modules.
"""

import argparse
import sys
import os
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
import io
import contextlib

# Suppress import-time messages
with contextlib.redirect_stderr(io.StringIO()):
    import yaml
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

def train_from_config(config_path: str) -> None:
    """
    Train from configuration file - delegates to AstroTrainer.
    
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
    from astro_lab.data.datamodule import AstroDataModule
    from astro_lab.models.factory import ModelFactory
    from astro_lab.training.lightning_module import AstroLightningModule
    from astro_lab.data.config import data_config
    
    # Create a custom stderr that filters duplicate messages
    class FilteredStderr:
        def __init__(self):
            self.seen_messages = set()
            self.original_stderr = sys.stderr
            
        def write(self, msg):
            # Filter duplicate Lightning messages
            if msg.strip() and msg not in self.seen_messages:
                if any(x in msg for x in ["GPU available", "TPU available", "HPU available", "Using 16bit"]):
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
        experiment_name = config.get("mlflow", {}).get("experiment_name", "astro_training")
        data_config.ensure_experiment_directories(experiment_name)
        
        # Extract data config
        data_config_section = config.get("data", {})
        dataset_name = data_config_section.get("dataset", "gaia")
        batch_size = data_config_section.get("batch_size", 32)
        max_samples = data_config_section.get("max_samples", 1000)
        k_neighbors = data_config_section.get("k_neighbors", 8)
        
        # Create datamodule
        datamodule = AstroDataModule(
            survey=dataset_name,
            batch_size=batch_size,
            max_samples=max_samples,
            k_neighbors=k_neighbors,
        )
        
        # Extract model config
        model_config = config.get("model", {})
        model_type = model_config.get("type", "gaia_classifier")
        model_params = model_config.get("params", {})
        
        # Auto-detect classes if needed (silently)
        if "output_dim" not in model_params or model_params["output_dim"] is None:
            try:
                train_loader = datamodule.train_dataloader()
                targets = []
                
                for i, batch in enumerate(train_loader):
                    # Handle PyTorch Geometric Data objects
                    if hasattr(batch, 'y'):
                        t = batch.y
                    elif isinstance(batch, list) and len(batch) > 0 and hasattr(batch[0], 'y'):
                        t = batch[0].y
                    elif isinstance(batch, dict):
                        t = batch.get("target") or batch.get("y")
                    elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                        t = batch[1]
                    else:
                        t = None
                        
                    if t is not None:
                        targets.append(t.flatten())
                    if i > 5:
                        break
                
                if targets:
                    import torch
                    all_targets = torch.cat(targets)
                    num_classes = int(all_targets.max().item()) + 1
                    model_params["output_dim"] = num_classes
                    print(f"Auto-detected {num_classes} classes from data")
                else:
                    model_params["output_dim"] = 4
                    print("Using default 4 classes")
            except:
                model_params["output_dim"] = 4
                print("Using default 4 classes")
        
        # Create model
        if model_type == "gaia_classifier":
            model = ModelFactory.create_survey_model(
                survey="gaia",
                task="stellar_classification",
                data_loader=datamodule.train_dataloader(),
                **model_params
            )
        elif model_type == "sdss_galaxy_classifier":
            model = ModelFactory.create_survey_model(
                survey="sdss",
                task="galaxy_classification",
                data_loader=datamodule.train_dataloader(),
                **model_params
            )
        else:
            from astro_lab.models.astro import AstroSurveyGNN
            model = AstroSurveyGNN(**model_params)
        
        print(f"Created model: {type(model).__name__}")
        
        # Create Lightning module
        training_config = config.get("training", {})
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
        
        # Setup callbacks
        callbacks = []
        callbacks.append(EarlyStopping(
            monitor="val_loss",
            patience=training_config.get("patience", 10),
            mode="min",
            verbose=False  # Disable verbose
        ))
        
        checkpoint_dir = data_config.checkpoints_dir / experiment_name
        callbacks.append(ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename=f"{experiment_name}_{{epoch:02d}}_{{val_loss:.4f}}",
            verbose=False  # Disable verbose
        ))
        
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        
        # Setup MLflow logger if configured
        logger_instance = None
        if config.get("mlflow", {}).get("tracking_uri"):
            try:
                from lightning.pytorch.loggers import MLFlowLogger
                logger_instance = MLFlowLogger(
                    experiment_name=experiment_name,
                    tracking_uri=config["mlflow"]["tracking_uri"],
                    log_model=False  # Disable model logging to reduce clutter
                )
            except:
                pass
        
        # Create and run trainer with minimal logging
        trainer = Trainer(
            max_epochs=training_config.get("max_epochs", 20),
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            callbacks=callbacks,
            logger=logger_instance,
            gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True,
            log_every_n_steps=50,  # Reduce logging frequency
        )
        
        print(f"Starting training for {training_config.get('max_epochs', 20)} epochs...")
        
        # Train
        trainer.fit(lightning_module, datamodule=datamodule)
        
        print("Training completed!")
        
        # Test
        trainer.test(lightning_module, datamodule=datamodule)
        
        # Save results
        results_dir = data_config.results_dir / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get best checkpoint path from callback
        checkpoint_callback = None
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        model_info = {
            "experiment_name": experiment_name,
            "model_type": type(model).__name__,
            "model_params": model_params,
            "training_config": training_config,
            "best_model_path": str(checkpoint_callback.best_model_path) if checkpoint_callback else None,
        }
        
        with open(results_dir / "training_summary.yaml", "w") as f:
            yaml.dump(model_info, f, default_flow_style=False, indent=2)
        
        print(f"Results saved to: {results_dir}")
        
        # Cleanup
        import torch
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

def train_quick(dataset: str, model: str, epochs: int = 10, batch_size: int = 32) -> None:
    """
    Quick training without config file.
    
    Args:
        dataset: Dataset name (gaia, sdss, nsa)
        model: Model type (gaia_classifier, etc.)
        epochs: Number of epochs
        batch_size: Batch size
    """
    from astro_lab.data.config import data_config
    
    # Create temporary config in proper temp directory
    temp_dir = data_config.cache_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config_path = temp_dir / "quick_train_config.yaml"
    
    temp_config = {
        "mlflow": {
            "experiment_name": f"quick_{model}_{dataset}",
            "tracking_uri": "./data/experiments"
        },
        "model": {
            "type": model,
            "params": {
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2
            }
        },
        "data": {
            "dataset": dataset,
            "batch_size": batch_size,
            "max_samples": 1000
        },
        "training": {
            "max_epochs": epochs,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "patience": 5
        }
    }
    
    # Save temporary config
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f, default_flow_style=False, indent=2)
    
    try:
        # Run training
        train_from_config(str(temp_config_path))
    finally:
        # Always clean up temp file
        if temp_config_path.exists():
            temp_config_path.unlink()

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="AstroLab Training")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--dataset", help="Dataset for quick training")
    parser.add_argument("--model", help="Model for quick training") 
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.config:
        train_from_config(args.config)
    elif args.dataset and args.model:
        train_quick(args.dataset, args.model, args.epochs, args.batch_size)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main() 