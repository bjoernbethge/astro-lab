#!/usr/bin/env python3
"""
AstroLab Training CLI - Simple and Focused
==========================================

Clean training interface without optimization complexity.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        raise

def train_from_config(config_path: str) -> None:
    """
    Simple training from config file.
    
    Args:
        config_path: Path to YAML configuration file
    """
    logger.info(f"🚀 Starting training with config: {config_path}")
    
    try:
        # Load config
        config = load_config(config_path)
        
        # Get experiment name
        experiment_name = config.get("mlflow", {}).get("experiment_name", "astro_training")
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Experiment: {experiment_name}")
        
        # Setup experiment directories
        from astro_lab.data.config import data_config
        data_config.ensure_experiment_directories(experiment_name)
        
        # Create datamodule
        logger.info("🔧 Creating datamodule...")
        from astro_lab.data.datamodule import AstroDataModule
        
        data_config_section = config.get("data", {})
        dataset_name = data_config_section.get("dataset", "gaia")
        
        # Extract datamodule parameters
        batch_size = data_config_section.get("batch_size", 32)
        max_samples = data_config_section.get("max_samples", 1000)
        k_neighbors = data_config_section.get("k_neighbors", 8)
        
        logger.info(f"📊 Loading dataset: {dataset_name}")
        datamodule = AstroDataModule(
            survey=dataset_name,
            batch_size=batch_size,
            max_samples=max_samples,
            k_neighbors=k_neighbors,
        )
        logger.info(f"✅ Datamodule created successfully")
        
        # Create model
        logger.info("🔧 Creating model...")
        from astro_lab.models.factory import ModelFactory
        
        model_config = config.get("model", {})
        model_type = model_config.get("type", "gaia_classifier")
        model_params = model_config.get("params", {})
        
        # Auto-detect classes if needed
        if "output_dim" not in model_params or model_params["output_dim"] is None:
            try:
                train_loader = datamodule.train_dataloader()
                targets = []
                logger.info("🔍 Auto-detecting classes from training data...")
                
                for i, batch in enumerate(train_loader):
                    t = None
                    if isinstance(batch, dict):
                        t = batch.get("target") or batch.get("y")
                    elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                        t = batch[1]
                    elif hasattr(batch, "y"):
                        t = batch.y
                        
                    if t is not None:
                        targets.append(t.flatten())
                    if i > 5:  # Limit for efficiency
                        break
                
                if targets:
                    import torch
                    all_targets = torch.cat(targets)
                    num_classes = int(all_targets.max().item()) + 1
                    model_params["output_dim"] = num_classes
                    logger.info(f"✅ Detected {num_classes} classes")
                else:
                    model_params["output_dim"] = 4  # Default fallback
                    logger.warning("⚠️ Could not detect classes, using default: 4")
            except Exception as e:
                model_params["output_dim"] = 4  # Safe fallback
                logger.warning(f"⚠️ Class detection failed: {e}, using default: 4")
        
        # Create model using factory
        if model_type == "gaia_classifier":
            model = ModelFactory.create_survey_model(survey="gaia", task="stellar_classification", **model_params)
        elif model_type == "sdss_galaxy_classifier":
            model = ModelFactory.create_survey_model(survey="sdss", task="galaxy_classification", **model_params)
        else:
            # Generic model creation
            from astro_lab.models.astro import AstroSurveyGNN
            model = AstroSurveyGNN(**model_params)
            
        logger.info(f"✅ Model created: {type(model).__name__}")
        
        # Create Lightning module
        logger.info("🔧 Creating Lightning module...")
        from astro_lab.training.lightning_module import AstroLightningModule
        
        training_config = config.get("training", {})
        
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
        )
        logger.info("✅ Lightning module created")
        
        # Create trainer
        logger.info("🔧 Creating trainer...")
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=training_config.get("patience", 10),
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint
        checkpoint_dir = data_config.checkpoints_dir / experiment_name
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            filename=f"{experiment_name}_epoch_{{epoch:02d}}_loss_{{val_loss:.4f}}",
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        # Setup MLflow logger
        logger_instance = None
        if config.get("mlflow", {}).get("tracking_uri"):
            try:
                from lightning.pytorch.loggers import MLFlowLogger
                logger_instance = MLFlowLogger(
                    experiment_name=experiment_name,
                    tracking_uri=config["mlflow"]["tracking_uri"]
                )
                logger.info("✅ MLflow logger configured")
            except Exception as e:
                logger.warning(f"⚠️ MLflow logger failed: {e}")
        
        # Create trainer (Lightning 2.0 compatible)
        trainer = Trainer(
            max_epochs=training_config.get("max_epochs", 20),
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
            callbacks=callbacks,
            logger=logger_instance,
            gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
            # Removed deprecated Lightning 2.0 parameters
        )
        logger.info("✅ Trainer created")
        
        # Start training
        logger.info("🚀 Starting training...")
        logger.info(f"   Model: {type(model).__name__}")
        logger.info(f"   Task: classification")
        logger.info(f"   Device: {trainer.device_ids if trainer.device_ids else 'auto'}")
        logger.info(f"   Classes: {model_params.get('output_dim', 'unknown')}")
        
        trainer.fit(lightning_module, datamodule=datamodule)
        logger.info("✅ Training completed successfully")
        
        # Test the model
        logger.info("🧪 Testing model...")
        test_results = trainer.test(lightning_module, datamodule=datamodule)
        logger.info(f"✅ Testing completed: {test_results}")
        
        # Save model info
        results_dir = data_config.results_dir / experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        model_info = {
            "experiment_name": experiment_name,
            "model_type": type(model).__name__,
            "model_params": model_params,
            "training_config": training_config,
            "test_results": test_results,
            "best_model_path": str(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else None,
            "last_model_path": str(checkpoint_callback.last_model_path) if checkpoint_callback.last_model_path else None,
        }
        
        with open(results_dir / "training_summary.yaml", "w") as f:
            yaml.dump(model_info, f, default_flow_style=False, indent=2)
        
        logger.info(f"📊 Results saved to: {results_dir}")
        logger.info(f"📊 Best model: {checkpoint_callback.best_model_path}")
        
        # Complete memory cleanup
        import torch
        import gc
        import psutil
        import os
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # CPU cleanup
        gc.collect()
        
        # Process memory cleanup
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            logger.info(f"🧹 Memory cleanup completed - RSS: {memory_info.rss / 1024**2:.1f} MB")
        except:
            logger.info("🧹 Memory cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        raise

def train_quick(dataset: str, model: str, epochs: int = 10, batch_size: int = 32) -> None:
    """
    Quick training without config file.
    
    Args:
        dataset: Dataset name (gaia, sdss, nsa)
        model: Model type (gaia_classifier, etc.)
        epochs: Number of epochs
        batch_size: Batch size
    """
    logger.info(f"🚀 Quick training: {dataset} + {model}")
    
    # Create temporary config
    temp_config = {
        "mlflow": {
            "experiment_name": f"quick_{model}_{dataset}",
            "tracking_uri": "./mlruns"
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
    temp_config_path = "temp_quick_config.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(temp_config, f, default_flow_style=False, indent=2)
    
    try:
        # Run training
        train_from_config(temp_config_path)
    finally:
        # Clean up temp file
        Path(temp_config_path).unlink(missing_ok=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AstroLab Training CLI")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--dataset", help="Dataset for quick training")
    parser.add_argument("--model", help="Model for quick training") 
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        if args.config:
            train_from_config(args.config)
        elif args.dataset and args.model:
            train_quick(args.dataset, args.model, args.epochs, args.batch_size)
        else:
            logger.error("❌ Provide either --config or --dataset + --model")
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1) 