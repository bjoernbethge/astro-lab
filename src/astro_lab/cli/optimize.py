"""
AstroLab CLI Training Module
============================

Unified command-line interface for training astronomical models with robust
error handling, consistent logging, and comprehensive debugging capabilities.
Optimized for Lightning 2.0+ compatibility and modern ML practices.
"""

import gc
import logging
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import click
import torch
from yaml import dump

from astro_lab.data import create_astro_datamodule
from astro_lab.models.factory import ModelFactory
from astro_lab.training.trainer import AstroTrainer
from astro_lab.training.lightning_module import AstroLightningModule
from astro_lab.utils.config.loader import ConfigLoader

# Setup logging
logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_default_config(output_path: str = "config.yaml") -> None:
    """
    Create a minimal default configuration file.

    Args:
        output_path: Path where to save the configuration file
    """
    logger.info(f"🔧 Creating default configuration at: {output_path}")
    
    try:
        minimal_config = {
            "model": {
                "name": "gaia_classifier",
                "type": "AstroSurveyGNN",
                "params": {
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.1,
                },
            },
            "data": {
                "dataset": "gaia",
                "batch_size": 32,
                "max_samples": 5000,
                "return_tensor": True,
            },
            "training": {
                "max_epochs": 100,
                "learning_rate": 0.001,
                "experiment_name": "default_experiment",
            },
            "mlflow": {
                "experiment_name": "default_experiment",
                "tracking_uri": "file:./mlruns"
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            dump(minimal_config, f, default_flow_style=False, indent=2)

        logger.info(f"✅ Minimal configuration created: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create default config: {e}")
        raise

def ensure_mlflow_block(config: dict) -> dict:
    """Ensure that the config contains a valid mlflow block."""
    if "mlflow" not in config or not isinstance(config["mlflow"], dict):
        config["mlflow"] = {}
    if "experiment_name" not in config["mlflow"]:
        config["mlflow"]["experiment_name"] = config.get("training", {}).get("experiment_name", "default_experiment")
    if "tracking_uri" not in config["mlflow"]:
        config["mlflow"]["tracking_uri"] = "file:./mlruns"
    return config

def train_from_config(config_path: str) -> None:
    """
    Train model from configuration file using modern Lightning 2.0+ approach.

    Args:
        config_path: Path to YAML configuration file
    """
    try:
        logger.info(f"🚀 Starting training with config: {config_path}")
        
        # Validate config file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Use ConfigLoader for proper config handling
        loader = ConfigLoader(config_path)
        config = loader.load_config()
        config = ensure_mlflow_block(config)

        # Extract experiment name from config
        experiment_name = config.get("mlflow", {}).get("experiment_name", "astro_experiment")
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Experiment: {experiment_name}")

        # Create datamodule with error handling
        logger.info("🔧 Creating datamodule...")
        try:
            data_config_section = config.get("data", {})
            enhanced_data_config = enhance_data_config_for_tensors(data_config_section)

            dataset_name = enhanced_data_config["dataset"]
            dataset_params = {
                k: v for k, v in enhanced_data_config.items() if k != "dataset"
            }
            logger.info(f"📊 Loading dataset: {dataset_name}")
            datamodule = create_astro_datamodule(dataset_name, **dataset_params)
            logger.info(f"✅ Datamodule created successfully")

            # Automatische Klassenanzahl aus Trainingsdaten bestimmen
            train_loader = datamodule.train_dataloader() if hasattr(datamodule, 'train_dataloader') else None
            num_classes = None
            if train_loader is not None:
                targets = []
                logger.info("🔍 Starting automatic class detection from training data...")
                for i, batch in enumerate(train_loader):
                    t = None
                    if isinstance(batch, dict):
                        t = batch.get('target') or batch.get('y')
                    elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                        t = batch[1]
                    if t is not None:
                        targets.append(t.flatten())
                    if i > 5:  # Limit for efficiency
                        break
                if targets:
                    import torch
                    all_targets = torch.cat(targets)
                    num_classes = int(all_targets.max().item()) + 1
                    logger.info(f"✅ Detected {num_classes} classes from training data (min={all_targets.min().item()}, max={all_targets.max().item()})")
                else:
                    logger.warning("⚠️ Could not detect classes from training data")
            else:
                logger.warning("⚠️ No training dataloader available for class detection")

            # Update model config with detected classes
            if num_classes is not None:
                model_config = loader.get_model_config()
                if 'params' not in model_config:
                    model_config['params'] = {}
                model_config['params']['output_dim'] = num_classes
                logger.info(f"🔄 Updated model config with {num_classes} classes")

            # Create model with updated config
            model_config = loader.get_model_config()
            survey = model_config.get("survey", "gaia")
            task = model_config.get("task", "stellar_classification")
            model_params = model_config.get("params", {})

            # Use the detected num_classes if available
            if num_classes is not None and ('output_dim' not in model_params or model_params['output_dim'] is None):
                model_params['output_dim'] = num_classes
                logger.info(f"🔄 Set output_dim to {model_params['output_dim']} from detected classes")



            model = ModelFactory.create_survey_model(
                survey=survey, 
                task=task, 
                **model_params
            )
            logger.info(f"✅ Model created: {type(model).__name__}")
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise

        # Create Lightning module and trainer with modern approach
        logger.info("🔧 Creating Lightning module and trainer...")
        try:
            # Load training config with proper error handling
            training_config = loader.get_training_config()
            logger.info(f"📋 Training config loaded: {training_config}")
            
            # Create Lightning module with the model
            lightning_module = AstroLightningModule(
                model=model,
                task_type="classification",
                learning_rate=training_config.get("learning_rate", 0.001),
                weight_decay=training_config.get("weight_decay", 0.0),
                scheduler_config=training_config.get("scheduler", {}),
            )
            logger.info(f"✅ Lightning module created")

            # Create trainer with experiment name
            trainer = AstroTrainer(
                lightning_module=lightning_module,
                training_config=None,  # Use default config
                experiment_name=experiment_name,  # Use experiment name from config
                max_epochs=training_config.get("max_epochs", 100),
                accelerator="auto",
                devices="auto",
                precision="16-mixed",  # Use mixed precision for efficiency
                enable_progress_bar=True,
                enable_model_summary=True,
                enable_checkpointing=True,
                log_every_n_steps=50,
            )
            logger.info(f"✅ Trainer created")

            # Start training
            logger.info("🚀 Starting training...")
            trainer.fit(datamodule=datamodule)
            logger.info("✅ Training completed successfully")

            # Test the model
            logger.info("🧪 Testing model...")
            test_results = trainer.test(datamodule=datamodule)
            logger.info(f"✅ Testing completed: {test_results}")

            # Save best models
            logger.info("💾 Saving best models...")
            saved_models = trainer.save_best_models_to_results(top_k=3)
            logger.info(f"✅ Saved {len(saved_models)} models to results")

            # Get results summary
            results_summary = trainer.get_results_summary()
            logger.info(f"📊 Results summary: {results_summary}")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Memory cleanup
            _cleanup_memory()

    except Exception as e:
        logger.error(f"❌ Training process failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        logger.error(f"🔍 Error details: {str(e)}")
        raise

def _cleanup_memory():
    """Clean up memory to prevent leaks."""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 CUDA cache cleared")
        
        # Force garbage collection
        gc.collect()
        logger.info("🧹 Memory cleanup completed")
        
    except Exception as e:
        logger.warning(f"⚠️ Memory cleanup failed: {e}")

def optimize_from_config(
    config_path: str,
    n_trials: Optional[int] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """
    Optimize hyperparameters from configuration file.

    Args:
        config_path: Path to YAML configuration file
        n_trials: Number of optimization trials
        experiment_name: Name for the optimization experiment
    """
    try:
        logger.info(f"🔍 Starting hyperparameter optimization with config: {config_path}")
        
        # Load configuration
        loader = ConfigLoader(config_path)
        config = loader.load_config()
        config = ensure_mlflow_block(config)
        
        # Extract experiment name from config
        experiment_name = config.get("mlflow", {}).get("experiment_name", "astro_experiment")
        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Experiment: {experiment_name}")

        # Create datamodule
        data_config_section = config.get("data", {})
        enhanced_data_config = enhance_data_config_for_tensors(data_config_section)

        dataset_name = enhanced_data_config["dataset"]
        dataset_params = {
            k: v for k, v in enhanced_data_config.items() if k != "dataset"
        }
        
        datamodule = create_astro_datamodule(dataset_name, **dataset_params)

        # Create base model and trainer
        model_config = loader.get_model_config()
        survey = model_config.get("survey", "gaia")
        task = model_config.get("task", "stellar_classification")
        model_params = model_config.get("params", {})
        
        # Use params from config - they should already have the correct output_dim
        logger.info(f"📋 Model params from config: {model_params}")
        
        model = ModelFactory.create_survey_model(survey=survey, task=task, **model_params)
        
        training_config = loader.get_training_config()
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 1e-3),
            num_classes=model_params.get("output_dim", 8),
        )
        
        trainer = AstroTrainer(
            lightning_module=lightning_module,
            max_epochs=training_config.get("max_epochs", 100),
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
        )

        # Run optimization
        logger.info("🔍 Starting hyperparameter optimization...")
        results = trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            n_trials=n_trials or 50,
        )
        
        logger.info("🎉 Hyperparameter optimization completed!")
        logger.info(f"🔍 Best parameters: {results}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        if config is None:
            config = {}
        return config

def create_model_from_config(model_config: Dict[str, Any]) -> Any:
    """Create a model from a config dict."""
    model_type = model_config.get("type", "AstroSurveyGNN")
    params = model_config.get("params", {})
    if model_type == "AstroSurveyGNN":
        from astro_lab.models.astro import AstroSurveyGNN
        return AstroSurveyGNN(**params)
    elif model_type == "AstroPhotGNN":
        from astro_lab.models.astrophot_models import AstroPhotGNN
        return AstroPhotGNN(**params)
    elif model_type == "ALCDEFTemporalGNN":
        from astro_lab.models.tgnn import ALCDEFTemporalGNN
        return ALCDEFTemporalGNN(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def enhance_data_config_for_tensors(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance data configuration for tensor support."""
    enhanced = data_config.copy()
    
    # Ensure tensor support is enabled
    enhanced["return_tensor"] = True
    
    # Set reasonable defaults
    if "batch_size" not in enhanced:
        enhanced["batch_size"] = 32
    if "max_samples" not in enhanced:
        enhanced["max_samples"] = 5000
        
    return enhanced

def train_model(
    config_path: str, optimize: bool = False, verbose: bool = False
) -> None:
    """
    Main training function with unified error handling.
    
    Args:
        config_path: Path to configuration file
        optimize: Whether to run hyperparameter optimization
        verbose: Whether to enable verbose logging
    """
    try:
        # Setup logging
        setup_logging(verbose=verbose)
        
        # Validate config file
        if not Path(config_path).exists():
            logger.error(f"❌ Configuration file not found: {config_path}")
            logger.info("💡 Use --create-config to generate a default configuration")
            return
        
        # Run training or optimization
        if optimize:
            logger.info("🔍 Running hyperparameter optimization...")
            optimize_from_config(config_path)
        else:
            logger.info("🚀 Running training...")
            train_from_config(config_path)
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

@click.group()
def cli():
    """AstroLab CLI - Astronomical Machine Learning Training Interface."""
    pass

@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--optimize-first", "-o", is_flag=True, help="Run hyperparameter optimization before training")
@click.option("--n-trials", "-n", default=20, help="Number of optimization trials (if optimize-first is used)")
@click.option("--auto-optimize", "-a", is_flag=True, help="Automatically optimize if no best parameters found")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def run(config: str, optimize_first: bool, n_trials: int, auto_optimize: bool, verbose: bool):
    """
    Run the complete ML workflow: optimize (optional) -> train -> evaluate.
    
    This is the recommended approach for most use cases.
    """
    try:
        setup_logging(verbose=verbose)
        logger.info("🚀 Starting complete ML workflow...")

    # Load configuration
        loader = ConfigLoader(config)
        config_dict = loader.load_config()
        config_dict = ensure_mlflow_block(config_dict)
        
        # Check if we should optimize first
        if optimize_first:
            logger.info("🔍 Step 1: Running hyperparameter optimization...")
            best_params = run_optimization(loader, config_dict, n_trials)
            
            # Update config with best parameters
            if best_params:
                logger.info("🔄 Updating config with best parameters...")
                update_config_with_best_params(config, best_params)
                # Reload config with updated parameters
                loader = ConfigLoader(config)
                config_dict = loader.load_config()
        
        # Auto-optimize if no good parameters found
        elif auto_optimize and should_optimize(config_dict):
            logger.info("🔍 Auto-detected need for optimization...")
            best_params = run_optimization(loader, config_dict, n_trials)
            if best_params:
                update_config_with_best_params(config, best_params)
                loader = ConfigLoader(config)
                config_dict = loader.load_config()
        
        # Run training
        logger.info("🚀 Step 2: Running training...")
        train_from_config(config)
        
        logger.info("🎉 Complete workflow finished successfully!")
        
    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def train(config: str, verbose: bool):
    """Train a model using the specified configuration (training only)."""
    train_model(config, optimize=False, verbose=verbose)

@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--n-trials", "-n", default=50, help="Number of optimization trials")
@click.option("--update-config", "-u", is_flag=True, help="Update config file with best parameters")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def optimize(config: str, n_trials: int, update_config: bool, verbose: bool):
    """Optimize hyperparameters using the specified configuration (optimization only)."""
    try:
        setup_logging(verbose=verbose)
        loader = ConfigLoader(config)
        config_dict = loader.load_config()
        config_dict = ensure_mlflow_block(config_dict)
        
        best_params = run_optimization(loader, config_dict, n_trials)
        
        if update_config and best_params:
            logger.info("🔄 Updating config file with best parameters...")
            update_config_with_best_params(config, best_params)
            logger.info(f"✅ Config updated: {config}")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        if verbose:
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option("--output", "-o", default="config.yaml", help="Output path for configuration")
def create_config(output: str):
    """Create a default configuration file."""
    try:
        create_default_config(output)
        logger.info(f"✅ Configuration created at: {output}")
        logger.info("💡 Edit the configuration file and run 'train' command")
    except Exception as e:
        logger.error(f"❌ Failed to create configuration: {e}")
        sys.exit(1)

def should_optimize(config: Dict[str, Any]) -> bool:
    """Check if optimization should be run based on config."""
    # Check if config has default/untuned parameters
    model_params = config.get("model", {}).get("params", {})
    
    # If learning rate is default, probably needs optimization
    if model_params.get("learning_rate") in [0.001, 1e-3, None]:
        return True
    
    # If hidden_dim is default, probably needs optimization
    if model_params.get("hidden_dim") in [128, 64, None]:
        return True
    
    # If dropout is default, probably needs optimization
    if model_params.get("dropout") in [0.1, 0.2, None]:
        return True
    
    return False

def run_optimization(loader: ConfigLoader, config: Dict[str, Any], n_trials: int) -> Optional[Dict[str, Any]]:
    """Run hyperparameter optimization and return best parameters."""
    try:
        logger.info(f"🔍 Starting optimization with {n_trials} trials...")
        
        # Create datamodule
        data_config_section = config.get("data", {})
        enhanced_data_config = enhance_data_config_for_tensors(data_config_section)
        
        dataset_name = enhanced_data_config["dataset"]
        dataset_params = {
            k: v for k, v in enhanced_data_config.items() if k != "dataset"
        }

        datamodule = create_astro_datamodule(dataset_name, **dataset_params)

        # Create base model and trainer
        model_config = loader.get_model_config()
        survey = model_config.get("survey", "gaia")
        task = model_config.get("task", "stellar_classification")
        model_params = model_config.get("params", {})
        
        # Use params from config - they should already have the correct output_dim
        logger.info(f"📋 Model params from config: {model_params}")
        
        model = ModelFactory.create_survey_model(survey=survey, task=task, **model_params)
        
        training_config = loader.get_training_config()
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 1e-3),
            num_classes=model_params.get("output_dim", 8),
        )
        
        trainer = AstroTrainer(
            lightning_module=lightning_module,
            max_epochs=training_config.get("max_epochs", 100),
            accelerator="auto",
            devices="auto",
            precision="16-mixed",
        )
        
        # Run optimization
        results = trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            n_trials=n_trials,
        )
        
        logger.info("🎉 Optimization completed!")
        logger.info(f"🔍 Best parameters: {results}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return None

def update_config_with_best_params(config_path: str, best_params: Dict[str, Any]) -> None:
    """Update configuration file with best parameters from optimization."""
    try:
        # Load current config
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Update model parameters
        if "model" not in config:
            config["model"] = {}
        if "params" not in config["model"]:
            config["model"]["params"] = {}
        
        # Update with best parameters
        for key, value in best_params.items():
            if key in ["learning_rate", "hidden_dim", "dropout", "num_layers"]:
                config["model"]["params"][key] = value
        
        # Save updated config
        with open(config_path, "w", encoding="utf-8") as f:
            dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Config updated with best parameters: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        raise

def main():
    """Main CLI entry point."""
    cli()

if __name__ == "__main__":
    main()
