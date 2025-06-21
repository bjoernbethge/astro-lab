"""
AstroLab CLI Training Module
============================

Unified command-line interface for training astronomical models with robust
error handling, consistent logging, and comprehensive debugging capabilities.
Optimized for Lightning 2.0+ compatibility and modern ML practices.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml
from yaml import dump as yaml_dump

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
            yaml_dump(minimal_config, f, default_flow_style=False, indent=2)

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

        logger.info(f"✅ Configuration loaded successfully")
        logger.info(f"   Experiment: {config['mlflow']['experiment_name']}")

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
            
            # Ensure output_dim is set
            if 'output_dim' not in model_params or model_params['output_dim'] is None:
                model_params['output_dim'] = num_classes or 8  # Fallback
                logger.info(f"🔄 Set output_dim to {model_params['output_dim']}")
            
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
                learning_rate=training_config.get("learning_rate", 1e-3),
                num_classes=num_classes,
            )
            logger.info("✅ Lightning module created")

            # Get survey name for better organization
            survey = model_config.get("survey", "gaia")
            
            # Create trainer with proper defaults and organization
            max_epochs = training_config.get("max_epochs", 100)
            logger.info(f"🎯 Setting max_epochs to: {max_epochs}")
            
            trainer = AstroTrainer(
                lightning_module=lightning_module,
                max_epochs=max_epochs,
                accelerator="auto",
                devices="auto",
                precision="16-mixed",
            )
            logger.info("✅ Trainer created successfully")
            
        except Exception as e:
            logger.error(f"❌ Trainer creation failed: {e}")
            raise

        # Train with comprehensive error handling
        logger.info("🎯 Starting training...")
        try:
            trainer.fit(datamodule=datamodule)
            logger.info("🎉 Training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"🔍 Training error details: {type(e).__name__}: {str(e)}")
            raise

        # Automatically organize results
        logger.info("📊 Organizing results...")
        try:
            saved_models = trainer.save_best_models_to_results(top_k=3)
            results_summary = trainer.get_results_summary()
            logger.info(f"📁 Results saved to: {results_summary['results_structure']['base']}")

        except Exception as e:
            logger.warning(f"⚠️ Could not organize results: {e}")
            logger.warning("Results organization is optional, training was successful")

    except Exception as e:
        logger.error(f"❌ Training process failed: {e}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        logger.error(f"🔍 Error details: {str(e)}")
        raise


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
        model = ModelFactory.create_survey_model(survey=survey, task=task, **model_config.get("params", {}))
        
        training_config = loader.get_training_config()
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=training_config.get("learning_rate", 1e-3),
            num_classes=model_config.get("params", {}).get("output_dim", 8),
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
    """Create model from configuration."""
    try:
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
            
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        raise


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
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.group()
def cli():
    """AstroLab CLI - Astronomical Machine Learning Training Interface."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def train(config: str, verbose: bool):
    """Train a model using the specified configuration."""
    train_model(config, optimize=False, verbose=verbose)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--n-trials", "-n", default=50, help="Number of optimization trials")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def optimize(config: str, n_trials: int, verbose: bool):
    """Optimize hyperparameters using the specified configuration."""
    try:
        setup_logging(verbose=verbose)
        optimize_from_config(config, n_trials=n_trials)
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        if verbose:
            import traceback
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


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
