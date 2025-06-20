"""
AstroLab Training CLI - Streamlined Interface with Full Tensor Integration
=========================================================================

Schlanke moderne Training-Schnittstelle mit YAML-Konfiguration.
Fokus auf Einfachheit und Performance mit Lightning DataModules.
🌟 Enhanced with native SurveyTensor support for optimal performance.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_lab.models.utils import create_gaia_classifier
from astro_lab.training import AstroTrainer, create_astro_datamodule
from astro_lab.utils.config_loader import ConfigLoader, load_experiment_config

# 🌟 NEW: Import tensor-native models
try:
    from astro_lab.models import AstroSurveyGNN
    from astro_lab.tensors import SurveyTensor

    TENSOR_MODELS_AVAILABLE = True
except ImportError:
    TENSOR_MODELS_AVAILABLE = False
    AstroSurveyGNN = None
    SurveyTensor = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_config(output_path: str = "config.yaml") -> None:
    """
    Create a default configuration file using ConfigLoader.

    Args:
        output_path: Path where to save the config file
    """
    try:
        # Use ConfigLoader to load default config
        loader = ConfigLoader("configs/default.yaml")
        config = loader.load_config("default_experiment")

        # Save to specified path
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"✅ Default configuration created: {output_path}")
        print("🚀 You can now edit this file and run:")
        print(f"   astro-lab train train --config {output_path}")

    except Exception as e:
        logger.error(f"❌ Failed to create default config: {e}")
        # Fallback: create minimal config
        minimal_config = {
            "model": {
                "type": "gaia_classifier",
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.1,
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
        }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(minimal_config, f, default_flow_style=False, indent=2)

        print(f"✅ Minimal configuration created: {output_path}")


def train_from_config(config_path: str) -> None:
    """
    Train model from configuration file using ConfigLoader.

    Args:
        config_path: Path to YAML configuration file
    """
    try:
        # Use ConfigLoader for proper config handling
        loader = ConfigLoader(config_path)
        config = loader.load_config()

        logger.info(f"🚀 Training with config: {config_path}")
        logger.info(f"   Experiment: {config['mlflow']['experiment_name']}")

        # Create model
        model_config = loader.get_model_config()
        model = create_model_from_config(model_config)

        # Create datamodule
        data_config = config["data"]
        enhanced_data_config = enhance_data_config_for_tensors(data_config)

        dataset_name = enhanced_data_config["dataset"]
        dataset_params = {
            k: v for k, v in enhanced_data_config.items() if k != "dataset"
        }
        datamodule = create_astro_datamodule(dataset_name, **dataset_params)

        # Create trainer with full config
        training_config = loader.get_training_config()
        mlflow_config = loader.get_mlflow_config()

        # Merge configs for trainer - exclude parameters that are not for Lightning Trainer
        trainer_config = {
            k: v
            for k, v in training_config.items()
            if k not in ["learning_rate", "tracking_uri"]
        }
        trainer_config["experiment_name"] = mlflow_config["experiment_name"]

        # Use centralized parameter distribution
        from astro_lab.utils.config_params import (
            get_lightning_params,
            get_mlflow_params,
            get_trainer_params,
            validate_parameter_conflicts,
        )

        # Validate configuration first
        is_valid, error_msg = validate_parameter_conflicts(config)
        if not is_valid:
            raise ValueError(f"❌ Configuration error: {error_msg}")

        # Get properly distributed parameters
        trainer_config = get_trainer_params(config)
        lightning_params = get_lightning_params(config)
        mlflow_params = get_mlflow_params(config)

        # Add experiment name to trainer config
        trainer_config["experiment_name"] = mlflow_params["experiment_name"]

        # Wrap model in Lightning module if needed
        from astro_lab.training.lightning_module import AstroLightningModule

        if not isinstance(model, AstroLightningModule):
            lightning_module = AstroLightningModule(model=model, **lightning_params)
        else:
            lightning_module = model

        trainer = AstroTrainer(lightning_module=lightning_module, **trainer_config)

        # Train
        logger.info("🎯 Starting training...")
        trainer.fit(datamodule=datamodule)

        logger.info("🎉 Training completed!")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


def optimize_from_config(config_path: str) -> None:
    """
    Run hyperparameter optimization from configuration file.

    Args:
        config_path: Path to YAML configuration file
    """
    try:
        # Use ConfigLoader for proper config handling
        loader = ConfigLoader(config_path)
        config = loader.load_config()

        logger.info(f"🎯 Optimizing with config: {config_path}")

        # Create model
        model_config = loader.get_model_config()
        model = create_model_from_config(model_config)

        # Create datamodule
        data_config = config["data"]
        enhanced_data_config = enhance_data_config_for_tensors(data_config)

        dataset_name = enhanced_data_config["dataset"]
        dataset_params = {
            k: v for k, v in enhanced_data_config.items() if k != "dataset"
        }
        datamodule = create_astro_datamodule(dataset_name, **dataset_params)

        # Use centralized parameter distribution
        from astro_lab.utils.config_params import (
            get_lightning_params,
            get_mlflow_params,
            get_trainer_params,
            validate_parameter_conflicts,
        )

        # Validate configuration first
        is_valid, error_msg = validate_parameter_conflicts(config)
        if not is_valid:
            raise ValueError(f"❌ Configuration error: {error_msg}")

        # Get properly distributed parameters
        trainer_config = get_trainer_params(config)
        lightning_params = get_lightning_params(config)
        mlflow_params = get_mlflow_params(config)

        # Add experiment name to trainer config
        trainer_config["experiment_name"] = mlflow_params["experiment_name"]

        # Wrap model in Lightning module if needed
        from astro_lab.training.lightning_module import AstroLightningModule

        if not isinstance(model, AstroLightningModule):
            lightning_module = AstroLightningModule(model=model, **lightning_params)
        else:
            lightning_module = model

        trainer = AstroTrainer(lightning_module=lightning_module, **trainer_config)

        # Run optimization
        logger.info("🎯 Starting hyperparameter optimization...")
        optimization_config = config.get("optimization", {})

        # Use centralized parameter distribution for optimization
        from astro_lab.utils.config_params import get_optuna_params

        # Get properly filtered Optuna parameters
        optuna_params = get_optuna_params(optimization_config)
        search_space = optimization_config.get("search_space", None)

        best_params = trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            search_space=search_space,
            **optuna_params,
        )

        logger.info(f"✨ Best parameters: {best_params}")

    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Empty or invalid configuration file: {config_path}")
        return config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_model_from_config(model_config: Dict[str, Any]) -> Any:
    """
    Create model from configuration with tensor-native support.

    🌟 Enhanced to automatically choose tensor-optimized models when available.
    """
    model_type = model_config.get("type", "gaia_classifier")
    model_params = model_config.get("params", {})
    use_tensors = model_config.get("use_tensors", True)  # 🌟 NEW: Enable by default

    # 🌟 TENSOR-NATIVE MODEL SELECTION
    if use_tensors and TENSOR_MODELS_AVAILABLE:
        if model_type in ["gaia_classifier", "survey_gnn", "astro_gnn"]:
            logger.info("🌟 Using tensor-native AstroSurveyGNN model")
            # Enhanced parameters for tensor models
            tensor_params = {
                "hidden_dim": model_params.get("hidden_dim", 128),
                "output_dim": model_params.get("output_dim", 1),
                "conv_type": model_params.get("conv_type", "gcn"),
                "num_layers": model_params.get("num_layers", 3),
                "dropout": model_params.get("dropout", 0.1),
                "task": model_params.get("task", "node_classification"),
                "use_photometry": model_params.get("use_photometry", True),
                "use_astrometry": model_params.get("use_astrometry", True),
                "use_spectroscopy": model_params.get("use_spectroscopy", False),
                "pooling": model_params.get("pooling", "mean"),
            }
            return AstroSurveyGNN(**tensor_params)

    # Fallback to legacy models
    if model_type == "gaia_classifier":
        logger.info("📊 Using legacy Gaia classifier")
        return create_gaia_classifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def enhance_data_config_for_tensors(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance data configuration to enable tensor usage by default.

    🌟 Automatically optimizes data loading for tensor-native workflows.
    """
    enhanced_config = data_config.copy()

    # Enable tensor conversion by default if not specified
    if "return_tensor" not in enhanced_config:
        enhanced_config["return_tensor"] = True
        logger.info("🌟 Enabled tensor conversion for optimal performance")

    # Set reasonable defaults for tensor usage
    if "max_samples" not in enhanced_config:
        enhanced_config["max_samples"] = 5000
        logger.info(f"📊 Set default max_samples to {enhanced_config['max_samples']}")

    return enhanced_config


def train_model(
    config_path: str, optimize: bool = False, verbose: bool = False
) -> None:
    """Train model with given configuration."""
    setup_logging(verbose)
    logger.info(f"🚀 Loading configuration from {config_path}")

    # Load configuration
    config = load_config(config_path)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]

    # 🌟 ENHANCE DATA CONFIG FOR TENSORS
    enhanced_data_config = enhance_data_config_for_tensors(data_config)

    # Create model with tensor support
    logger.info("🏗️ Creating model...")
    model = create_model_from_config(model_config)

    # Create datamodule with enhanced configuration
    logger.info("📊 Setting up data pipeline...")
    dataset_name = enhanced_data_config["dataset"]
    dataset_params = {k: v for k, v in enhanced_data_config.items() if k != "dataset"}

    # 🌟 Enhanced datamodule creation with tensor support
    datamodule = create_astro_datamodule(dataset_name, **dataset_params)

    # Log tensor integration status
    if hasattr(datamodule, "dataset") and hasattr(datamodule.dataset, "return_tensor"):
        if datamodule.dataset.return_tensor:
            logger.info("✨ Tensor-native data pipeline activated!")
        else:
            logger.info("📊 Using legacy PyG data pipeline")

    # Create trainer
    logger.info("⚡ Initializing trainer...")
    # Wrap model in Lightning module if needed
    from astro_lab.training.lightning_module import AstroLightningModule
    from astro_lab.utils.config_params import get_lightning_params

    if not isinstance(model, AstroLightningModule):
        lightning_params = get_lightning_params(config)
        lightning_module = AstroLightningModule(model=model, **lightning_params)
    else:
        lightning_module = model

    # Use centralized parameter distribution
    from astro_lab.utils.config_params import get_trainer_params

    trainer_params = get_trainer_params(config)
    trainer_params["experiment_name"] = config.get("mlflow", {}).get(
        "experiment_name", "astro_experiment"
    )

    trainer = AstroTrainer(lightning_module=lightning_module, **trainer_params)

    if optimize:
        # Hyperparameter optimization
        logger.info("🎯 Starting hyperparameter optimization...")
        optimization_config = config.get("optimization", {})
        best_params = trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            **optimization_config,
        )
        logger.info(f"✨ Best parameters: {best_params}")
    else:
        # Standard training
        logger.info("🎯 Starting training...")
        trainer.fit(datamodule=datamodule)

        # Optional: Run testing
        if training_config.get("run_test", False):
            logger.info("🧪 Running test evaluation...")
            trainer.test(datamodule=datamodule)

    logger.info("🎉 Training completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AstroLab Model Training with Tensor Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional argument - config file
    parser.add_argument("config", type=str, help="Path to YAML configuration file")

    # Optional flags
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization instead of training",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # 🌟 NEW: Tensor control flags
    parser.add_argument(
        "--disable-tensors",
        action="store_true",
        help="Disable tensor integration and use legacy models",
    )

    args = parser.parse_args()

    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        exit(1)

    # Override tensor settings if disabled
    if args.disable_tensors:
        logger.info("⚠️ Tensor integration disabled by command line flag")
        # This could modify global settings if needed

    try:
        train_model(args.config, args.optimize, args.verbose)
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        if args.verbose:
            import traceback

            logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()
