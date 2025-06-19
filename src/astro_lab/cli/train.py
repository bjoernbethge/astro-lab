"""
AstroLab Training CLI - Streamlined Interface

Schlanke moderne Training-Schnittstelle mit YAML-Konfiguration.
Fokus auf Einfachheit und Performance mit Lightning DataModules.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from astro_lab.training import AstroTrainer, create_astro_datamodule
from astro_lab.models.utils import create_gaia_classifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Empty or invalid configuration file: {config_path}")
        return config


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_model_from_config(model_config: Dict[str, Any]) -> Any:
    """Create model from configuration."""
    model_type = model_config.get("type", "gaia_classifier")
    model_params = model_config.get("params", {})
    
    # Simple model factory - can be extended
    if model_type == "gaia_classifier":
        return create_gaia_classifier(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(config_path: str, optimize: bool = False, verbose: bool = False) -> None:
    """Train model with given configuration."""
    setup_logging(verbose)
    logger.info(f"🚀 Loading configuration from {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    model_config = config["model"]
    data_config = config["data"]
    training_config = config["training"]
    
    # Create model
    logger.info("🏗️ Creating model...")
    model = create_model_from_config(model_config)
    
    # Create datamodule
    logger.info("📊 Setting up data pipeline...")
    dataset_name = data_config["dataset"]
    dataset_params = {k: v for k, v in data_config.items() if k != "dataset"}
    datamodule = create_astro_datamodule(dataset_name, **dataset_params)
    
    # Create trainer
    logger.info("⚡ Initializing trainer...")
    trainer = AstroTrainer(model=model, **training_config)
    
    if optimize:
        # Hyperparameter optimization
        logger.info("🎯 Starting hyperparameter optimization...")
        optimization_config = config.get("optimization", {})
        best_params = trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            **optimization_config
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
        description="AstroLab Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required positional argument - config file
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file"
    )
    
    # Optional flags
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization instead of training"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        exit(1)
    
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
