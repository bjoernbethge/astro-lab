"""
Hyperparameter Optimization Example
==================================

This example demonstrates how to use the integrated hyperparameter optimization
functionality in AstroTrainer.
"""

import logging
from pathlib import Path

from astro_lab.data import create_astro_datamodule
from astro_lab.models import ModelFactory
from astro_lab.training import AstroLightningModule, AstroTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_gaia_model():
    """Example of optimizing hyperparameters for a Gaia stellar classification model."""
    
    # Create data module
    logger.info("📊 Creating Gaia data module...")
    datamodule = create_astro_datamodule(
        "gaia",
        batch_size=64,
        num_workers=4,
        return_tensor=True,
    )
    
    # Create base model
    logger.info("🔧 Creating base model...")
    model = ModelFactory.create_survey_model(
        survey="gaia",
        task="stellar_classification",
        hidden_dim=128,  # Initial guess
        num_layers=3,    # Initial guess
        dropout=0.2,     # Initial guess
    )
    
    # Create Lightning module
    lightning_module = AstroLightningModule(
        model=model,
        task_type="classification",
        learning_rate=1e-3,  # Initial guess
    )
    
    # Create trainer
    trainer = AstroTrainer(
        lightning_module=lightning_module,
        experiment_name="gaia_optimization_example",
        max_epochs=50,  # Will be reduced during optimization
        accelerator="auto",
        devices="auto",
    )
    
    # Define custom search space (optional)
    search_space = {
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        "hidden_dim": {"type": "int", "low": 64, "high": 512},
        "num_layers": {"type": "int", "low": 2, "high": 6},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-3, "log": True},
    }
    
    # Run optimization
    logger.info("🔍 Starting hyperparameter optimization...")
    results = trainer.optimize_hyperparameters(
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        n_trials=20,  # Number of trials
        timeout=3600,  # 1 hour timeout
        search_space=search_space,
        monitor="val_loss",  # Metric to optimize
    )
    
    # Print results
    logger.info("✅ Optimization complete!")
    logger.info(f"Best parameters: {results['best_params']}")
    logger.info(f"Best validation loss: {results['best_value']:.4f}")
    logger.info(f"Number of trials: {results['n_trials']}")
    
    # Train with best parameters
    logger.info("🚀 Training with best parameters...")
    
    # Update model with best parameters
    best_model = ModelFactory.create_survey_model(
        survey="gaia",
        task="stellar_classification",
        hidden_dim=results['best_params']['hidden_dim'],
        num_layers=results['best_params']['num_layers'],
        dropout=results['best_params']['dropout'],
    )
    
    # Create new Lightning module with best parameters
    best_module = AstroLightningModule(
        model=best_model,
        task_type="classification",
        learning_rate=results['best_params']['learning_rate'],
        weight_decay=results['best_params'].get('weight_decay', 1e-4),
    )
    
    # Create final trainer
    final_trainer = AstroTrainer(
        lightning_module=best_module,
        experiment_name="gaia_best_model",
        max_epochs=100,
        accelerator="auto",
        devices="auto",
    )
    
    # Train final model
    final_trainer.fit(datamodule=datamodule)
    
    # Test final model
    test_results = final_trainer.test(datamodule=datamodule)
    logger.info(f"🎯 Test results: {test_results}")
    
    # Save best model
    saved_models = final_trainer.save_best_models_to_results(top_k=1)
    logger.info(f"💾 Saved best model: {saved_models}")


if __name__ == "__main__":
    optimize_gaia_model() 