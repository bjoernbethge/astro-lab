#!/usr/bin/env python3
"""
AstroLab Hyperparameter Optimization CLI
========================================

Modern CLI for hyperparameter optimization of astronomical ML models.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from astro_lab.training import (
    AstroLabOptimizer,
    TrainingConfig,
    get_data_config,
)
from astro_lab.utils.config import load_config

# Placeholder for data loading - replace with actual implementation
def load_data(config: TrainingConfig, max_samples: Optional[int] = None):
    """Load data based on configuration."""
    # This is a placeholder - implement based on your data module
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    import torch
    
    # Create dummy data for demonstration
    def create_dummy_data(num_samples=100):
        data_list = []
        for _ in range(num_samples):
            x = torch.randn(100, 10)  # 100 nodes, 10 features
            edge_index = torch.randint(0, 100, (2, 200))
            
            # Set target based on task
            if "classifier" in config.model_name:
                y = torch.randint(0, config.model_config.get("num_classes", 7), (1,))
            else:
                y = torch.randn(config.model_config.get("output_dim", 5))
            
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return data_list
    
    # Create dataloaders
    train_data = create_dummy_data(max_samples or 1000)
    val_data = create_dummy_data(max_samples // 10 if max_samples else 100)
    test_data = create_dummy_data(max_samples // 20 if max_samples else 50)
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def setup_logging() -> logging.Logger:
    """Setup modern logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for optimization CLI."""
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for astronomical ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with config file
  astro-lab optimize my_experiment.yaml --trials 50

  # Quick optimization
  astro-lab optimize config.yaml --trials 10 --timeout 3600
  
  # Optimize specific model
  astro-lab optimize config.yaml --algorithm optuna --metric val_accuracy
        """,
    )
    
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Optimization timeout in seconds",
    )
    parser.add_argument(
        "--algorithm",
        choices=["optuna", "ray", "grid", "random"],
        default="optuna",
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples for debugging",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run a quick development test",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Path to the dataset for training",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to optimize (e.g., val_loss, val_accuracy)",
    )
    parser.add_argument(
        "--direction",
        choices=["minimize", "maximize"],
        help="Optimization direction",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=50,
        help="Number of epochs per trial",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=200,
        help="Number of epochs for final training with best params",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="Name for the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        help="Optuna storage backend URL",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        default=True,
        help="Enable trial pruning",
    )
    parser.add_argument(
        "--no-pruning",
        dest="pruning",
        action="store_false",
        help="Disable trial pruning",
    )
    
    return parser


def main(args=None) -> int:
    """Main CLI entry point."""
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    
    logger = setup_logging()
    
    try:
        # 1. Load configuration
        config_dict = load_config(str(args.config))
        config = TrainingConfig(**config_dict)
        
        # Apply overrides
        if args.max_samples:
            config.max_samples = args.max_samples
        
        logger.info(f"🔍 Starting hyperparameter optimization for {config.name}")
        logger.info(f"📊 Model: {config.model_name}")
        logger.info(f"🔧 Algorithm: {args.algorithm}")
        logger.info(f"🔄 Trials: {args.trials}")
        
        if args.timeout:
            logger.info(f"⏱️  Timeout: {args.timeout} seconds")
        
        # 2. Load data
        logger.info("📂 Loading data...")
        train_loader, val_loader, test_loader = load_data(config, args.max_samples)
        
        # 3. Get model configuration
        if not config.model_config:
            # Get default model config
            data_config = get_data_config(config.model_name)
            config.model_config = {
                "input_dim": 10,  # Should come from data
                "num_classes": 7 if "classifier" in config.model_name else None,
                "output_dim": 5 if "classifier" not in config.model_name else None,
            }
        
        # 4. Determine optimization metric and direction
        metric = args.metric
        direction = args.direction
        
        if not metric:
            # Default based on task
            if "classifier" in config.model_name:
                metric = "val_accuracy"
                direction = direction or "maximize"
            else:
                metric = "val_loss"
                direction = direction or "minimize"
        
        if not direction:
            # Infer from metric
            if "loss" in metric or "error" in metric:
                direction = "minimize"
            else:
                direction = "maximize"
        
        logger.info(f"📈 Optimizing {metric} ({direction})")
        
        # 5. Create optimizer
        optimizer = AstroLabOptimizer(
            model_name=config.model_name,
            base_model_config=config.model_config,
            experiment_name=config.experiment_name + "_optimization",
            study_name=args.study_name or f"{config.model_name}_study",
            n_trials=args.trials,
            timeout=args.timeout,
            direction=direction,
            metric=metric,
            pruning=args.pruning,
            storage=args.storage,
        )
        
        # 6. Run optimization
        best_params = optimizer.optimize(
            train_loader,
            val_loader,
            epochs_per_trial=args.epochs_per_trial
        )
        
        logger.info(f"✅ Optimization completed!")
        logger.info(f"🏆 Best {metric}: {optimizer.best_value:.4f}")
        logger.info(f"🎯 Best parameters: {best_params}")
        
        # 7. Train final model with best parameters
        if args.final_epochs > 0:
            logger.info(f"🚀 Training final model with best parameters for {args.final_epochs} epochs")
            
            trainer = optimizer.train_best_model(
                train_loader,
                val_loader,
                test_loader,
                max_epochs=args.final_epochs
            )
            
            logger.info("✅ Final model training completed!")
            
            # Test the model
            if test_loader:
                test_results = trainer.test(test_loader)
                logger.info(f"📊 Test results: {test_results}")
        
        # 8. Log results location
        results_dir = Path(f"optimization_results/{optimizer.study_name}")
        if results_dir.exists():
            logger.info(f"📁 Optimization results saved to: {results_dir}")
            logger.info(f"   - best_params.json: Best hyperparameters")
            logger.info(f"   - optimization_history.html: Optimization progress")
            logger.info(f"   - param_importance.html: Parameter importance")
        
        logger.info(f"📁 MLflow UI: mlflow ui --backend-store-uri ./mlruns")
        
        return 0
        
    except KeyboardInterrupt:
        logger.error("❌ Optimization interrupted by user")
        return 1
    except ImportError as e:
        if "optuna" in str(e):
            logger.error("❌ Optuna is required for hyperparameter optimization")
            logger.error("   Install with: pip install optuna")
        else:
            logger.error(f"❌ Import error: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
