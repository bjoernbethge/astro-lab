"""
Train a GNN model on real Gaia DR3 data.

This example demonstrates training with the actual Gaia data available in the project.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from astro_lab.data import create_datamodule
from astro_lab.models import create_model
from astro_lab.config import get_data_paths


def main():
    print("=== Training GNN on Real Gaia DR3 Data ===")
    print("Using 27.8M stars from Gaia DR3 bright star catalog")
    
    # Configuration
    config = {
        "max_epochs": 30,
        "learning_rate": 1e-3,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.2,
        "conv_type": "gcn",
    }
    
    # Create data module
    print("\nCreating data module...")
    dm = create_datamodule(
        survey="gaia",
        task="node_classification",
        max_samples=100000,  # Use 100k stars for manageable training
        num_workers=0,  # Windows compatibility
        k_neighbors=20,
        graph_method="knn",
        astronomical_features=True,
        cosmic_web_features=False,
        multi_scale=False,
    )
    
    # Prepare data
    print("\nLoading and processing Gaia data...")
    dm.prepare_data()
    dm.setup()
    
    # Get dataset info
    info = dm.get_info()
    print(f"\n✅ Data loaded successfully!")
    print(f"  Number of stars: {info['num_nodes']:,}")
    print(f"  Number of edges: {info['num_edges']:,}")
    print(f"  Number of features: {info['num_features']}")
    print(f"  Number of classes: {info['num_classes']}")
    
    if 'graph_stats' in info:
        print(f"\nGraph statistics:")
        print(f"  Average degree: {info['graph_stats']['avg_degree']:.2f}")
        print(f"  Max degree: {info['graph_stats']['max_degree']}")
    
    # Create model
    print(f"\nCreating {config['conv_type'].upper()} model...")
    model = create_model(
        model_type="astro_model",
        in_channels=info['num_features'],
        hidden_channels=config['hidden_dim'],
        out_channels=info['num_classes'],
        num_layers=config['num_layers'],
        conv_type=config['conv_type'],
        dropout=config['dropout'],
        task="node_classification",
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/gaia',
        filename='gaia-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
    )
    
    # Logger
    data_paths = get_data_paths()
    mlf_logger = MLFlowLogger(
        experiment_name="gaia_real_data",
        tracking_uri=f"file:///{data_paths['mlruns_dir']}",
        tags={
            "survey": "gaia",
            "task": "node_classification",
            "conv_type": config['conv_type'],
            "data_size": info['num_nodes'],
        }
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlf_logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )
    
    # Log hyperparameters
    mlf_logger.log_hyperparams(config)
    
    # Train
    print(f"\nStarting training...")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Classes are based on G magnitude bins for demonstration")
    print("-" * 60)
    
    trainer.fit(model, dm)
    
    # Test
    print("\nTesting model...")
    trainer.test(model, dm)
    
    print(f"\n✅ Training completed!")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"\nTo view results: mlflow ui --backend-store-uri {data_paths['mlruns_dir']}")


if __name__ == "__main__":
    main()
