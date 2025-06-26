#!/usr/bin/env python3
"""
Example: Train Gaia Classifier with AstroLab
===========================================

This example shows how to train a Gaia stellar classifier using the
optimized training pipeline for RTX 4070 Mobile GPU.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from astro_lab.training import train_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Training configuration optimized for RTX 4070 Mobile
config = {
    # Model configuration
    "model": "gaia_classifier",
    "dataset": "gaia",
    
    # Training parameters
    "max_epochs": 10,  # Quick training for demo
    "batch_size": 64,  # Good batch size for RTX 4070
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "scheduler": "cosine",
    
    # RTX 4070 optimizations
    "precision": "16-mixed",  # Mixed precision for faster training
    "gradient_clip_val": 1.0,
    "num_workers": 4,  # Optimal for laptop
    
    # Experiment tracking
    "experiment_name": "gaia_classifier_demo",
    "checkpoint_dir": Path("checkpoints/gaia_demo"),
    
    # Early stopping
    "early_stopping_patience": 5,
    
    # Optional: limit samples for quick testing
    # "max_samples": 1000,
    
    # Optional: fast dev run for testing
    # "fast_dev_run": True,
}

if __name__ == "__main__":
    print("🚀 Starting Gaia Classifier Training")
    print("📊 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Run training
    success = train_model(config)
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
        sys.exit(1)
