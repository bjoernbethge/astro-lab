#!/usr/bin/env python
"""Simple test script to verify the fixed training pipeline."""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from astro_lab.data import AstroLabDataModule, AstroLabInMemoryDataset, KNNSampler
from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer


def test_training():
    """Test the complete training pipeline with minimal data."""
    print("=" * 60)
    print("Testing AstroLab Training Pipeline")
    print("=" * 60)
    
    # 1. Create dataset with small sample
    print("\n1. Creating dataset...")
    try:
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
            max_samples=1000,  # Use only 1000 samples for quick test
        )
        print(f"   ✓ Dataset created successfully")
    