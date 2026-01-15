#!/usr/bin/env python3
"""Test script to verify the fixed dataset works correctly."""

from torch_geometric.loader import DataLoader

from astro_lab.data.dataset.astrolab import create_dataset


def test_dataset():
    print("Testing AstroLab dataset...")

    # Create dataset
    dataset = create_dataset(survey_name="gaia", task="node_classification")
    print("Dataset created successfully")
    print(f"Dataset info: {dataset.get_info()}")

    # Test with PyG DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"DataLoader created with {len(loader)} batches")

    # Get first batch
    batch = next(iter(loader))
    print(f"Batch type: {type(batch)}")
    print(f"Batch has x: {hasattr(batch, 'x')}")
    print(f"Batch x shape: {batch.x.shape if hasattr(batch, 'x') else 'None'}")
    print(f"Batch y shape: {batch.y.shape if hasattr(batch, 'y') else 'None'}")
    print(f"Batch pos shape: {batch.pos.shape if hasattr(batch, 'pos') else 'None'}")

    if hasattr(batch, "x") and batch.x is not None:
        print("✅ SUCCESS: Batch has x features!")
        return True
    else:
        print("❌ FAILURE: Batch missing x features!")
        return False


if __name__ == "__main__":
    success = test_dataset()
    exit(0 if success else 1)
