"""Test the complete training workflow for Gaia data."""

import torch

from astro_lab.data import AstroLabDataModule, AstroLabInMemoryDataset
from astro_lab.data.samplers import KNNSampler
from astro_lab.models import AstroModel


def test_workflow():
    print("=== Testing AstroLab Training Workflow ===")

    # 1. Create dataset
    print("\n1. Creating dataset...")
    dataset = AstroLabInMemoryDataset(
        survey_name="gaia",
        task="node_classification",
        sampling_strategy="knn",
        sampler_kwargs={"k": 8},
    )
    print(f"   Dataset length: {len(dataset)}")
    info = dataset.get_info()
    print(
        f"   Dataset info: num_features={info['num_features']}, num_classes={info['num_classes']}"
    )

    # 2. Create data module
    print("\n2. Creating data module...")
    sampler = KNNSampler(k=8)
    datamodule = AstroLabDataModule(
        dataset=dataset,
        sampler=sampler,
        batch_size=16,
        num_workers=0,  # 0 for testing
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    datamodule.setup()

    # 3. Check data loaders
    print("\n3. Checking data loaders...")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 4. Check sample batch
    print("\n4. Checking sample batch...")
    sample_batch = next(iter(train_loader))
    print(f"   Batch type: {type(sample_batch)}")
    print(
        f"   Batch x shape: {sample_batch.x.shape if hasattr(sample_batch, 'x') else 'No x'}"
    )
    print(
        f"   Batch y shape: {sample_batch.y.shape if hasattr(sample_batch, 'y') else 'No y'}"
    )
    print(
        f"   Batch edge_index shape: {sample_batch.edge_index.shape if hasattr(sample_batch, 'edge_index') else 'No edge_index'}"
    )

    if hasattr(sample_batch, "y"):
        print(f"   Unique labels in batch: {torch.unique(sample_batch.y).tolist()}")

    # 5. Create model
    print("\n5. Creating model...")
    model = AstroModel(
        num_features=info["num_features"],
        num_classes=info["num_classes"],
        hidden_dim=128,
        num_layers=3,
        conv_type="gcn",
        task="node_classification",
        dropout=0.1,
        learning_rate=1e-3,
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 6. Test forward pass
    print("\n6. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
        print(f"   Output shape: {output.shape}")
        print(
            f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]"
        )

    # 7. Test loss computation
    print("\n7. Testing loss computation...")
    if hasattr(sample_batch, "y"):
        loss = model._compute_loss(output, sample_batch.y)
        print(f"   Loss: {loss.item():.4f}")

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_workflow()
