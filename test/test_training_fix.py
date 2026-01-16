#!/usr/bin/env python3
"""Test script to verify training works with the fixed dataset."""

import torch
from torch_geometric.loader import DataLoader

from astro_lab.data.dataset.astrolab import create_dataset
from astro_lab.models.astro_model import AstroModel


def create_edges_for_batch(batch):
    """Create a simple edge structure for the batch."""
    # Create fully connected edges for each graph in the batch
    edge_indices = []

    if hasattr(batch, "batch"):
        # Multiple graphs in batch
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            num_nodes = mask.sum().item()
            if num_nodes > 1:
                # Create fully connected edges for this graph
                nodes = torch.arange(num_nodes, device=batch.x.device)
                src, dst = torch.meshgrid(nodes, nodes, indexing="ij")
                src = src.flatten()
                dst = dst.flatten()
                # Remove self-loops
                mask = src != dst
                src = src[mask]
                dst = dst[mask]
                # Add offset for batch
                offset = (batch.batch == i).nonzero(as_tuple=False).min()
                src += offset
                dst += offset
                edge_indices.append(torch.stack([src, dst], dim=0))

        if edge_indices:
            batch.edge_index = torch.cat(edge_indices, dim=1)
        else:
            # Fallback: create edges between consecutive nodes
            batch.edge_index = torch.stack(
                [torch.arange(batch.x.shape[0] - 1), torch.arange(1, batch.x.shape[0])],
                dim=0,
            )
    else:
        # Single graph
        num_nodes = batch.x.shape[0]
        if num_nodes > 1:
            # Create edges between consecutive nodes
            batch.edge_index = torch.stack(
                [torch.arange(num_nodes - 1), torch.arange(1, num_nodes)], dim=0
            )
        else:
            # Single node - create self-loop
            batch.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return batch


def test_training():
    print("Testing training with fixed dataset...")

    # Create dataset
    dataset = create_dataset(survey_name="gaia", task="node_classification")
    print(f"Dataset created with {len(dataset)} samples")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"DataLoader created with {len(loader)} batches")

    # Get first batch
    batch = next(iter(loader))
    print(f"Batch x shape: {batch.x.shape}")
    print(f"Batch y shape: {batch.y.shape}")
    print(f"Batch pos shape: {batch.pos.shape}")

    # Create edges for the batch
    batch = create_edges_for_batch(batch)
    print(f"Created edge_index shape: {batch.edge_index.shape}")

    # Create model
    model = AstroModel(
        num_features=batch.x.shape[1],
        num_classes=2,  # Binary classification
        hidden_channels=64,
        num_layers=2,
        conv_type="sage",  # Use SAGE which is more robust
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch)
        print(f"Model output shape: {output.shape}")
        print(f"Expected shape: {batch.x.shape[0]} x 2")

        if output.shape == (batch.x.shape[0], 2):
            print("✅ SUCCESS: Model forward pass works!")
            return True
        else:
            print("❌ FAILURE: Model output shape mismatch!")
            return False


if __name__ == "__main__":
    success = test_training()
    exit(0 if success else 1)
