"""
Visualization Mixin for AstroLab Models
======================================

Visualization capabilities for astronomical models.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData


class VisualizationMixin:
    """Visualization capabilities for astronomical models."""

    def plot_attention_weights(
        self, batch: Union[Data, HeteroData, Batch], layer_idx: int = -1
    ) -> Dict[str, Any]:
        """Extract and visualize attention weights from GAT layers."""
        # This would be implemented based on the specific model architecture
        return {
            "attention_weights": None,
            "node_importance": None,
            "edge_importance": None,
        }

    def get_node_embeddings_tsne(
        self, batch: Union[Data, HeteroData, Batch], perplexity: float = 30.0
    ) -> Optional[Tensor]:
        """Get 2D t-SNE embeddings for nodes (placeholder for future implementation)."""
        # Get node embeddings
        if hasattr(self, "get_node_embeddings"):
            embeddings = self.get_node_embeddings(batch)
        else:
            # Fallback: use final layer before output
            embeddings = self(batch)

        # For now, return PCA-like reduction
        try:
            u, s, v = torch.pca_lowrank(embeddings, q=2)
            return embeddings @ v
        except Exception:
            # Fallback to simple dimensionality reduction
            return embeddings[:, :2] if embeddings.size(1) > 2 else embeddings

    def create_graph_visualization_data(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, Any]:
        """Create data structure for graph visualization."""
        # Safely extract data with proper type checking
        try:
            if hasattr(batch, "x") and batch.x is not None:
                positions = batch.x[:, :3] if batch.x.size(1) >= 3 else None
                features = batch.x
                num_nodes = batch.x.size(0)
                num_features = batch.x.size(1)
            else:
                positions = None
                features = None
                num_nodes = 0
                num_features = 0

            if hasattr(batch, "edge_index") and batch.edge_index is not None:
                edge_index = batch.edge_index
                num_edges = batch.edge_index.size(1)
            else:
                edge_index = None
                num_edges = 0

            if hasattr(batch, "y") and batch.y is not None:
                labels = batch.y
            else:
                labels = None

            return {
                "nodes": {
                    "positions": positions,
                    "features": features,
                    "labels": labels,
                },
                "edges": {
                    "edge_index": edge_index,
                    "edge_attr": getattr(batch, "edge_attr", None),
                },
                "metadata": {
                    "num_nodes": num_nodes,
                    "num_edges": num_edges,
                    "num_features": num_features,
                },
            }
        except Exception:
            # Fallback for any errors
            return {
                "nodes": {"positions": None, "features": None, "labels": None},
                "edges": {"edge_index": None, "edge_attr": None},
                "metadata": {"num_nodes": 0, "num_edges": 0, "num_features": 0},
            }

    def get_feature_importance(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Optional[Tensor]:
        """Get feature importance scores using gradient-based methods."""
        try:
            # Enable gradients for input
            if hasattr(batch, "x") and batch.x is not None:
                batch.x.requires_grad_(True)

                # Forward pass
                logits = self(batch)

                # Calculate gradients
                if logits.dim() > 1:
                    target = logits.sum()
                else:
                    target = logits.sum()

                grad = torch.autograd.grad(
                    outputs=target,
                    inputs=batch.x,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Feature importance as absolute gradient mean
                feature_importance = grad.abs().mean(dim=0)
                return feature_importance
            else:
                return None
        except Exception:
            return None

    def get_model_predictions_summary(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, Any]:
        """Get summary of model predictions for visualization."""
        try:
            with torch.no_grad():
                logits = self(batch)

                # Convert to probabilities
                if hasattr(self, "num_classes") and self.num_classes > 1:
                    probs = torch.softmax(logits, dim=-1)
                    predictions = logits.argmax(dim=-1)
                else:
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).long()

                return {
                    "logits": logits,
                    "probabilities": probs,
                    "predictions": predictions,
                    "confidence": probs.max(dim=-1)[0] if probs.dim() > 1 else probs,
                }
        except Exception:
            return {
                "logits": None,
                "probabilities": None,
                "predictions": None,
                "confidence": None,
            }

    def create_astronomical_visualization_data(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, Any]:
        """Create astronomical-specific visualization data."""
        base_data = self.create_graph_visualization_data(batch)

        # Add astronomical-specific information
        astronomical_data = {
            **base_data,
            "astronomical": {
                "coordinate_system": "galactocentric",  # Default
                "distance_units": "parsecs",
                "magnitude_units": "AB_mag",
                "redshift_info": None,
                "stellar_properties": None,
            },
        }

        # Try to extract astronomical features
        if base_data["nodes"]["features"] is not None:
            features = base_data["nodes"]["features"]
            if features.size(1) >= 5:
                # Assume first 3 are coordinates, next 2 are magnitudes
                astronomical_data["astronomical"].update(
                    {
                        "coordinates": features[:, :3],
                        "magnitudes": features[:, 3:5]
                        if features.size(1) >= 5
                        else None,
                        "additional_features": features[:, 5:]
                        if features.size(1) > 5
                        else None,
                    }
                )

        return astronomical_data
