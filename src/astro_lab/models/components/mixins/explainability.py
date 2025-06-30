"""
Explainability Mixin for AstroLab Models
=======================================

Advanced explainability and interpretability features for astronomical models.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import GATConv, TransformerConv


class ExplainabilityMixin:
    """Advanced explainability capabilities for astronomical models."""

    def extract_attention_weights(
        self, batch: Union[Data, HeteroData, Batch], layer_idx: int = -1
    ) -> Dict[str, Any]:
        """Extract attention weights from GAT/Transformer layers."""
        attention_data = {
            "attention_weights": None,
            "node_importance": None,
            "edge_importance": None,
            "layer_attention": {},
        }

        try:
            # Find attention layers in the model
            attention_layers = []
            for name, module in self.named_modules():
                if isinstance(module, (GATConv, TransformerConv)):
                    attention_layers.append((name, module))

            if not attention_layers:
                return attention_data

            # Use specified layer or last attention layer
            if layer_idx < 0:
                layer_name, layer_module = attention_layers[layer_idx]
            else:
                layer_name, layer_module = attention_layers[layer_idx]

            # Extract attention weights
            with torch.no_grad():
                # Forward pass with attention weights
                if hasattr(layer_module, "forward"):
                    # For GATConv
                    if isinstance(layer_module, GATConv):
                        x = batch.x if hasattr(batch, "x") else None
                        edge_index = (
                            batch.edge_index if hasattr(batch, "edge_index") else None
                        )
                        edge_attr = getattr(batch, "edge_attr", None)

                        if x is not None and edge_index is not None:
                            # Get attention weights
                            out, (row, col), att = layer_module(
                                x, edge_index, edge_attr, return_attention_weights=True
                            )

                            attention_data["attention_weights"] = att
                            attention_data["edge_attention"] = {
                                "row": row,
                                "col": col,
                                "weights": att,
                            }

                            # Calculate node importance from attention
                            node_importance = (
                                self._calculate_node_importance_from_attention(
                                    edge_index, att, x.size(0)
                                )
                            )
                            attention_data["node_importance"] = node_importance

                    # For TransformerConv
                    elif isinstance(layer_module, TransformerConv):
                        x = batch.x if hasattr(batch, "x") else None
                        edge_index = (
                            batch.edge_index if hasattr(batch, "edge_index") else None
                        )
                        edge_attr = getattr(batch, "edge_attr", None)

                        if x is not None and edge_index is not None:
                            # Get attention weights
                            out, (row, col), att = layer_module(
                                x, edge_index, edge_attr, return_attention_weights=True
                            )

                            attention_data["attention_weights"] = att
                            attention_data["edge_attention"] = {
                                "row": row,
                                "col": col,
                                "weights": att,
                            }

                            # Calculate node importance
                            node_importance = (
                                self._calculate_node_importance_from_attention(
                                    edge_index, att, x.size(0)
                                )
                            )
                            attention_data["node_importance"] = node_importance

            attention_data["layer_attention"][layer_name] = attention_data.copy()

        except Exception as e:
            print(f"Error extracting attention weights: {e}")

        return attention_data

    def _calculate_node_importance_from_attention(
        self, edge_index: Tensor, attention_weights: Tensor, num_nodes: int
    ) -> Tensor:
        """Calculate node importance from attention weights."""
        # Aggregate incoming attention weights for each node
        node_importance = torch.zeros(num_nodes, device=edge_index.device)

        # Sum attention weights for each target node
        for i in range(edge_index.size(1)):
            target_node = edge_index[1, i]
            weight = (
                attention_weights[i]
                if attention_weights.dim() == 1
                else attention_weights[i].mean()
            )
            node_importance[target_node] += weight

        # Normalize by number of incoming edges
        edge_counts = torch.bincount(edge_index[1], minlength=num_nodes)
        edge_counts = torch.clamp(edge_counts, min=1)  # Avoid division by zero
        node_importance = node_importance / edge_counts

        return node_importance

    def integrated_gradients(
        self,
        batch: Union[Data, HeteroData, Batch],
        target_class: Optional[int] = None,
        steps: int = 50,
        baseline: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute integrated gradients for feature attribution."""
        if not hasattr(batch, "x") or batch.x is None:
            return {"attributions": None, "baseline": None}

        x = batch.x.clone()
        original_x = x.clone()

        # Create baseline (zero baseline if not provided)
        if baseline is None:
            baseline = torch.zeros_like(x)

        # Interpolate between baseline and input
        interpolated = []
        for step in range(steps + 1):
            alpha = step / steps
            interpolated_x = baseline + alpha * (original_x - baseline)
            interpolated.append(interpolated_x)

        interpolated = torch.stack(interpolated)

        # Compute gradients
        attributions = torch.zeros_like(x)

        for i, interp_x in enumerate(interpolated):
            # Create new batch with interpolated features
            interp_batch = batch.clone()
            interp_batch.x = interp_x.requires_grad_(True)

            # Forward pass
            logits = self(interp_batch)

            # Select target class
            if target_class is not None and logits.dim() > 1:
                target = logits[:, target_class]
            else:
                target = logits.sum()

            # Compute gradients
            grad = torch.autograd.grad(
                outputs=target,
                inputs=interp_batch.x,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Accumulate gradients
            attributions += grad

        # Average and multiply by input difference
        attributions = attributions / (steps + 1)
        attributions = attributions * (original_x - baseline)

        return {
            "attributions": attributions,
            "baseline": baseline,
            "feature_importance": attributions.abs().mean(dim=0),
        }

    def saliency_map(
        self, batch: Union[Data, HeteroData, Batch], target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute saliency maps using gradient-based attribution."""
        if not hasattr(batch, "x") or batch.x is None:
            return {"saliency": None, "feature_importance": None}

        x = batch.x.clone().requires_grad_(True)

        # Create new batch with gradients enabled
        grad_batch = batch.clone()
        grad_batch.x = x

        # Forward pass
        logits = self(grad_batch)

        # Select target
        if target_class is not None and logits.dim() > 1:
            target = logits[:, target_class].sum()
        else:
            target = logits.sum()

        # Compute gradients
        grad = torch.autograd.grad(
            outputs=target, inputs=x, create_graph=False, retain_graph=False
        )[0]

        # Saliency map is the absolute gradient
        saliency = grad.abs()

        return {
            "saliency": saliency,
            "feature_importance": saliency.mean(dim=0),
            "gradients": grad,
        }

    def astronomical_feature_importance(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, Any]:
        """Compute feature importance specific to astronomical data."""
        if not hasattr(batch, "x") or batch.x is None:
            return {}

        # Get general feature importance
        general_importance = self.get_feature_importance(batch)

        if general_importance is None:
            return {}

        # Assume first 3 features are coordinates (RA, Dec, Distance)
        # and next features are magnitudes/photometry
        num_features = general_importance.size(0)

        astronomical_importance = {
            "coordinates": {
                "ra": general_importance[0] if num_features > 0 else 0.0,
                "dec": general_importance[1] if num_features > 1 else 0.0,
                "distance": general_importance[2] if num_features > 2 else 0.0,
            },
            "photometry": {},
            "other_features": {},
        }

        # Photometric features (assume features 3-7 are magnitudes)
        for i in range(3, min(8, num_features)):
            mag_name = f"mag_{i - 2}"
            astronomical_importance["photometry"][mag_name] = general_importance[i]

        # Other features
        for i in range(8, num_features):
            feature_name = f"feature_{i}"
            astronomical_importance["other_features"][feature_name] = (
                general_importance[i]
            )

        return {
            "general_importance": general_importance,
            "astronomical_breakdown": astronomical_importance,
            "most_important_coordinate": max(
                astronomical_importance["coordinates"].items(), key=lambda x: x[1]
            )[0],
            "most_important_magnitude": max(
                astronomical_importance["photometry"].items(), key=lambda x: x[1]
            )[0]
            if astronomical_importance["photometry"]
            else None,
        }

    def explain_prediction(
        self, batch: Union[Data, HeteroData, Batch], target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Comprehensive prediction explanation."""
        explanation = {
            "prediction": None,
            "confidence": None,
            "feature_importance": None,
            "attention_weights": None,
            "saliency": None,
            "integrated_gradients": None,
            "astronomical_importance": None,
        }

        try:
            # Get prediction
            with torch.no_grad():
                logits = self(batch)
                if hasattr(self, "num_classes") and self.num_classes > 1:
                    probs = torch.softmax(logits, dim=-1)
                    prediction = logits.argmax(dim=-1)
                    confidence = probs.max(dim=-1)[0]
                else:
                    probs = torch.sigmoid(logits)
                    prediction = (probs > 0.5).long()
                    confidence = probs

            explanation["prediction"] = prediction
            explanation["confidence"] = confidence
            explanation["logits"] = logits
            explanation["probabilities"] = probs

            # Get various explanations
            explanation["feature_importance"] = self.get_feature_importance(batch)
            explanation["attention_weights"] = self.extract_attention_weights(batch)
            explanation["saliency"] = self.saliency_map(batch, target_class)
            explanation["integrated_gradients"] = self.integrated_gradients(
                batch, target_class
            )
            explanation["astronomical_importance"] = (
                self.astronomical_feature_importance(batch)
            )

        except Exception as e:
            print(f"Error in explain_prediction: {e}")

        return explanation

    def create_explanation_visualization_data(
        self, batch: Union[Data, HeteroData, Batch]
    ) -> Dict[str, Any]:
        """Create data structure for explanation visualization."""
        explanation = self.explain_prediction(batch)

        # Extract visualization-friendly data
        viz_data = {
            "predictions": {
                "prediction": explanation["prediction"].cpu().numpy()
                if explanation["prediction"] is not None
                else None,
                "confidence": explanation["confidence"].cpu().numpy()
                if explanation["confidence"] is not None
                else None,
                "probabilities": explanation["probabilities"].cpu().numpy()
                if explanation["probabilities"] is not None
                else None,
            },
            "feature_importance": {
                "general": explanation["feature_importance"].cpu().numpy()
                if explanation["feature_importance"] is not None
                else None,
                "astronomical": explanation["astronomical_importance"],
            },
            "attention": {
                "node_importance": explanation["attention_weights"]["node_importance"]
                .cpu()
                .numpy()
                if explanation["attention_weights"]["node_importance"] is not None
                else None,
                "edge_attention": explanation["attention_weights"]["edge_attention"],
            },
            "attributions": {
                "saliency": explanation["saliency"]["saliency"].cpu().numpy()
                if explanation["saliency"]["saliency"] is not None
                else None,
                "integrated_gradients": explanation["integrated_gradients"][
                    "attributions"
                ]
                .cpu()
                .numpy()
                if explanation["integrated_gradients"]["attributions"] is not None
                else None,
            },
        }

        return viz_data
