"""Simplified survey GNN using composition."""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components import (
    DeviceMixin,
    FeatureProcessor,
    GraphProcessor,
    PoolingModule,
    create_output_head,
)


class AstroSurveyGNN(nn.Module):
    """Simplified survey GNN using composition."""

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        conv_type: str = "gcn",
        task: str = "classification",
        use_photometry: bool = True,
        use_astrometry: bool = True,
        use_spectroscopy: bool = False,
        pooling: str = "mean",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        super().__init__()

        # Device management
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task = task
        self.pooling = pooling

        # Calculate input dimension if not provided
        if input_dim is None:
            calculated_dim = 0
            if use_photometry:
                calculated_dim += kwargs.get("photometry_dim", 5)
            if use_astrometry:
                calculated_dim += kwargs.get("astrometry_dim", 5)
            if use_spectroscopy:
                calculated_dim += kwargs.get("spectroscopy_dim", 3)

            # Default fallback
            if calculated_dim == 0:
                calculated_dim = 13  # Default total features
            input_dim = calculated_dim

        # Use composition instead of complex inheritance
        self.feature_processor = FeatureProcessor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            use_photometry=use_photometry,
            use_astrometry=use_astrometry,
            use_spectroscopy=use_spectroscopy,
            **kwargs,
        )

        self.graph_processor = GraphProcessor(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout,
            **kwargs,
        )

        # Pooling for graph-level tasks
        if "graph" in task:
            self.pooling_module = PoolingModule(pooling)
        else:
            self.pooling_module = None

        # Output head based on task
        head_type = task.replace("graph_", "").replace("node_", "")
        self.output_head = create_output_head(
            head_type,
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            **kwargs,
        )

        # Move to device
        self.to(self.device)

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fully robust forward pass that handles all calling conventions.

        This handles:
        - Lightning calling model(batch)
        - torch.compile calling model(batch)
        - PyTorch Geometric calling model(batch.x, batch.edge_index)
        - Manual calling model(data=batch) or model(batch=batch)
        - Empty calls for model introspection
        """
        try:
            # Extract batch from various calling conventions
            batch = None
            edge_index = None
            batch_idx = None

            # Case 1: Positional argument (most common)
            if args:
                batch = args[0]
                # Extract additional args if provided
                if len(args) > 1:
                    edge_index = args[1]
                if len(args) > 2:
                    batch_idx = args[2]

            # Case 2: Keyword arguments
            if batch is None:
                batch = kwargs.get("batch", kwargs.get("data", kwargs.get("x")))

            if edge_index is None:
                edge_index = kwargs.get("edge_index")

            if batch_idx is None:
                batch_idx = kwargs.get("batch_index", kwargs.get("batch"))

            # Case 3: Empty call for model introspection - create dummy data
            if batch is None and not args and not kwargs:
                # Create minimal dummy data for introspection
                dummy_batch_size = 1
                dummy_features = self.feature_processor.input_dim
                x = torch.zeros((dummy_batch_size, dummy_features), device=self.device)
                edge_index = torch.tensor(
                    [[0], [0]], device=self.device, dtype=torch.long
                )
                batch_idx = torch.zeros(
                    dummy_batch_size, device=self.device, dtype=torch.long
                )

                # Process with dummy data
                features = self.feature_processor(x)
                embeddings = self.graph_processor(features, edge_index)
                if self.pooling_module is not None:
                    pooled = self.pooling_module(embeddings, batch_idx)
                else:
                    pooled = embeddings
                output = self.output_head(pooled)
                return output

            # Case 4: If still no batch with arguments, something is wrong
            if batch is None:
                raise ValueError(
                    f"No batch data provided. args: {[type(arg).__name__ for arg in args]}, kwargs: {list(kwargs.keys())}"
                )

            # Handle different input formats
            if hasattr(batch, "x") and hasattr(batch, "edge_index"):
                # PyTorch Geometric Data object
                x = batch.x
                if edge_index is None:
                    edge_index = batch.edge_index
                if batch_idx is None and hasattr(batch, "batch"):
                    batch_idx = batch.batch
            elif isinstance(batch, torch.Tensor):
                # Raw tensor
                x = batch
            elif isinstance(batch, dict):
                # Dictionary format
                x = batch.get("x", batch)
                if edge_index is None:
                    edge_index = batch.get("edge_index")
                if batch_idx is None:
                    batch_idx = batch.get("batch")
            else:
                # Assume it's a tensor
                x = batch

            # Ensure tensors are on correct device
            if isinstance(x, torch.Tensor):
                x = x.to(self.device)
            if edge_index is not None and isinstance(edge_index, torch.Tensor):
                edge_index = edge_index.to(self.device)
            if batch_idx is not None and isinstance(batch_idx, torch.Tensor):
                batch_idx = batch_idx.to(self.device)

            # Process features
            features = self.feature_processor(x)

            # Process through graph layers if edge_index available
            if edge_index is not None:
                embeddings = self.graph_processor(features, edge_index)
            else:
                embeddings = features

            # Apply pooling if needed
            if self.pooling_module is not None and batch_idx is not None:
                pooled = self.pooling_module(embeddings, batch_idx)
            else:
                pooled = embeddings

            # Generate output
            output = self.output_head(pooled)

            # Return embeddings if requested
            if kwargs.get("return_embeddings", False):
                return {"logits": output, "embeddings": embeddings}

            return output

        except Exception as e:
            # error reporting
            import logging
            import traceback

            logger = logging.getLogger(__name__)
            logger.error(f"❌ Forward pass failed: {e}")
            logger.error(f"   Args: {[type(arg).__name__ for arg in args]}")
            logger.error(f"   Kwargs: {list(kwargs.keys())}")
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise

    def get_num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
