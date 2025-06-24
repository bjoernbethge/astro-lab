"""Simplified temporal GNN models."""

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from ..components import (
    GraphProcessor,
    PoolingModule,
    create_conv_layer,
    create_mlp,
    create_output_head,
)


class TemporalGCN(nn.Module):
    """Simple Temporal Graph Convolutional Network."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_graph_layers: int = 3,
        num_rnn_layers: int = 2,
        rnn_type: str = "lstm",
        conv_type: str = "gcn",
        task: str = "regression",
        dropout: float = 0.1,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.hidden_dim = hidden_dim
        self.task = task
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph processor for spatial features
        self.graph_processor = GraphProcessor(
            hidden_dim=hidden_dim,
            num_layers=num_graph_layers,
            conv_type=conv_type,
            dropout=dropout,
            **kwargs
        )
        
        # RNN for temporal processing
        if rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                dropout=dropout if num_rnn_layers > 1 else 0,
                batch_first=True,
            )
        elif rnn_type.lower() == "gru":
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                dropout=dropout if num_rnn_layers > 1 else 0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
            
        # Pooling
        self.pooling = PoolingModule("mean")
        
        # Output head
        head_type = task.replace('graph_', '').replace('node_', '')
        self.output_head = create_output_head(
            head_type,
            input_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            **kwargs
        )
        
        self.to(self.device)
        
    def encode_snapshot(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a single graph snapshot."""
        # Project input
        h = self.input_projection(x)
        
        # Process through graph layers
        h = self.graph_processor(h, edge_index)
        
        # Pool to graph level
        h = self.pooling(h, batch)
        
        return h
        
    def forward(
        self,
        snapshot_sequence: Union[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Process temporal sequence of graphs."""
        
        # Handle single snapshot
        if isinstance(snapshot_sequence, dict):
            snapshot_sequence = [snapshot_sequence]
            
        # Encode each snapshot
        graph_embeddings = []
        for snapshot in snapshot_sequence:
            x = snapshot["x"].to(self.device)
            edge_index = snapshot["edge_index"].to(self.device)
            batch = snapshot.get("batch")
            if batch is not None:
                batch = batch.to(self.device)
                
            h = self.encode_snapshot(x, edge_index, batch)
            graph_embeddings.append(h)
            
        # Stack for temporal processing
        if len(graph_embeddings) > 1:
            # Multiple snapshots - use RNN
            graph_sequence = torch.stack(graph_embeddings, dim=1)
            rnn_out, _ = self.rnn(graph_sequence)
            final_embedding = rnn_out[:, -1, :]  # Last output
        else:
            # Single snapshot
            final_embedding = graph_embeddings[0]
            
        # Output prediction
        output = self.output_head(final_embedding)
        
        if return_embeddings:
            return {"predictions": output, "embeddings": final_embedding}
        return output 