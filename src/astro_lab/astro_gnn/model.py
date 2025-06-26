"""
AstroGNN Model - PointNet++ für astronomische Punktwolken

Dieses Modul implementiert ein Graph Neural Network basierend auf PointNet++
für die Klassifikation astronomischer Punktwolken (z.B. Sternhaufen).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, knn_graph
from torch_geometric.data import Data
from tensordict import TensorDict
from typing import Dict, Optional, Tuple


class PointNetLayer(nn.Module):
    """
    Custom PointNet Layer für astronomische Features
    
    Diese Layer verarbeitet 3D-Koordinaten und zusätzliche Features
    (z.B. Helligkeit, Farbe, Geschwindigkeit)
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_batch_norm: bool = True):
        super().__init__()
        
        # Local MLP: Verarbeitet Features + relative Positionen
        self.local_nn = nn.Sequential(
            nn.Linear(in_channels + 3, 64),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )
        
        # Global MLP: Verarbeitet aggregierte Features
        self.global_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU()
        )
        
        self.pointnet = PointNetConv(self.local_nn, self.global_nn)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass durch PointNet Layer"""
        return self.pointnet(x, pos, edge_index)


class AstroPointCloudGNN(nn.Module):
    """
    Haupt-Modell für astronomische Punktwolken-Klassifikation
    
    Architektur:
    - 3 PointNet++ Layer mit zunehmender Feature-Dimension
    - Global Max Pooling
    - Klassifikations-Kopf mit Dropout
    
    Args:
        num_features: Anzahl der Input-Features (Standard: 3 für x,y,z)
        num_classes: Anzahl der Klassen (z.B. 7 für Sterntypen)
        hidden_dim: Dimension der versteckten Layer
        dropout: Dropout-Rate für Regularisierung
        k_neighbors: Anzahl der Nachbarn für k-NN Graph
    """
    
    def __init__(
        self,
        num_features: int = 3,
        num_classes: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        k_neighbors: int = 16
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        
        # PointNet++ Layers
        self.conv1 = PointNetLayer(num_features, hidden_dim)
        self.conv2 = PointNetLayer(hidden_dim, hidden_dim * 2)
        self.conv3 = PointNetLayer(hidden_dim * 2, hidden_dim * 4)
        
        # Klassifikations-Kopf
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def create_graph_from_pointcloud(
        self, 
        pos: torch.Tensor, 
        features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Erstellt einen k-NN Graph aus einer Punktwolke
        
        Args:
            pos: Positionen [N, 3]
            features: Optionale Features [N, F]
            batch: Batch-Zuordnung [N]
            
        Returns:
            PyG Data Objekt
        """
        # Erstelle k-NN Graph
        edge_index = knn_graph(pos, k=self.k_neighbors, batch=batch)
        
        # Features sind Positionen wenn keine anderen gegeben
        x = features if features is not None else pos
        
        return Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward Pass durch das Netzwerk
        
        Args:
            data: PyG Data Objekt mit x, pos, edge_index, batch
            
        Returns:
            Logits für jede Klasse [B, num_classes]
        """
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        
        # Durchlaufe PointNet++ Layers
        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = F.relu(self.conv3(x, pos, edge_index))
        
        # Global Max Pooling
        x = global_max_pool(x, batch)
        
        # Klassifikation
        logits = self.classifier(x)
        
        return logits
    
    def forward_tensordict(self, td: TensorDict) -> TensorDict:
        """
        Forward Pass mit TensorDict Interface
        
        Args:
            td: TensorDict mit 'positions', optional 'features' und 'batch'
            
        Returns:
            TensorDict mit 'logits' und 'predictions'
        """
        # Extrahiere Daten aus TensorDict
        pos = td["positions"]
        features = td.get("features", None)
        batch = td.get("batch", None)
        
        # Erstelle Graph
        data = self.create_graph_from_pointcloud(pos, features, batch)
        
        # Forward Pass
        logits = self.forward(data)
        predictions = torch.argmax(logits, dim=-1)
        
        # Erstelle Output TensorDict
        output = TensorDict({
            "logits": logits,
            "predictions": predictions,
            "embeddings": data.x  # Letzte Feature-Repräsentation
        }, batch_size=td.batch_size)
        
        return output
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Extrahiert Feature-Embeddings ohne Klassifikation
        
        Nützlich für:
        - Visualisierung (t-SNE, UMAP)
        - Transfer Learning
        - Ähnlichkeitsanalyse
        """
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        
        # Durchlaufe PointNet++ Layers
        x = F.relu(self.conv1(x, pos, edge_index))
        x = F.relu(self.conv2(x, pos, edge_index))
        x = F.relu(self.conv3(x, pos, edge_index))
        
        # Global Max Pooling
        embeddings = global_max_pool(x, batch)
        
        return embeddings
