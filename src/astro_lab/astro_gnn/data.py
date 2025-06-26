"""
Datenverarbeitung für astronomische Punktwolken

Dieses Modul stellt Dataset-Klassen und Utilities für die Verarbeitung
astronomischer Punktwolken bereit.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader
from tensordict import TensorDict
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
import h5py
import pandas as pd
import polars as pl

# Import Survey-Integration
try:
    from .surveys import SurveyDataLoader, preprocess_survey_for_pointcloud
    HAS_SURVEY_INTEGRATION = True
except ImportError:
    HAS_SURVEY_INTEGRATION = False


class AstroPointCloudDataset(Dataset):
    """
    Dataset für astronomische Punktwolken
    
    Unterstützt verschiedene Datenformate:
    - HDF5 mit Sternkatalogen
    - CSV mit Koordinaten und Features
    - Numpy Arrays
    
    Features können sein:
    - Position (RA, Dec, Distanz oder x, y, z)
    - Helligkeit (verschiedene Bänder)
    - Farbe (B-V, G-R, etc.)
    - Kinematik (Eigenbewegung, Radialgeschwindigkeit)
    """
    
    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        num_points: int = 1024,
        normalize: bool = True,
        feature_columns: Optional[List[str]] = None
    ):
        self.num_points = num_points
        self.normalize = normalize
        self.feature_columns = feature_columns or ['x', 'y', 'z']
        
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        """Liste der erwarteten Rohdateien"""
        return ['star_clusters.h5', 'star_clusters.csv']
    
    @property
    def processed_file_names(self) -> List[str]:
        """Liste der prozessierten Dateien"""
        return [f'data_{i}.pt' for i in range(self.len())]
    
    def download(self):
        """Download oder Kopiere Rohdaten (falls nötig)"""
        # Hier würde man echte Daten herunterladen
        # Für die Demo erstellen wir synthetische Daten
        self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """Erstellt synthetische Sternhaufen-Daten für Tests"""
        import h5py
        
        n_clusters = 100
        n_stars_per_cluster = 2048
        
        # Sterntypen: O, B, A, F, G, K, M
        star_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        
        with h5py.File(self.raw_paths[0], 'w') as f:
            for i in range(n_clusters):
                # Simuliere verschiedene Cluster-Typen
                cluster_type = np.random.choice(len(star_types))
                
                # Generiere Sternpositionen (Gauss-verteilt um Zentrum)
                center = np.random.randn(3) * 100
                positions = np.random.randn(n_stars_per_cluster, 3) * 10 + center
                
                # Generiere Features basierend auf Sterntyp
                # Helligkeit, Farbe, Temperatur
                features = np.zeros((n_stars_per_cluster, 4))
                features[:, 0] = np.random.normal(5.0 - cluster_type, 1.0)  # Magnitude
                features[:, 1] = np.random.normal(cluster_type * 0.3, 0.1)  # B-V Color
                features[:, 2] = np.random.normal(7000 - cluster_type * 800, 200)  # Temperature
                features[:, 3] = np.random.normal(1.0, 0.2)  # Mass
                
                # Speichere Cluster
                grp = f.create_group(f'cluster_{i}')
                grp.create_dataset('positions', data=positions)
                grp.create_dataset('features', data=features)
                grp.attrs['label'] = cluster_type
                grp.attrs['name'] = f'Synthetic_Cluster_{i}'
    
    def process(self):
        """Prozessiert Rohdaten zu PyG Data Objekten"""
        with h5py.File(self.raw_paths[0], 'r') as f:
            for i, cluster_name in enumerate(f.keys()):
                cluster = f[cluster_name]
                
                # Lade Daten
                positions = torch.tensor(cluster['positions'][:], dtype=torch.float32)
                features = torch.tensor(cluster['features'][:], dtype=torch.float32)
                label = torch.tensor(cluster.attrs['label'], dtype=torch.long)
                
                # Sample Punkte wenn nötig
                if len(positions) > self.num_points:
                    indices = torch.randperm(len(positions))[:self.num_points]
                    positions = positions[indices]
                    features = features[indices]
                elif len(positions) < self.num_points:
                    # Padding mit Wiederholung
                    indices = torch.randint(0, len(positions), (self.num_points,))
                    positions = positions[indices]
                    features = features[indices]
                
                # Normalisiere Positionen
                if self.normalize:
                    positions = positions - positions.mean(dim=0)
                    scale = positions.abs().max()
                    if scale > 0:
                        positions = positions / scale
                
                # Erstelle Data Objekt
                data = Data(
                    x=features,
                    pos=positions,
                    y=label
                )
                
                # Speichere prozessierte Daten
                torch.save(data, self.processed_paths[i])
    
    def len(self) -> int:
        """Anzahl der Samples im Dataset"""
        if Path(self.raw_paths[0]).exists():
            with h5py.File(self.raw_paths[0], 'r') as f:
                return len(f.keys())
        return 0
    
    def get(self, idx: int) -> Data:
        """Lädt ein einzelnes Sample"""
        data = torch.load(self.processed_paths[idx])
        return data


def create_dataloaders(
    dataset: AstroPointCloudDataset,
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Erstellt Train/Val/Test DataLoader
    
    Args:
        dataset: AstroPointCloudDataset
        batch_size: Batch-Größe
        train_split: Anteil für Training
        val_split: Anteil für Validatierung
        num_workers: Anzahl Worker für DataLoader
        seed: Random Seed für Reproduzierbarkeit
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Berechne Split-Größen
    n_samples = len(dataset)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    n_test = n_samples - n_train - n_val
    
    # Erstelle zufällige Splits
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )
    
    # Erstelle DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def collate_tensordict(batch: List[Data]) -> TensorDict:
    """
    Custom collate function für TensorDict
    
    Konvertiert PyG Data Batch zu TensorDict für
    kompatibilität mit anderen Frameworks.
    """
    # Stack positions und features
    positions = torch.stack([data.pos for data in batch])
    features = torch.stack([data.x for data in batch])
    labels = torch.tensor([data.y for data in batch])
    
    # Erstelle batch assignment
    batch_size = len(batch)
    num_points = positions.shape[1]
    batch_assignment = torch.repeat_interleave(
        torch.arange(batch_size), num_points
    )
    
    return TensorDict({
        "positions": positions.reshape(-1, 3),  # Flatten für PyG
        "features": features.reshape(-1, features.shape[-1]),
        "labels": labels,
        "batch": batch_assignment
    }, batch_size=[batch_size])
