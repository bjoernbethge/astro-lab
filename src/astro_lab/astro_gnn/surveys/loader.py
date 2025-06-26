"""
Survey Data Loader für AstroGNN

Lädt vorverarbeitete Survey-Daten und erstellt PyTorch Geometric Datasets
für das AstroGNN Training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from ..data import AstroPointCloudDataset
from .preprocessing import preprocess_survey_for_pointcloud, create_survey_pointcloud_graph

logger = logging.getLogger(__name__)


class SurveyDataLoader:
    """
    Vereinfachter Loader für astronomische Survey-Daten.
    
    Unterstützt:
    - Gaia DR3
    - SDSS
    - NSA
    - TNG50 Simulationen
    """
    
    def __init__(
        self,
        survey: str,
        data_root: str = "data",
        num_points: int = 1024,
        k_neighbors: int = 16,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize Survey Data Loader.
        
        Args:
            survey: Survey name ('gaia', 'sdss', 'nsa', 'tng50')
            data_root: Root directory for data
            num_points: Number of points per sample
            k_neighbors: Number of neighbors for graph
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
        """
        self.survey = survey.lower()
        self.data_root = Path(data_root)
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Survey-spezifische Pfade
        self.raw_dir = self.data_root / "raw" / self.survey
        self.processed_dir = self.data_root / "processed" / self.survey
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache für geladene Daten
        self._data_cache = {}
    
    def load_and_preprocess(
        self,
        max_samples: Optional[int] = None,
        force_reload: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Lädt und preprocessed Survey-Daten.
        
        Args:
            max_samples: Maximale Anzahl von Samples
            force_reload: Erzwinge Neuladen der Daten
            
        Returns:
            Dictionary mit Tensoren
        """
        # Check Cache
        cache_key = f"{self.survey}_{max_samples}"
        if not force_reload and cache_key in self._data_cache:
            logger.info(f"📦 Using cached data for {self.survey}")
            return self._data_cache[cache_key]
        
        # Finde Input-Datei
        input_file = self._find_input_file()
        if not input_file:
            raise FileNotFoundError(f"No data file found for survey {self.survey}")
        
        logger.info(f"🔄 Loading {self.survey} data from {input_file}")
        
        # Preprocess für Punktwolken
        result = preprocess_survey_for_pointcloud(
            survey=self.survey,
            input_path=input_file,
            output_dir=self.processed_dir,
            max_samples=max_samples,
            normalize_positions=True,
            add_velocity_features=True
        )
        
        # Cache Result
        self._data_cache[cache_key] = result
        
        return result
    
    def create_dataset(
        self,
        split: str = "full",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples: Optional[int] = None
    ) -> Union[AstroPointCloudDataset, Tuple[Dataset, Dataset, Dataset]]:
        """
        Erstellt PyTorch Geometric Dataset(s).
        
        Args:
            split: 'full', 'train', 'val', 'test' oder 'all' (für train/val/test)
            train_ratio: Anteil für Training
            val_ratio: Anteil für Validierung
            max_samples: Maximale Anzahl Samples
            
        Returns:
            Dataset oder Tuple von Datasets (train, val, test)
        """
        # Lade Daten
        data = self.load_and_preprocess(max_samples)
        
        positions = data["positions"]
        features = data["features"]
        labels = data["labels"]
        
        n_samples = len(positions)
        
        if split == "full":
            # Einzelnes Dataset mit allen Daten
            dataset = SurveyPointCloudDataset(
                positions=positions,
                features=features,
                labels=labels,
                survey_name=self.survey,
                num_points=self.num_points,
                k_neighbors=self.k_neighbors
            )
            return dataset
            
        elif split == "all":
            # Train/Val/Test Split
            indices = torch.randperm(n_samples)
            
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]
            
            # Erstelle Datasets
            train_dataset = SurveyPointCloudDataset(
                positions=positions[train_idx],
                features=features[train_idx],
                labels=labels[train_idx] if labels is not None else None,
                survey_name=self.survey,
                num_points=self.num_points,
                k_neighbors=self.k_neighbors
            )
            
            val_dataset = SurveyPointCloudDataset(
                positions=positions[val_idx],
                features=features[val_idx],
                labels=labels[val_idx] if labels is not None else None,
                survey_name=self.survey,
                num_points=self.num_points,
                k_neighbors=self.k_neighbors
            )
            
            test_dataset = SurveyPointCloudDataset(
                positions=positions[test_idx],
                features=features[test_idx],
                labels=labels[test_idx] if labels is not None else None,
                survey_name=self.survey,
                num_points=self.num_points,
                k_neighbors=self.k_neighbors
            )
            
            return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Erstellt DataLoader für Training.
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Erstelle Datasets
        train_dataset, val_dataset, test_dataset = self.create_dataset(
            split="all",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_samples=max_samples
        )
        
        # Erstelle DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_survey_info(self) -> Dict[str, any]:
        """
        Gibt Informationen über den Survey zurück.
        """
        # Lade Daten um Info zu bekommen
        data = self.load_and_preprocess(max_samples=100)  # Kleine Probe
        
        return data["survey_info"]
    
    def _find_input_file(self) -> Optional[Path]:
        """
        Findet die Input-Datei für den Survey.
        """
        # Suche in verschiedenen Formaten und Orten
        possible_files = [
            # Processed
            self.processed_dir / f"{self.survey}_pointcloud.parquet",
            self.processed_dir / f"{self.survey}.parquet",
            # Raw
            self.raw_dir / f"{self.survey}.parquet",
            self.raw_dir / f"{self.survey}.csv",
            # Alternative Namen
            self.data_root / "raw" / f"{self.survey.upper()}" / "*.parquet",
            self.data_root / "raw" / f"{self.survey.lower()}" / "*.parquet",
        ]
        
        for pattern in possible_files:
            if "*" in str(pattern):
                # Glob pattern
                parent = pattern.parent
                glob_pattern = pattern.name
                if parent.exists():
                    matches = list(parent.glob(glob_pattern))
                    if matches:
                        return matches[0]
            elif pattern.exists():
                return pattern
        
        return None


class SurveyPointCloudDataset(Dataset):
    """
    PyTorch Geometric Dataset für Survey-Punktwolken.
    
    Jeder Sample ist ein Ausschnitt mit `num_points` Punkten.
    """
    
    def __init__(
        self,
        positions: torch.Tensor,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        survey_name: str = "unknown",
        num_points: int = 1024,
        k_neighbors: int = 16,
        transform=None,
        pre_transform=None
    ):
        self.positions = positions
        self.features = features
        self.labels = labels
        self.survey_name = survey_name
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        
        super().__init__(None, transform, pre_transform)
    
    def len(self) -> int:
        """Anzahl der möglichen Samples."""
        # Für große Surveys: Overlapping Windows
        return max(1, len(self.positions) // (self.num_points // 2))
    
    def get(self, idx: int) -> Data:
        """
        Gibt einen Sample zurück.
        
        Extrahiert ein Fenster von `num_points` Punkten und
        erstellt einen lokalen Graph.
        """
        # Berechne Start-Index mit Overlap
        start_idx = idx * (self.num_points // 2)
        end_idx = min(start_idx + self.num_points, len(self.positions))
        
        # Falls nicht genug Punkte, wiederhole letzte
        if end_idx - start_idx < self.num_points:
            # Padding durch Wiederholung
            indices = list(range(start_idx, end_idx))
            while len(indices) < self.num_points:
                indices.extend(indices[:min(len(indices), self.num_points - len(indices))])
            indices = indices[:self.num_points]
            indices = torch.tensor(indices)
        else:
            indices = torch.arange(start_idx, end_idx)
        
        # Extrahiere Daten
        pos = self.positions[indices]
        x = self.features[indices]
        y = self.labels[indices] if self.labels is not None else None
        
        # Erstelle lokalen Graph
        data = create_survey_pointcloud_graph(
            positions=pos,
            features=x,
            labels=y,
            k_neighbors=self.k_neighbors,
            survey_name=self.survey_name
        )
        
        # Füge Index hinzu (für Tracking)
        data.idx = idx
        
        return data


# Convenience Functions
def load_gaia_pointclouds(
    max_samples: Optional[int] = 10000,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Lädt Gaia DR3 Sternhaufen als Punktwolken."""
    loader = SurveyDataLoader(
        survey="gaia",
        num_points=1024,
        k_neighbors=16,
        batch_size=batch_size
    )
    return loader.create_dataloaders(max_samples=max_samples)


def load_sdss_galaxies(
    max_samples: Optional[int] = 5000,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Lädt SDSS Galaxien als Punktwolken."""
    loader = SurveyDataLoader(
        survey="sdss",
        num_points=512,
        k_neighbors=8,
        batch_size=batch_size
    )
    return loader.create_dataloaders(max_samples=max_samples)


def load_tng50_simulation(
    max_samples: Optional[int] = 50000,
    batch_size: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Lädt TNG50 Simulationsdaten als Punktwolken."""
    loader = SurveyDataLoader(
        survey="tng50",
        num_points=2048,
        k_neighbors=32,
        batch_size=batch_size
    )
    return loader.create_dataloaders(max_samples=max_samples)
