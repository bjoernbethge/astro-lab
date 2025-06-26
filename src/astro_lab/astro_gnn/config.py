"""
Konfigurationsmanagement für AstroGNN

Dieses Modul verwaltet alle Konfigurationen über YAML-Dateien.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """Konfiguration für das GNN-Modell"""
    input_features: int = 3  # x, y, z Koordinaten
    hidden_dim: int = 128
    output_classes: int = 7  # z.B. Sterntypen O, B, A, F, G, K, M
    num_layers: int = 3
    dropout: float = 0.1
    k_neighbors: int = 16  # für k-NN Graph
    aggregation: str = "max"  # max, mean, add


@dataclass
class DataConfig:
    """Konfiguration für Daten"""
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    num_points: int = 1024  # Anzahl Punkte pro Sample
    normalize: bool = True
    augmentation: bool = True


@dataclass
class TrainingConfig:
    """Konfiguration für Training"""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "cosine"  # cosine, step, plateau
    patience: int = 10
    min_lr: float = 1e-6
    gradient_clip: Optional[float] = 1.0


@dataclass
class Config:
    """Haupt-Konfiguration"""
    experiment_name: str = "astro_pointcloud_classification"
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, auto
    
    # Sub-Konfigurationen
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Pfade
    data_path: str = "./data"
    output_path: str = "./outputs"
    checkpoint_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Lädt Konfiguration aus YAML-Datei"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Erstelle verschachtelte Konfigurationen
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            config_dict['data'] = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
            
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Speichert Konfiguration als YAML-Datei"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'seed': self.seed,
            'device': self.device,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'data_path': self.data_path,
            'output_path': self.output_path,
            'checkpoint_path': self.checkpoint_path
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def __str__(self) -> str:
        """Schöne String-Repräsentation für Logging"""
        return yaml.dump(self.__dict__, default_flow_style=False, sort_keys=False)
