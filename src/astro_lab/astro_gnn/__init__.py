"""
AstroGNN - Graph Neural Networks for Astronomical Point Cloud Analysis

Ein modulares Framework zur Analyse astronomischer Punktwolken mit PyTorch Geometric
und TensorDict Integration.

Autor: [Ihr Name]
Projekt: Abschlussarbeit
"""

from .model import AstroPointCloudGNN
from .data import AstroPointCloudDataset
from .trainer import AstroGNNTrainer

__version__ = "0.1.0"
__all__ = ["AstroPointCloudGNN", "AstroPointCloudDataset", "AstroGNNTrainer"]
