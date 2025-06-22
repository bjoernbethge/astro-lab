"""
AstroDataset - Clean Dataset Implementation
==========================================

Clean dataset implementation for loading astronomical data.
No Lightning modules or other mixtures.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)


class AstroDataset(InMemoryDataset):
    """
    Clean dataset class for astronomical data.

    Uses standardized file structures:
    - data/processed/{survey}/{survey}.parquet
    - data/processed/{survey}/{survey}_graph_k{k}.pt
    - data/processed/{survey}/{survey}_metadata.json
    """

    def __init__(
        self,
        survey: str,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        root: Optional[str] = None,
        transform=None,
        **kwargs,
    ):
        self.survey = survey
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples

        # Standardized paths
        if root is None:
            project_root = Path(__file__).parent.parent.parent.parent
            root = str(project_root / "data" / "processed" / survey)

        super().__init__(root, transform)

    def _get_file_paths(self) -> Dict[str, Path]:
        """Standardized file paths."""
        root_path = Path(self.root)
        return {
            "parquet": root_path / f"{self.survey}.parquet",
            "graph": root_path / f"{self.survey}_graph.pt",
            "metadata": root_path / f"{self.survey}_metadata.json",
        }

    def _load_metadata(self) -> Dict[str, Any]:
        """Lade Metadaten."""
        paths = self._get_file_paths()
        metadata_path = paths["metadata"]

        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"⚠️ Metadata load failed: {e}")

        return {
            "survey_name": self.survey,
            "k_neighbors": self.k_neighbors,
            "num_samples": 0,
        }

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        paths = self._get_file_paths()
        return [paths["graph"].name]

    def download(self):
        """Keine Downloads - Daten müssen bereits vorhanden sein."""
        pass

    def process(self):
        """Keine Verarbeitung - .pt Dateien müssen bereits vorhanden sein."""
        pass

    def _load_graph_data(self):
        """Lade Graph-Daten aus .pt Datei."""
        paths = self._get_file_paths()
        graph_path = paths["graph"]

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        logger.info(f"📦 Loading graph data from {graph_path}")

        try:
            data = torch.load(graph_path, map_location="cpu", weights_only=False)

            # Handle verschiedene Datenformate
            if isinstance(data, dict) and "data" in data and "slices" in data:
                self.data, self.slices = data["data"], data["slices"]
            elif isinstance(data, list):
                valid_data = [
                    item
                    for item in data
                    if isinstance(item, Data) and item.x is not None
                ]
                if not valid_data:
                    raise ValueError("No valid Data objects")
                self.data, self.slices = self.collate(valid_data)
            elif isinstance(data, Data):
                if data.x is None:
                    raise ValueError("Invalid Data object - no features")
                self.data, self.slices = self.collate([data])
            else:
                raise ValueError(f"Unknown data format: {type(data)}")

            logger.info("✅ Loaded graph data successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load {graph_path}: {e}")
            raise

    def len(self):
        if self.data is None:
            self._load_graph_data()

        # Handle single graph case
        if self.slices is None:
            # Single graph dataset
            if hasattr(self.data, "num_nodes"):
                return 1
            return 0

        # Multiple graphs case
        return len(self.slices[list(self.slices.keys())[0]]) - 1

    def get(self, idx):
        if self.data is None:
            self._load_graph_data()

        # Handle single graph case
        if self.slices is None:
            if idx == 0:
                return self.data
            else:
                raise IndexError(f"Index {idx} out of range for single graph dataset")

        # Multiple graphs case
        return super().get(idx)

    def get_info(self) -> Dict[str, Any]:
        """Dataset-Informationen."""
        if len(self) == 0:
            return {"error": "Dataset empty"}

        sample = self[0]
        metadata = self._load_metadata()

        return {
            "survey": self.survey,
            "num_samples": len(self),
            "num_nodes": sample.num_nodes if hasattr(sample, "num_nodes") else 0,
            "num_edges": sample.edge_index.shape[1]
            if hasattr(sample, "edge_index")
            else 0,
            "num_features": sample.x.shape[1] if hasattr(sample, "x") else 0,
            "k_neighbors": self.k_neighbors,
            **metadata,
        }
 