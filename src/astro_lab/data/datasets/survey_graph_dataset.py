"""
Survey Graph Dataset
===================

Simplified PyTorch Geometric Dataset using existing components.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import KNNGraph

from astro_lab.config import get_data_config, get_survey_config
from astro_lab.data.datasets.base import AstroDatasetBase
from astro_lab.data.preprocessors import get_preprocessor
from astro_lab.memory import clear_cuda_cache

logger = logging.getLogger(__name__)


class SurveyGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for astronomical survey data.

    Uses existing components:
    - PyG transforms for graph construction
    - Preprocessors for data loading
    - Central config for all settings
    """

    def __init__(
        self,
        root: str,
        survey: str,
        task: str = "node_classification",
        max_samples: Optional[int] = None,
        k_neighbors: Optional[int] = None,
        use_3d_coordinates: bool = True,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        force_reload: bool = False,
        **kwargs,
    ):
        """
        Initialize survey graph dataset.

        Args:
            root: Root directory for dataset
            survey: Survey name (gaia, sdss, etc.)
            task: Task type (node_classification, graph_classification, etc.)
            max_samples: Maximum number of samples for development/testing
            k_neighbors: Number of k-nearest neighbors for graph construction
            use_3d_coordinates: Whether to use 3D coordinates
            transform: Optional transform to apply
            pre_transform: Optional pre-transform to apply
            pre_filter: Optional pre-filter to apply
            force_reload: Whether to force reload data
            **kwargs: Additional arguments
        """
        self.use_3d_coordinates = use_3d_coordinates
        self.force_reload = force_reload
        self.survey = survey
        self.task = task
        self.max_samples = max_samples

        # Get configurations
        self.survey_config = get_survey_config(survey)
        self.data_config = get_data_config()

        # Set k_neighbors with defaults
        self.k_neighbors = k_neighbors or self.survey_config.get("k_neighbors", 8)

        # Initialize base class
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load data
        self._load_data()

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names."""
        return [f"{self.survey}_raw.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        filename = f"{self.survey}_k{self.k_neighbors}"
        if self.max_samples:
            filename += f"_max{self.max_samples}"
        return [f"{filename}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        if not raw_path.exists():
            logger.info(f"Downloading {self.survey} data...")
            preprocessor = get_preprocessor(self.survey)
            df = preprocessor.load_data()
            df.write_parquet(raw_path)
            logger.info(f"Saved to {raw_path}")

    def process(self):
        """Process raw data into graph format."""
        logger.info(f"Processing {self.survey} data...")

        # Load data via preprocessor
        preprocessor = get_preprocessor(self.survey)
        df = preprocessor.load_data()

        # Convert to TensorDict
        tensor_dict = preprocessor.create_tensordict(df)

        # Build graph using PyG transforms
        graph = self._build_graph(tensor_dict)

        # Add metadata
        graph.survey_name = self.survey
        graph.num_nodes = graph.x.size(0) if hasattr(graph, "x") else 0
        setattr(
            graph,
            "num_edges",
            graph.edge_index.size(1) if hasattr(graph, "edge_index") else 0,
        )

        # Save processed data
        processed_path = Path(self.processed_dir) / self.processed_file_names[0]
        torch.save(graph, processed_path)
        logger.info(f"Saved graph to {processed_path}")

    def _build_graph(self, tensor_dict) -> Data:
        """Build graph using PyG transforms."""
        # Extract features from tensor_dict
        if "features" in tensor_dict:
            features = tensor_dict["features"]
        else:
            features = None

        # Extract coordinates from tensor_dict
        if "spatial" in tensor_dict:
            coords = tensor_dict["spatial"].get(
                "coordinates", tensor_dict["spatial"].get("pos", None)
            )
        elif features is not None:
            coords = features[:, :3] if features.shape[1] >= 3 else features
        elif "x" in tensor_dict:
            coords = tensor_dict["x"]
        else:
            raise ValueError("No coordinate data found in tensor_dict")

        if coords is None:
            raise ValueError("No coordinates found")

        # Create PyG Data object with features or coordinates
        if features is not None:
            graph = Data(x=features, pos=coords)
        else:
            graph = Data(x=coords, pos=coords)

        # Add labels if available in tensor_dict
        if "labels" in tensor_dict:
            graph.y = tensor_dict["labels"]
        elif "y" in tensor_dict:
            graph.y = tensor_dict["y"]
        else:
            # Create synthetic labels for demonstration/unsupervised learning
            # Based on spatial clustering or feature quantiles
            num_nodes = coords.size(0)
            if features is not None and features.size(1) > 0:
                # Use first feature for quantile-based labels
                first_feature = features[:, 0]
                quantiles = torch.quantile(
                    first_feature, torch.tensor([0.2, 0.4, 0.6, 0.8])
                )
                labels = torch.zeros(num_nodes, dtype=torch.long)
                for i, q in enumerate(quantiles):
                    labels[first_feature > q] = i + 1
                graph.y = labels
            else:
                # Use spatial clustering for labels
                from sklearn.cluster import KMeans

                kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(coords.cpu().numpy())
                graph.y = torch.tensor(labels, dtype=torch.long)

        # Apply k-NN transform
        knn_transform = KNNGraph(k=self.k_neighbors)
        graph = knn_transform(graph)

        return graph

    def _load_data(self):
        """Load or process data."""
        processed_path = Path(self.processed_dir) / self.processed_file_names[0]

        if processed_path.exists() and not self.force_reload:
            try:
                logger.info(f"Loading cached graph from {processed_path}")
                self.data = torch.load(processed_path, weights_only=False)
                clear_cuda_cache()
                return
            except Exception as e:
                logger.warning(f"Failed to load cached graph: {e}")

        # Process if needed
        if not Path(self.raw_dir).exists():
            Path(self.raw_dir).mkdir(parents=True, exist_ok=True)
            self.download()

        if not processed_path.exists():
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            self.process()

        # Load processed data
        self.data = torch.load(processed_path, weights_only=False)
        clear_cuda_cache()

    def len(self) -> int:
        """Number of graphs (always 1 for single graph datasets)."""
        return 1

    def get(self, idx: int) -> Data:
        """Get graph by index."""
        if idx != 0:
            raise IndexError(f"Index {idx} out of range for single graph dataset")
        return self.data

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if not hasattr(self, "data"):
            return {"error": "Dataset not loaded"}

        return {
            "survey": self.survey,
            "k_neighbors": self.k_neighbors,
            "max_samples": self.max_samples,
            "num_nodes": getattr(self.data, "num_nodes", 0),
            "num_edges": getattr(self.data, "num_edges", 0),
            "num_features": self.get_feature_dim(),
            "num_classes": self.get_num_classes(),
            "survey_config": {
                "name": self.survey_config.get("name", self.survey),
                "type": self.survey_config.get("type", "catalog"),
                "coordinate_system": self.survey_config.get(
                    "coordinate_system", "icrs"
                ),
            },
        }

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        if not hasattr(self, "data") or not hasattr(self.data, "x"):
            return 0
        return self.data.x.size(1)

    def get_num_classes(self) -> int:
        """Get number of classes."""
        if not hasattr(self, "data") or not hasattr(self.data, "y"):
            return 0
        return int(self.data.y.max().item() + 1)
