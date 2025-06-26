"""
Survey Graph Dataset
===================

Loads SurveyTensorDict data and builds PyG graphs using centralized graph builders.
Provides a clean pipeline: SurveyTensorDict → Graph → InMemoryDataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset

from astro_lab.data.graphs import create_astronomical_graph, create_knn_graph
from astro_lab.tensors import PhotometricTensorDict, SpatialTensorDict, SurveyTensorDict
from astro_lab.utils.config.surveys import get_survey_config

logger = logging.getLogger(__name__)


class SurveyGraphDataset(InMemoryDataset):
    """
    PyG Dataset that loads SurveyTensorDict data and builds graphs.

    Pipeline: Raw Data → SurveyTensorDict → Graph → InMemoryDataset
    """

    def __init__(
        self,
        root: str,
        survey: str,
        graph_method: str = "knn",
        k_neighbors: int = 8,
        use_3d_coordinates: bool = True,
        max_samples: Optional[int] = None,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        **kwargs,
    ):
        self.survey = survey
        self.graph_method = graph_method
        self.k_neighbors = k_neighbors
        self.use_3d_coordinates = use_3d_coordinates
        self.max_samples = max_samples
        self.survey_config = get_survey_config(survey)

        super().__init__(root, transform, pre_transform, pre_filter)

        # Setup paths
        self._processed_dir = Path(self.root) / "processed" / self.survey
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        # Graph file path
        graph_filename = f"{self.survey}_{graph_method}_k{k_neighbors}"
        if self.use_3d_coordinates:
            graph_filename += "_3d"
        graph_filename += ".pt"
        self._graph_path = self._processed_dir / graph_filename

        # SurveyTensorDict file path (for debugging/analysis)
        self._tensor_path = self._processed_dir / f"{self.survey}_tensor.pt"

        # Load data
        self._load_data()

    def _load_data(self):
        """Load data and build graph if needed."""
        if self._graph_path.exists():
            logger.info(f"🔄 Loading existing graph: {self._graph_path}")
            try:
                graph_data = torch.load(self._graph_path, weights_only=False)
                self.data, self.slices = self.collate([graph_data])
                return
            except Exception as e:
                logger.warning(f"⚠️ Failed to load graph: {e}. Rebuilding...")

        # Build graph from SurveyTensorDict
        logger.info(f"🔄 Building graph for {self.survey} using {self.graph_method}")
        survey_tensor = self._load_survey_tensor()
        graph_data = self._build_graph(survey_tensor)

        # Save graph
        torch.save(graph_data, self._graph_path)
        logger.info(f"💾 Graph saved: {self._graph_path}")

        # Save SurveyTensorDict for debugging
        torch.save(survey_tensor, self._tensor_path)
        logger.info(f"💾 SurveyTensorDict saved: {self._tensor_path}")

        # Set data for PyG
        self.data, self.slices = self.collate([graph_data])

    def _load_survey_tensor(self) -> SurveyTensorDict:
        """Load or create SurveyTensorDict from raw data."""
        # Try to load existing SurveyTensorDict
        if self._tensor_path.exists():
            try:
                logger.info(f"🔄 Loading SurveyTensorDict: {self._tensor_path}")
                return torch.load(self._tensor_path, weights_only=False)
            except Exception:
                logger.warning(
                    "⚠️ Failed to load SurveyTensorDict. Creating from raw data..."
                )

        # Create from raw data
        logger.info(f"🔄 Creating SurveyTensorDict from raw data for {self.survey}")
        return self._create_survey_tensor_from_raw()

    def _create_survey_tensor_from_raw(self) -> SurveyTensorDict:
        """Create SurveyTensorDict from raw survey data."""
        # Load raw data (CSV, Parquet, etc.)
        raw_data_path = self._find_raw_data()
        if raw_data_path is None:
            raise FileNotFoundError(f"No raw data found for survey: {self.survey}")

        logger.info(f"🔄 Loading raw data: {raw_data_path}")

        # Load with Polars
        if raw_data_path.suffix == ".parquet":
            df = pl.read_parquet(raw_data_path)
        elif raw_data_path.suffix == ".csv":
            df = pl.read_csv(raw_data_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_data_path.suffix}")

        # Apply sampling if requested
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, seed=42)
            logger.info(f"📊 Sampled {self.max_samples} objects from {len(df)} total")

        # Create SurveyTensorDict
        return self._dataframe_to_survey_tensor(df)

    def _find_raw_data(self) -> Optional[Path]:
        """Find raw data file for the survey."""
        # Common locations - check processed first, then raw
        search_paths = [
            Path("data/processed") / self.survey / f"{self.survey}.parquet",
            Path("data/processed") / self.survey / f"{self.survey}.csv",
            Path("data/raw") / f"{self.survey}.parquet",
            Path("data/raw") / f"{self.survey}.csv",
            Path("data") / f"{self.survey}.parquet",
            Path("data") / f"{self.survey}.csv",
        ]

        for path in search_paths:
            if path.exists():
                return path

        return None

    def _dataframe_to_survey_tensor(self, df: pl.DataFrame) -> SurveyTensorDict:
        """
        Convert a DataFrame to a SurveyTensorDict.
        Always pass a tensor to SpatialTensorDict (not a dict),
        and ensure all meta fields are set as in other TensorDicts.
        """
        # Get column configuration
        coord_cols = self.survey_config.get("coord_cols", ["ra", "dec"])
        mag_cols = self.survey_config.get("mag_cols", [])

        # Extract coordinates as tensor [N, D]
        coords = []
        for col in coord_cols:
            if col in df.columns:
                coords.append(torch.tensor(df[col].to_numpy(), dtype=torch.float32))
        if not coords:
            raise ValueError(f"No coordinate columns found in DataFrame: {coord_cols}")
        coordinates = torch.stack(coords, dim=1)  # [N, D]
        # If only 2D, pad to 3D with zeros (for compatibility)
        if coordinates.shape[1] == 2:
            zeros = torch.zeros(coordinates.shape[0], 1, dtype=coordinates.dtype)
            coordinates = torch.cat([coordinates, zeros], dim=1)
        spatial_tensor = SpatialTensorDict(coordinates=coordinates)

        # Extract photometric data if available
        photometric_tensor = None
        if mag_cols:
            mags = []
            bands = []
            for col in mag_cols:
                if col in df.columns:
                    mags.append(torch.tensor(df[col].to_numpy(), dtype=torch.float32))
                    bands.append(col)
            if mags:
                magnitudes = torch.stack(mags, dim=1)  # [N, B]
                photometric_tensor = PhotometricTensorDict(
                    magnitudes=magnitudes, bands=bands, filter_system="AB"
                )

        # Build SurveyTensorDict with correct signature
        survey_name = self.survey
        data_release = self.survey_config.get("data_release", "unknown")
        if photometric_tensor is not None:
            return SurveyTensorDict(
                spatial=spatial_tensor,
                photometric=photometric_tensor,
                survey_name=survey_name,
                data_release=data_release,
            )
        else:
            # Minimal SurveyTensorDict (for tests or special cases)
            # Use dummy photometric tensor if required by constructor
            dummy_mags = torch.zeros(coordinates.shape[0], 1)
            dummy_bands = ["dummy"]
            dummy_phot = PhotometricTensorDict(dummy_mags, dummy_bands)
            return SurveyTensorDict(
                spatial=spatial_tensor,
                photometric=dummy_phot,
                survey_name=survey_name,
                data_release=data_release,
            )

    def _build_graph(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build PyG graph from SurveyTensorDict using centralized builders."""
        logger.info(f"🔗 Building {self.graph_method} graph with k={self.k_neighbors}")

        if self.graph_method == "knn":
            graph = create_knn_graph(
                survey_tensor,
                k_neighbors=self.k_neighbors,
                use_3d_coordinates=self.use_3d_coordinates,
            )
        elif self.graph_method == "astronomical":
            graph = create_astronomical_graph(
                survey_tensor,
                k_neighbors=self.k_neighbors,
                use_3d_coordinates=self.use_3d_coordinates,
            )
        else:
            raise ValueError(f"Unknown graph method: {self.graph_method}")

        # Add survey metadata
        graph.survey_name = self.survey
        graph.graph_method = self.graph_method
        graph.k_neighbors = self.k_neighbors
        graph.use_3d = self.use_3d_coordinates

        # Defensive: Only log node/edge count if present
        num_nodes = getattr(graph, "num_nodes", None)
        num_edges = getattr(graph, "num_edges", None)
        logger.info(
            f"✅ Built graph: {num_nodes if num_nodes is not None else '?'} nodes, {num_edges if num_edges is not None else '?'} edges"
        )
        return graph

    def len(self) -> int:
        """Number of graphs in dataset."""
        return 1  # Single graph dataset

    def get(self, idx: int) -> Data:
        """Get graph by index."""
        if idx == 0:
            return self.data
        else:
            raise IndexError(f"Index {idx} out of range for single graph dataset")

    def get_survey_tensor(self) -> SurveyTensorDict:
        """Get the underlying SurveyTensorDict (for analysis/debugging)."""
        if self._tensor_path.exists():
            return torch.load(self._tensor_path, weights_only=False)
        else:
            raise FileNotFoundError(
                "SurveyTensorDict not found. Run _load_data() first."
            )

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if len(self) == 0:
            return {"error": "Dataset empty"}

        graph = self[0]
        # Defensive: Only access attributes if present
        num_nodes = getattr(graph, "num_nodes", 0)
        num_edges = getattr(graph, "num_edges", 0)
        # Use getattr to access 'x' safely
        x = getattr(graph, "x", None)
        num_features = x.size(1) if x is not None and x.dim() > 1 else 0
        info = {
            "survey": self.survey,
            "graph_method": self.graph_method,
            "k_neighbors": self.k_neighbors,
            "use_3d_coordinates": self.use_3d_coordinates,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": num_features,
            "graph_type": getattr(graph, "graph_type", "unknown"),
        }

        return info
