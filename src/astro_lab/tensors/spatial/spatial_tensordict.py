"""
Enhanced Spatial TensorDict for AstroLab - PyG 2025 Optimized with Unified 3D Processing
===============================================================================================

Modern implementation leveraging:
- Direct integration with Enhanced3DPreprocessors
- Unified 3D coordinate handling from all surveys
- Native PyG 2.6.1 operations with torch.compile compatibility
- Automatic preprocessing data ingestion
- Cosmic web analysis optimized for multi-survey data
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import (
    FK5,
    ICRS,
    CartesianRepresentation,
    Galactic,
    Galactocentric,
    SkyCoord,
)
from astropy.time import Time
from tensordict import MemoryMappedTensor, TensorDict

# PyTorch Geometric imports - PyG 2.6.1 compatible
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import (
    coalesce,
    contains_self_loops,
    remove_self_loops,
    to_undirected,
)

from ..base import AstroTensorDict
from ..mixins import CoordinateConversionMixin, ValidationMixin
from .astronomical_mixin import AstronomicalMixin
from .open3d_mixin import Open3DMixin

logger = logging.getLogger(__name__)


class SpatialTensorDict(
    AstroTensorDict,
    CoordinateConversionMixin,
    ValidationMixin,
    AstronomicalMixin,
    Open3DMixin,
):
    """
    Enhanced Spatial tensor with unified 3D coordinate processing for all astronomical surveys.

    Features:
    - Direct integration with Enhanced3DPreprocessors
    - Automatic 3D coordinate extraction from processed dataframes
    - Multi-survey support with unified coordinate system
    - Native torch operations for graph construction
    - torch.compile compatible graph operations
    - Cosmic web clustering algorithms
    - AstroPy coordinate system integration (via AstronomicalMixin)
    - Open3D point cloud processing (via Open3DMixin)
    """

    # Coordinate frame mappings
    FRAME_MAPPING = {
        "icrs": ICRS,
        "galactic": Galactic,
        "galactocentric": Galactocentric,
        "fk5": FK5,
    }

    def __init__(
        self,
        coordinates: Union[
            torch.Tensor, np.ndarray, SkyCoord, MemoryMappedTensor, pl.DataFrame
        ],
        coordinate_system: str = "cartesian",
        unit: Union[str, u.Unit] = u.Unit("pc"),
        epoch: Union[float, Time] = Time("J2000"),
        survey_name: Optional[str] = None,
        object_type: Optional[str] = None,
        enable_compilation: bool = True,
        cache_edge_indices: bool = True,
        use_memory_mapping: bool = False,
        chunk_size: int = 1_000_000,
        preprocessor_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize with enhanced 3D coordinate processing.

        Args:
            coordinates: Spatial coordinates or processed dataframe
            coordinate_system: Coordinate frame ('cartesian', 'icrs', 'galactic')
            unit: Distance unit
            epoch: Time epoch
            survey_name: Name of the survey (e.g., 'gaia', 'nsa')
            object_type: Type of objects (e.g., 'star', 'galaxy')
            enable_compilation: Enable torch.compile
            cache_edge_indices: Cache edge_index objects
            use_memory_mapping: Use memory-mapped tensors for large data
            chunk_size: Chunk size for processing large datasets
            preprocessor_data: Additional data from preprocessor
        """

        # Set compilation and caching flags first
        self.enable_compilation = enable_compilation
        self.cache_edge_indices = cache_edge_indices
        self.use_memory_mapping = use_memory_mapping
        self.chunk_size = chunk_size
        self._edge_index_cache = {}
        self._graph_cache = {}
        self._compiled_functions = {}
        self._spatial_index = None

        # Survey metadata
        self.survey_name = survey_name or "unknown"
        self.object_type = object_type or "unknown"

        # Use simple string for epoch
        self.epoch = epoch if isinstance(epoch, str) else "J2000"

        # Process coordinates from various sources
        if isinstance(coordinates, pl.DataFrame):
            # Extract from processed dataframe (from Enhanced3DPreprocessors)
            coords_tensor, survey_data = self._extract_from_preprocessed_dataframe(
                coordinates
            )
            self.survey_name = survey_data.get("survey_name", self.survey_name)
            self.object_type = survey_data.get("object_type", self.object_type)

            # Store additional survey data
            self.survey_data = survey_data

        elif use_memory_mapping and not isinstance(coordinates, MemoryMappedTensor):
            # Convert to memory-mapped for large data
            coords_tensor = self._create_memory_mapped_coords(coordinates)
            self.survey_data = {}
        else:
            # Process various coordinate formats
            self.skycoord, coords_tensor = self._process_coordinates(
                coordinates, coordinate_system, unit, epoch
            )
            self.survey_data = {}

        # Core data structure with TensorDict
        data = TensorDict(
            {
                "coordinates": coords_tensor,
                "meta": {
                    "coordinate_system": coordinate_system,
                    "unit": str(unit),
                    "epoch": str(epoch),
                    "n_objects": coords_tensor.shape[0],
                    "survey_name": self.survey_name,
                    "object_type": self.object_type,
                    "pyg_version": "2.6+",
                    "compilation_enabled": enable_compilation,
                    "caching_enabled": cache_edge_indices,
                    "memory_mapped": bool(
                        isinstance(coords_tensor, MemoryMappedTensor)
                    ),
                },
            },
            batch_size=coords_tensor.shape[:-1],
        )

        # Add survey data to TensorDict if available
        if self.survey_data:
            n_objects = coords_tensor.shape[0]
            for key, value in self.survey_data.items():
                data[key] = self._fix_tensor_shape(value, n_objects)

        super().__init__(data, batch_size=coords_tensor.shape[:-1], **kwargs)

    def _fix_tensor_shape(self, value, n_objects):
        if value is None:
            return torch.full((n_objects,), float("nan"))
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return torch.full((n_objects,), float("nan"))
            elif value.ndim == 0:
                return value.expand(n_objects)
            elif value.ndim == 1 and value.shape[0] != n_objects:
                return value.expand(n_objects)
            else:
                return value.float()
        if isinstance(value, (int, float)):
            return torch.full((n_objects,), value, dtype=torch.float32)
        if isinstance(value, (list, np.ndarray)):
            arr = np.array(value)
            if arr.size == 0:
                return torch.full((n_objects,), float("nan"))
            t = torch.tensor(arr, dtype=torch.float32)
            if t.ndim == 0:
                return t.expand(n_objects)
            elif t.ndim == 1 and t.shape[0] != n_objects:
                return t.expand(n_objects)
            return t
        return value

    def _extract_from_preprocessed_dataframe(
        self, df: pl.DataFrame
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Extract unified 3D coordinates and metadata from processed dataframe."""
        logger.info(
            f"Extracting 3D coordinates from processed dataframe with {len(df)} objects"
        )

        # Check for unified 3D coordinates (from Enhanced3DPreprocessors)
        if not all(col in df.columns for col in ["x", "y", "z"]):
            raise ValueError(
                "Dataframe must have unified 3D coordinates (x, y, z) from Enhanced3DPreprocessor"
            )

        # Extract coordinates
        coords_data = df.select(["x", "y", "z"]).to_numpy()
        coords_tensor = torch.tensor(coords_data, dtype=torch.float32)

        # Extract metadata
        survey_data = {}

        # Survey identification
        if "survey_name" in df.columns:
            survey_data["survey_name"] = df["survey_name"][0]
        if "object_type" in df.columns:
            survey_data["object_type"] = df["object_type"][0]

        # Object identifiers
        if "object_id" in df.columns:
            survey_data["object_ids"] = df["object_id"].to_numpy()

        # Essential properties for visualization and analysis
        if "magnitude" in df.columns:
            survey_data["magnitude"] = df["magnitude"].to_numpy()
        if "brightness" in df.columns:
            survey_data["brightness"] = df["brightness"].to_numpy()
        if "mass" in df.columns:
            survey_data["mass"] = df["mass"].to_numpy()

        # Spherical coordinates (for backward compatibility)
        if "ra" in df.columns and "dec" in df.columns:
            survey_data["ra"] = df["ra"].to_numpy()
            survey_data["dec"] = df["dec"].to_numpy()
        if "distance_pc" in df.columns:
            survey_data["distance_pc"] = df["distance_pc"].to_numpy()

        # Survey-specific properties
        survey_specific_cols = [
            col
            for col in df.columns
            if col not in ["x", "y", "z", "survey_name", "object_type", "object_id"]
        ]

        for col in survey_specific_cols:
            try:
                # Only include numeric columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    survey_data[col] = df[col].to_numpy()
            except Exception:
                # Skip problematic columns
                continue

        logger.info(
            f"Extracted coordinates and {len(survey_data)} additional properties"
        )

        return coords_tensor, survey_data

    @classmethod
    def from_preprocessor(
        cls, preprocessor, processed_df: pl.DataFrame, **kwargs
    ) -> "SpatialTensorDict":
        """Create SpatialTensorDict directly from Enhanced3DPreprocessor output.

        Args:
            preprocessor: Enhanced3DPreprocessor instance
            processed_df: Processed dataframe with unified 3D coordinates
            **kwargs: Additional initialization parameters

        Returns:
            SpatialTensorDict with unified 3D coordinates and survey metadata
        """
        logger.info(
            f"Creating SpatialTensorDict from {preprocessor.get_survey_name()} preprocessor"
        )

        # Get preprocessor information
        preprocessor_info = preprocessor.get_info()

        # Create spatial tensor
        spatial_tensor = cls(
            coordinates=processed_df,
            coordinate_system="cartesian",  # Enhanced3DPreprocessors output Cartesian
            unit=u.Unit("pc"),  # Enhanced3DPreprocessors standardize to parsecs
            survey_name=preprocessor.get_survey_name(),
            object_type=preprocessor.get_object_type(),
            preprocessor_data=preprocessor_info,
            **kwargs,
        )

        # Add preprocessor metadata
        spatial_tensor.preprocessor_info = preprocessor_info

        logger.info(
            f"Created SpatialTensorDict: {spatial_tensor.n_objects} {preprocessor.get_object_type()}s from {preprocessor.get_survey_name()}"
        )

        return spatial_tensor

    @classmethod
    def from_multiple_surveys(
        cls,
        survey_data: Dict[str, Tuple[Any, pl.DataFrame]],
        coordinate_system: str = "cartesian",
        combine_method: str = "concatenate",
        **kwargs,
    ) -> "SpatialTensorDict":
        """Create unified SpatialTensorDict from multiple surveys.

        Args:
            survey_data: Dict mapping survey names to (preprocessor, dataframe) tuples
            coordinate_system: Target coordinate system
            combine_method: How to combine surveys ('concatenate', 'merge')
            **kwargs: Additional parameters

        Returns:
            Unified SpatialTensorDict with multi-survey data
        """
        logger.info(
            f"Combining {len(survey_data)} surveys into unified SpatialTensorDict"
        )

        all_coords = []
        all_survey_data = {}
        survey_offsets = {}
        current_offset = 0

        for survey_name, (preprocessor, df) in survey_data.items():
            logger.info(f"Processing {survey_name}: {len(df)} objects")

            # Extract coordinates
            if not all(col in df.columns for col in ["x", "y", "z"]):
                raise ValueError(f"Survey {survey_name} missing unified 3D coordinates")

            coords = df.select(["x", "y", "z"]).to_numpy()
            all_coords.append(coords)

            # Track survey boundaries
            survey_offsets[survey_name] = (current_offset, current_offset + len(coords))
            current_offset += len(coords)

            # Store survey-specific data with prefixes
            for col in df.columns:
                if col not in ["x", "y", "z"]:
                    key = (
                        f"{survey_name}_{col}"
                        if combine_method == "concatenate"
                        else col
                    )
                    try:
                        if df[col].dtype in [
                            pl.Float32,
                            pl.Float64,
                            pl.Int32,
                            pl.Int64,
                        ]:
                            if key not in all_survey_data:
                                all_survey_data[key] = []
                            all_survey_data[key].append(df[col].to_numpy())
                    except Exception:
                        continue

        # Combine coordinates
        combined_coords = np.vstack(all_coords)

        # Combine survey data
        combined_survey_data = {}
        for key, arrays in all_survey_data.items():
            if len(arrays) == len(survey_data):
                # Only include if all surveys have this field
                combined_survey_data[key] = np.concatenate(arrays)

        # Add survey labels
        survey_labels = []
        object_type_labels = []
        for survey_name, (preprocessor, df) in survey_data.items():
            n_objects = len(df)
            survey_labels.extend([survey_name] * n_objects)
            object_type_labels.extend([preprocessor.get_object_type()] * n_objects)

        combined_survey_data["survey_labels"] = survey_labels
        combined_survey_data["object_type_labels"] = object_type_labels

        # Create unified tensor
        unified_tensor = cls(
            coordinates=torch.tensor(combined_coords, dtype=torch.float32),
            coordinate_system=coordinate_system,
            survey_name="multi_survey",
            object_type="mixed",
            **kwargs,
        )

        # Add combined survey data
        for key, value in combined_survey_data.items():
            if isinstance(value, np.ndarray):
                unified_tensor[key] = torch.tensor(value, dtype=torch.float32)

        # Store survey metadata
        unified_tensor.survey_offsets = survey_offsets
        unified_tensor.survey_list = list(survey_data.keys())

        logger.info(
            f"Created unified SpatialTensorDict: {unified_tensor.n_objects} objects from {len(survey_data)} surveys"
        )

        return unified_tensor

    def _process_coordinates(
        self,
        coordinates: Union[torch.Tensor, np.ndarray, SkyCoord],
        coordinate_system: str,
        unit: Union[str, u.Unit],
        epoch: Union[float, Time],
    ) -> Tuple[Optional[SkyCoord], torch.Tensor]:
        """Process coordinates into SkyCoord and tensor formats."""
        # Store original device if coordinates is a tensor
        original_device = None
        if isinstance(coordinates, torch.Tensor):
            original_device = coordinates.device
            coords_np = coordinates.detach().cpu().numpy()
        elif isinstance(coordinates, np.ndarray):
            coords_np = coordinates
        elif isinstance(coordinates, SkyCoord):
            # If already SkyCoord, convert to tensor directly
            coords_tensor = self._skycoord_to_tensor(coordinates)
            return coordinates, coords_tensor
        else:
            coords_np = np.array(coordinates)

        # Convert to tensor
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32)
        if original_device is not None:
            coords_tensor = coords_tensor.to(original_device)

        # Create a simplified SkyCoord object for basic functionality
        skycoord = None
        try:
            if coords_np.shape[1] == 3:
                # Cartesian coordinates
                skycoord = SkyCoord(
                    x=coords_np[:, 0] * u.Unit("pc"),
                    y=coords_np[:, 1] * u.Unit("pc"),
                    z=coords_np[:, 2] * u.Unit("pc"),
                    representation_type="cartesian",
                    frame=coordinate_system,
                )
        except Exception:
            # If SkyCoord creation fails, continue without it
            logger.warning(
                "Could not create SkyCoord object, continuing without astronomical features"
            )
            skycoord = None

        return skycoord, coords_tensor

    def _create_memory_mapped_coords(
        self, coordinates: Union[torch.Tensor, np.ndarray]
    ) -> MemoryMappedTensor:
        """Create memory-mapped tensor for large coordinate data."""
        if isinstance(coordinates, torch.Tensor):
            shape = coordinates.shape
            dtype = coordinates.dtype
        else:
            shape = coordinates.shape
            dtype = torch.float32

        # Create memory-mapped tensor
        mmap_tensor = MemoryMappedTensor.empty(
            shape, dtype=dtype, filename=f"coords_{shape[0]}.memmap"
        )

        # Fill in chunks to avoid memory overflow
        if isinstance(coordinates, torch.Tensor):
            coords_np = coordinates.detach().cpu().numpy()
        else:
            coords_np = coordinates

        n_chunks = max(1, shape[0] // self.chunk_size)
        for i in range(n_chunks + 1):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, shape[0])
            if start_idx < shape[0]:
                mmap_tensor[start_idx:end_idx] = torch.from_numpy(
                    coords_np[start_idx:end_idx]
                )

        # Set skycoord to None for memory-mapped case
        self.skycoord = None

        return mmap_tensor

    def _skycoord_to_tensor(self, skycoord: SkyCoord) -> torch.Tensor:
        """Convert SkyCoord to tensor representation."""
        if hasattr(skycoord, "cartesian"):
            cart = skycoord.cartesian
        else:
            cart = skycoord.represent_as(CartesianRepresentation)

        coords = np.stack(
            [
                cart.x.to_value("pc"),
                cart.y.to_value("pc"),
                cart.z.to_value("pc"),
            ],
            axis=-1,
        )

        return torch.tensor(coords, dtype=torch.float32)

    @property
    def coordinates(self) -> torch.Tensor:
        """Coordinates tensor in parsecs."""
        return self["coordinates"]

    @property
    def x(self) -> torch.Tensor:
        """X coordinate in parsecs."""
        return self.coordinates[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Y coordinate in parsecs."""
        return self.coordinates[..., 1]

    @property
    def z(self) -> torch.Tensor:
        """Z coordinate in parsecs."""
        return self.coordinates[..., 2]

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return self.coordinates.size(0)

    def build_edge_index(
        self,
        method: str = "knn",
        k: int = 10,
        r: Optional[float] = None,
        max_num_neighbors: int = 64,
        force_undirected: bool = True,
        use_spatial_partitioning: bool = False,
        partition_size: int = 100_000,
        **kwargs,
    ) -> torch.Tensor:
        """
        Build edge_index with optimizations for large-scale data.

        Args:
            method: 'knn' or 'radius'
            k: Number of neighbors for kNN
            r: Radius for radius graph (parsecs)
            max_num_neighbors: Maximum neighbors to prevent memory issues
            force_undirected: Ensure undirected graph
            use_spatial_partitioning: Use spatial partitioning for 50M+ objects
            partition_size: Size of spatial partitions

        Returns:
            Edge index tensor [2, num_edges]
        """
        # Check cache first
        cache_key = f"{method}_{k}_{r}_{max_num_neighbors}_{force_undirected}"
        if self.cache_edge_indices and cache_key in self._edge_index_cache:
            return self._edge_index_cache[cache_key]

        coords = self.coordinates
        n_objects = coords.size(0)

        # For very large datasets, use spatial partitioning
        if use_spatial_partitioning or n_objects > 10_000_000:
            edge_index = self._build_partitioned_edge_index(
                coords, method, k, r, max_num_neighbors, partition_size
            )
        else:
            # Standard edge index building
            if method == "knn":
                edge_index = knn_graph(
                    coords,
                    k=k,
                    loop=False,
                    cosine=kwargs.get("cosine", False),
                    num_workers=kwargs.get("num_workers", 4),
                )
            elif method == "radius":
                if r is None:
                    raise ValueError("Radius r must be specified for radius graph")
                edge_index = radius_graph(
                    coords,
                    r=r,
                    loop=False,
                    max_num_neighbors=max_num_neighbors,
                    num_workers=kwargs.get("num_workers", 4),
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        # Make undirected if requested
        if force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=n_objects)

        # Remove self-loops and duplicates
        edge_index = coalesce(edge_index, num_nodes=n_objects)
        if contains_self_loops(edge_index):
            edge_index, _ = remove_self_loops(edge_index)

        # Cache result
        if self.cache_edge_indices:
            self._edge_index_cache[cache_key] = edge_index

        logger.debug(
            f"Built edge_index: {edge_index.size(1)} edges for {n_objects} nodes"
        )

        return edge_index

    def _build_partitioned_edge_index(
        self,
        coords: torch.Tensor,
        method: str,
        k: int,
        r: Optional[float],
        max_num_neighbors: int,
        partition_size: int,
    ) -> torch.Tensor:
        """Build edge index using spatial partitioning for very large datasets."""
        n_objects = coords.size(0)

        # Create spatial partitions using simple grid approach
        partitions = self._create_spatial_partitions(coords, partition_size)
        edge_indices = []

        for partition_idx, partition in enumerate(partitions):
            # Get coordinates for this partition
            partition_coords = coords[partition]

            # Build local edges
            if method == "knn":
                local_edges = knn_graph(
                    partition_coords,
                    k=min(k, len(partition)),
                    loop=False,
                )
            else:  # radius
                local_edges = radius_graph(
                    partition_coords,
                    r=r,
                    loop=False,
                    max_num_neighbors=max_num_neighbors,
                )

            # Map back to global indices
            global_edges = partition[local_edges]
            edge_indices.append(global_edges)

        # Combine all edges
        if edge_indices:
            combined_edges = torch.cat(edge_indices, dim=1)
            return combined_edges
        else:
            return torch.empty((2, 0), dtype=torch.long)

    def _create_spatial_partitions(
        self, coords: torch.Tensor, partition_size: int
    ) -> List[torch.Tensor]:
        """Create spatial partitions using grid-based subdivision."""
        n_objects = coords.size(0)

        if n_objects <= partition_size:
            return [torch.arange(n_objects)]

        # Simple grid-based partitioning
        # Get bounding box
        min_coords = coords.min(dim=0)[0]
        max_coords = coords.max(dim=0)[0]

        # Calculate grid size
        n_partitions = (n_objects // partition_size) + 1
        grid_size = int(np.ceil(n_partitions ** (1 / 3)))

        partitions = []
        cell_size = (max_coords - min_coords) / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    # Define cell bounds
                    cell_min = (
                        min_coords
                        + torch.tensor([i, j, k], dtype=torch.float32) * cell_size
                    )
                    cell_max = cell_min + cell_size

                    # Find points in this cell
                    mask = ((coords >= cell_min) & (coords < cell_max)).all(dim=1)
                    indices = torch.where(mask)[0]

                    if len(indices) > 0:
                        partitions.append(indices)

        return partitions

    def build_pyg_data(
        self,
        method: str = "knn",
        k: int = 10,
        r: Optional[float] = None,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        include_survey_features: bool = True,
        **kwargs,
    ) -> Data:
        """
        Build optimized PyG Data object with survey-specific features.

        Args:
            method: Graph construction method
            k: Number of neighbors
            r: Radius in parsecs
            node_features: Node feature matrix [N, F]
            edge_features: Edge feature matrix [E, F_edge]
            target: Target labels/values
            include_survey_features: Include survey-specific features automatically

        Returns:
            PyG Data object optimized for PyG 2025
        """
        # Build edge index
        edge_index = self.build_edge_index(method=method, k=k, r=r, **kwargs)

        # Prepare node features
        if node_features is None:
            feature_list = [self.coordinates]

            # Add survey-specific features if available
            if include_survey_features:
                if "magnitude" in self:
                    feature_list.append(self["magnitude"].unsqueeze(-1))
                if "brightness" in self:
                    feature_list.append(self["brightness"].unsqueeze(-1))
                if "mass" in self:
                    feature_list.append(self["mass"].unsqueeze(-1))
                if "ra" in self and "dec" in self:
                    feature_list.extend(
                        [self["ra"].unsqueeze(-1), self["dec"].unsqueeze(-1)]
                    )

            node_features = torch.cat(feature_list, dim=-1)

        # Calculate edge features if not provided
        if edge_features is None:
            edge_features = self._calculate_edge_features(edge_index)

        # Create Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            pos=self.coordinates,  # Position for visualization
            y=target,
        )

        # Add survey metadata
        data.meta = {
            "num_nodes": self.coordinates.size(0),
            "num_edges": edge_index.size(1),
            "graph_method": method,
            "coordinate_system": self["meta"]["coordinate_system"],
            "unit": self["meta"]["unit"],
            "survey_name": self.survey_name,
            "object_type": self.object_type,
        }

        # Add survey-specific attributes
        if hasattr(self, "survey_data") and self.survey_data:
            for key, value in self.survey_data.items():
                if isinstance(value, torch.Tensor) and value.size(0) == self.n_objects:
                    setattr(data, key, value)

        return data

    def _calculate_edge_features(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Calculate edge features (distances, directions, etc.)."""
        coords = self.coordinates
        row, col = edge_index

        # Edge vectors and distances
        edge_vectors = coords[row] - coords[col]
        edge_distances = torch.norm(edge_vectors, dim=1, keepdim=True)

        # Unit direction vectors
        edge_directions = edge_vectors / (edge_distances + 1e-8)

        # Combine into edge features
        edge_features = torch.cat([edge_distances, edge_directions], dim=1)

        return edge_features

    def cosmic_web_clustering(
        self,
        method: str = "fof",
        linking_length: float = 10.0,
        min_group_size: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        """
        Cosmic web clustering optimized for large datasets.

        Args:
            method: 'fof' (Friends of Friends) or 'hierarchical'
            linking_length: Linking length in parsecs
            min_group_size: Minimum cluster size

        Returns:
            Cluster labels tensor
        """
        coords = self.coordinates

        if method == "fof":
            labels = self._friends_of_friends_clustering(
                coords, linking_length, min_group_size
            )
        elif method == "hierarchical":
            labels = self._hierarchical_clustering(
                coords, linking_length, min_group_size
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return labels.to(coords.device)

    def _friends_of_friends_clustering(
        self,
        coords: torch.Tensor,
        linking_length: float,
        min_group_size: int,
    ) -> torch.Tensor:
        """Optimized Friends-of-Friends clustering."""
        # Build radius graph
        edge_index = radius_graph(
            coords, r=linking_length, loop=False, max_num_neighbors=256
        )

        num_nodes = coords.size(0)
        device = coords.device

        # Connected components using simple union-find
        labels = self._connected_components(edge_index, num_nodes, device)

        # Filter small groups
        filtered_labels = self._filter_small_clusters(labels, min_group_size)
        return filtered_labels

    def _connected_components(
        self, edge_index: torch.Tensor, num_nodes: int, device=None
    ) -> torch.Tensor:
        """Pure PyTorch connected components implementation."""
        if device is None:
            device = edge_index.device

        labels = torch.arange(num_nodes, device=device, dtype=torch.long)

        # Simple union-find
        row, col = edge_index
        for i in range(edge_index.size(1)):
            u, v = row[i].item(), col[i].item()

            # Find roots
            root_u = u
            while labels[root_u] != root_u:
                root_u = labels[root_u].item()

            root_v = v
            while labels[root_v] != root_v:
                root_v = labels[root_v].item()

            # Union
            if root_u != root_v:
                labels[root_v] = root_u

        # Path compression pass
        for i in range(num_nodes):
            node = i
            while labels[node] != node:
                parent = labels[node].item()
                labels[node] = labels[parent]
                node = parent

        return labels

    def _filter_small_clusters(
        self, labels: torch.Tensor, min_size: int
    ) -> torch.Tensor:
        """Filter out clusters smaller than min_size."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        valid_mask = counts >= min_size
        valid_labels = unique_labels[valid_mask]

        # Create final labels with noise = -1
        final_labels = torch.full_like(labels, -1, device=labels.device)
        for i, valid_label in enumerate(valid_labels):
            mask = labels == valid_label
            final_labels[mask] = i

        return final_labels

    def _hierarchical_clustering(
        self, coords: torch.Tensor, threshold: float, min_group_size: int
    ) -> torch.Tensor:
        """Hierarchical clustering using PyTorch operations."""
        n_points = coords.size(0)

        # Build initial kNN graph for efficiency
        k = min(64, n_points - 1)
        edge_index = knn_graph(coords, k=k, loop=False)

        # Compute edge weights (distances)
        row, col = edge_index
        edge_weights = torch.norm(coords[row] - coords[col], dim=1)

        # Filter edges by threshold
        mask = edge_weights <= threshold
        edge_index = edge_index[:, mask]

        # Find connected components
        labels = self._connected_components(edge_index, n_points)

        # Filter small groups
        labels = self._filter_small_clusters(labels, min_group_size)

        return labels

    def get_survey_statistics(self) -> Dict[str, Any]:
        """Get comprehensive survey-specific statistics."""
        stats = {
            "survey_name": self.survey_name,
            "object_type": self.object_type,
            "n_objects": self.n_objects,
            "coordinate_system": self["meta"]["coordinate_system"],
        }

        # Coordinate bounds
        coords = self.coordinates
        bounds = {
            "x_min": coords[:, 0].min().item(),
            "x_max": coords[:, 0].max().item(),
            "y_min": coords[:, 1].min().item(),
            "y_max": coords[:, 1].max().item(),
            "z_min": coords[:, 2].min().item(),
            "z_max": coords[:, 2].max().item(),
        }

        # Volume
        volume = (
            (bounds["x_max"] - bounds["x_min"])
            * (bounds["y_max"] - bounds["y_min"])
            * (bounds["z_max"] - bounds["z_min"])
        )

        stats.update(
            {
                "coordinate_bounds": bounds,
                "volume_pc3": volume,
                "number_density": self.n_objects / volume if volume > 0 else 0,
            }
        )

        # Survey-specific properties
        if hasattr(self, "survey_data") and self.survey_data:
            survey_stats = {}
            for key, value in self.survey_data.items():
                if isinstance(value, torch.Tensor) and value.numel() > 0:
                    if value.dtype.is_floating_point:
                        survey_stats[key] = {
                            "mean": value.mean().item(),
                            "std": value.std().item(),
                            "min": value.min().item(),
                            "max": value.max().item(),
                        }
                    elif value.dtype in [torch.int32, torch.int64]:
                        survey_stats[key] = {
                            "min": value.min().item(),
                            "max": value.max().item(),
                            "unique_values": len(torch.unique(value)),
                        }
            stats["survey_properties"] = survey_stats

        return stats

    def extract_features(
        self, feature_types: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Extract spatial and survey-specific features from the TensorDict.

        Args:
            feature_types: Types of features to extract ('spatial', 'kinematic', 'photometric')
            **kwargs: Additional extraction parameters

        Returns:
            Dictionary of extracted features
        """
        # Get base features
        features = super().extract_features(feature_types, **kwargs)

        # Add spatial-specific computed features if requested
        if feature_types is None or "spatial" in feature_types:
            # Add computed spatial features
            coords = self.coordinates

            # Basic spatial properties
            features["distance_from_origin"] = torch.norm(coords, dim=-1)
            features["galactic_height"] = torch.abs(coords[:, 2])  # |z| coordinate

            # Spherical coordinates from survey data
            if "ra" in self and "dec" in self:
                features["ra"] = self["ra"]
                features["dec"] = self["dec"]
                features["abs_galactic_latitude"] = torch.abs(self["dec"])
            if "distance_pc" in self:
                features["spherical_distance"] = self["distance_pc"]

        if feature_types is None or "kinematic" in feature_types:
            # Add kinematic features if available from survey
            if "pmra" in self and "pmdec" in self:
                features["total_proper_motion"] = torch.sqrt(
                    self["pmra"] ** 2 + self["pmdec"] ** 2
                )

            if "radial_velocity" in self:
                features["radial_velocity"] = self["radial_velocity"]

        if feature_types is None or "photometric" in feature_types:
            # Add photometric features from survey
            if "magnitude" in self:
                features["magnitude"] = self["magnitude"]
            if "brightness" in self:
                features["brightness"] = self["brightness"]

        # Add survey-specific features
        if hasattr(self, "survey_data") and self.survey_data:
            for key, value in self.survey_data.items():
                if isinstance(value, torch.Tensor) and value.size(0) == self.n_objects:
                    features[f"survey_{key}"] = value

        return features

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for visualization with survey-specific properties."""
        viz_data = {
            "positions": self.coordinates,
            "survey_name": self.survey_name,
            "object_type": self.object_type,
            "n_objects": self.n_objects,
        }

        # Add brightness/magnitude for visualization
        if "brightness" in self:
            viz_data["brightness"] = self["brightness"]
        elif "magnitude" in self:
            # Convert magnitude to brightness (inverted scale)
            viz_data["brightness"] = 25.0 - self["magnitude"]

        # Add mass for size visualization
        if "mass" in self:
            viz_data["mass"] = self["mass"]

        # Add object IDs
        if "object_ids" in self.survey_data:
            viz_data["object_ids"] = self.survey_data["object_ids"]

        return viz_data

    def validate(self) -> bool:
        """Validate spatial tensor data with survey-specific checks."""
        # Parent validation
        if not super().validate():
            return False

        # Spatial-specific validation
        coords = self.coordinates

        # Check coordinate dimensionality
        if coords.ndim != 2 or coords.size(-1) != 3:
            logger.error(f"Invalid coordinate shape: {coords.shape}")
            return False

        # Check for NaN/inf values
        if torch.isnan(coords).any() or torch.isinf(coords).any():
            logger.error("Coordinates contain NaN or infinity values")
            return False

        # Check coordinate system
        if "coordinate_system" not in self["meta"]:
            logger.error("Missing coordinate system metadata")
            return False

        # Validate survey-specific data consistency
        if hasattr(self, "survey_data") and self.survey_data:
            for key, value in self.survey_data.items():
                if isinstance(value, torch.Tensor) and value.size(0) != self.n_objects:
                    logger.error(
                        f"Survey data '{key}' has inconsistent size: {value.size(0)} vs {self.n_objects}"
                    )
                    return False

        return True

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._edge_index_cache.clear()
        self._graph_cache.clear()
        self._compiled_functions.clear()
        logger.info("Cleared all caches")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SpatialTensorDict(n_objects={self.n_objects}, "
            f"survey='{self.survey_name}', object_type='{self.object_type}', "
            f"coordinate_system='{self['meta']['coordinate_system']}', "
            f"unit='{self['meta']['unit']}', device={self.device})"
        )
