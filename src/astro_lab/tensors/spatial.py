"""
Spatial TensorDict for AstroLab - PyG 2025 Optimized
============================================================

implementation leveraging:
- EdgeIndex with metadata caching and optimizations
- torch.compile compatible operations
- Native pyg-lib when available
- cosmic web analysis
- AstroPy coordinate transformations
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_cluster
from tensordict import MemoryMappedTensor, TensorDict
from torch import Tensor

# PyTorch Geometric
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.nn import fps, knn_graph, radius_graph
from torch_geometric.transforms import Compose
from torch_geometric.utils import (
    coalesce,
    contains_self_loops,
    degree,
    k_hop_subgraph,
    remove_self_loops,
    subgraph,
    to_undirected,
)

# pyg-lib for advanced operations
try:
    import pyg_lib

    HAS_PYG_LIB = True
except ImportError:
    HAS_PYG_LIB = False

# AstroPy for coordinate handling
import astropy.units as u
from astropy.coordinates import (
    FK5,
    ICRS,
    CartesianRepresentation,
    Galactic,
    Galactocentric,
    SkyCoord,
)
from astropy.time import Time

from .base import AstroTensorDict
from .mixins import CoordinateConversionMixin, ValidationMixin

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import trimesh
    from scipy.spatial import cKDTree as _cKDTree

    TrimeshType = trimesh.Trimesh
    KDTreeType = _cKDTree
else:
    TrimeshType = Any
    KDTreeType = Any


class SpatialTensorDict(AstroTensorDict, CoordinateConversionMixin, ValidationMixin):
    """
    spatial tensor with PyG 2025 optimizations.

    Features:
    - EdgeIndex with automatic caching and metadata
    - torch.compile compatible graph operations
    - Native pyg-lib operations when available
    - cosmic web clustering algorithms
    - AstroPy coordinate system integration
    - Distributed processing support
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
        coordinates: Union[torch.Tensor, np.ndarray, SkyCoord, MemoryMappedTensor],
        coordinate_system: str = "icrs",
        unit: Union[str, u.Unit] = u.Unit("pc"),
        epoch: Union[float, Time] = Time("J2000"),
        enable_compilation: bool = True,
        cache_edge_indices: bool = True,
        use_memory_mapping: bool = False,
        chunk_size: int = 1_000_000,
        **kwargs,
    ):
        """
        Initialize with modern PyG 2025 optimizations for large-scale data.

        Args:
            coordinates: Spatial coordinates (supports MemoryMappedTensor for 50M+ objects)
            coordinate_system: Coordinate frame
            unit: Distance unit
            epoch: Time epoch
            enable_compilation: Enable torch.compile
            cache_edge_indices: Cache EdgeIndex objects
            use_memory_mapping: Use memory-mapped tensors for large data
            chunk_size: Chunk size for processing large datasets
        """

        # Set compilation and caching flags first
        self.enable_compilation = enable_compilation
        self.cache_edge_indices = cache_edge_indices
        self.use_memory_mapping = use_memory_mapping
        self.chunk_size = chunk_size
        self._edge_index_cache = {}
        self._graph_cache = {}
        self._compiled_functions = {}
        self._spatial_index = None  # For efficient queries

        # Use simple string for epoch instead of Time object
        self.epoch = epoch if isinstance(epoch, str) else "J2000"

        # Process coordinates to tensor format
        if use_memory_mapping and not isinstance(coordinates, MemoryMappedTensor):
            # Convert to memory-mapped for large data
            coords_tensor = self._create_memory_mapped_coords(coordinates)
        else:
            self.skycoord, coords_tensor = self._process_coordinates(
                coordinates, coordinate_system, unit, epoch
            )

        # Core data structure with TensorDict
        if isinstance(coords_tensor, MemoryMappedTensor):
            # Use TensorDict with memory-mapped tensors
            data = TensorDict(
                {
                    "coordinates": coords_tensor,
                    "meta": {
                        "coordinate_system": coordinate_system,
                        "unit": str(unit),
                        "epoch": str(epoch),
                        "n_objects": coords_tensor.shape[0],
                        "pyg_version": "2.6+",
                        "has_pyg_lib": HAS_PYG_LIB,
                        "compilation_enabled": enable_compilation,
                        "caching_enabled": cache_edge_indices,
                        "memory_mapped": True,
                    },
                },
                batch_size=coords_tensor.shape[:-1],
            )
        else:
            data = {
                "coordinates": coords_tensor,
                "meta": {
                    "coordinate_system": coordinate_system,
                    "unit": str(unit),
                    "epoch": str(epoch),
                    "n_objects": len(coords_tensor),
                    "pyg_version": "2.6+",
                    "has_pyg_lib": HAS_PYG_LIB,
                    "compilation_enabled": enable_compilation,
                    "caching_enabled": cache_edge_indices,
                    "memory_mapped": False,
                },
            }

        super().__init__(data, batch_size=coords_tensor.shape[:-1], **kwargs)

    def _process_coordinates(
        self,
        coordinates: Union[torch.Tensor, np.ndarray, SkyCoord],
        coordinate_system: str,
        unit: Union[str, u.Unit],
        epoch: Union[float, Time],
    ) -> Tuple[SkyCoord, torch.Tensor]:
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

        # Assume coordinates are already in parsecs
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32)
        if original_device is not None:
            coords_tensor = coords_tensor.to(original_device)

        # Create a real SkyCoord object with proper frame-specific parameters
        if coords_np.shape[1] == 3:
            # Cartesian coordinates
            skycoord = SkyCoord(
                x=coords_np[:, 0] * u.Unit("pc"),
                y=coords_np[:, 1] * u.Unit("pc"),
                z=coords_np[:, 2] * u.Unit("pc"),
                representation_type="cartesian",
                frame=coordinate_system,
                obstime=None,
            )
        elif coords_np.shape[1] == 2:
            # Spherical coordinates - use frame-specific parameter names
            if coordinate_system.lower() in ["icrs", "fk5", "fk4"]:
                # Use ra/dec for equatorial frames
                skycoord = SkyCoord(
                    ra=coords_np[:, 0] * u.Unit("deg"),
                    dec=coords_np[:, 1] * u.Unit("deg"),
                    distance=np.ones(coords_np.shape[0]) * u.Unit("pc"),
                    frame=coordinate_system,
                    obstime=None,
                )
            elif coordinate_system.lower() in ["galactic", "galactocentric"]:
                # Use l/b for galactic frames
                skycoord = SkyCoord(
                    l=coords_np[:, 0] * u.Unit("deg"),
                    b=coords_np[:, 1] * u.Unit("deg"),
                    distance=np.ones(coords_np.shape[0]) * u.Unit("pc"),
                    frame=coordinate_system,
                    obstime=None,
                )
            else:
                # Generic spherical coordinates
                skycoord = SkyCoord(
                    lon=coords_np[:, 0] * u.Unit("deg"),
                    lat=coords_np[:, 1] * u.Unit("deg"),
                    distance=np.ones(coords_np.shape[0]) * u.Unit("pc"),
                    frame=coordinate_system,
                    obstime=None,
                )
        else:
            raise ValueError("Coordinates must be [N,2] or [N,3] shape.")
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

        # Create dummy SkyCoord for metadata (using subset)
        sample_coords = coords_np[: min(1000, shape[0])]
        self.skycoord = SkyCoord(
            x=sample_coords[:, 0] * u.Unit("pc"),
            y=sample_coords[:, 1] * u.Unit("pc"),
            z=sample_coords[:, 2] * u.Unit("pc"),
            representation_type="cartesian",
            frame="icrs",
        )

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
    ) -> EdgeIndex:
        """
        Build EdgeIndex with modern PyG 2025 optimizations for large-scale data.

        Args:
            method: 'knn' or 'radius'
            k: Number of neighbors for kNN
            r: Radius for radius graph (parsecs)
            max_num_neighbors: Maximum neighbors to prevent memory issues
            force_undirected: Ensure undirected graph
            use_spatial_partitioning: Use spatial partitioning for 50M+ objects
            partition_size: Size of spatial partitions

        Returns:
            EdgeIndex with metadata and caching
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

        # Create EdgeIndex object with metadata - PyG 2025 API
        edge_index_obj = EdgeIndex(
            edge_index,
            sparse_size=(n_objects, n_objects),
            sort_order="row",
            is_undirected=force_undirected,
        )

        # No manual fill_cache_() for torch.compile compatibility

        # Cache result
        if self.cache_edge_indices:
            self._edge_index_cache[cache_key] = edge_index_obj

        logger.debug(
            f"Built EdgeIndex: {edge_index_obj.size(1)} edges for {n_objects} nodes"
        )

        return edge_index_obj

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

        # Create spatial partitions using octree-like structure
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

            # Also check neighboring partitions for boundary connections
            neighbor_partitions = self._get_neighbor_partitions(
                partition_idx, partitions
            )
            for neighbor_idx in neighbor_partitions:
                neighbor_coords = coords[partitions[neighbor_idx]]

                # Build cross-partition edges
                if method == "knn":
                    # Use radius for cross-partition to limit connections
                    cross_edges = radius_graph(
                        torch.cat([partition_coords, neighbor_coords]),
                        r=r if r is not None else k * 2.0,  # Heuristic radius
                        loop=False,
                        max_num_neighbors=max_num_neighbors,
                    )
                else:
                    cross_edges = radius_graph(
                        torch.cat([partition_coords, neighbor_coords]),
                        r=r,
                        loop=False,
                        max_num_neighbors=max_num_neighbors,
                    )

                # Filter to only keep cross-partition edges
                mask = (cross_edges[0] < len(partition)) & (
                    cross_edges[1] >= len(partition)
                )
                if mask.any():
                    cross_edges = cross_edges[:, mask]
                    # Map to global indices
                    cross_edges[0] = partition[cross_edges[0]]
                    cross_edges[1] = partitions[neighbor_idx][
                        cross_edges[1] - len(partition)
                    ]
                    edge_indices.append(cross_edges)

        # Combine all edges
        if edge_indices:
            combined_edges = torch.cat(edge_indices, dim=1)
            return combined_edges
        else:
            return torch.empty((2, 0), dtype=torch.long)

    def _create_spatial_partitions(
        self, coords: torch.Tensor, partition_size: int
    ) -> List[torch.Tensor]:
        """Create spatial partitions using octree-like subdivision."""
        n_objects = coords.size(0)

        if n_objects <= partition_size:
            return [torch.arange(n_objects)]

        # Get bounding box
        min_coords = coords.min(dim=0)[0]
        max_coords = coords.max(dim=0)[0]
        center = (min_coords + max_coords) / 2

        # Recursive octree partitioning
        partitions = []
        self._octree_partition(
            coords, torch.arange(n_objects), center, partitions, partition_size
        )

        return partitions

    def _octree_partition(
        self,
        coords: torch.Tensor,
        indices: torch.Tensor,
        center: torch.Tensor,
        partitions: List[torch.Tensor],
        partition_size: int,
    ):
        """Recursive octree partitioning."""
        if len(indices) <= partition_size:
            partitions.append(indices)
            return

        # Split based on center
        subset_coords = coords[indices]

        # Create 8 octants
        octants = []
        for i in range(8):
            mask = torch.ones(len(indices), dtype=torch.bool)
            if i & 1:  # x > center
                mask &= subset_coords[:, 0] > center[0]
            else:
                mask &= subset_coords[:, 0] <= center[0]
            if i & 2:  # y > center
                mask &= subset_coords[:, 1] > center[1]
            else:
                mask &= subset_coords[:, 1] <= center[1]
            if i & 4:  # z > center
                mask &= subset_coords[:, 2] > center[2]
            else:
                mask &= subset_coords[:, 2] <= center[2]

            octant_indices = indices[mask]
            if len(octant_indices) > 0:
                octants.append(octant_indices)

        # Recursively partition octants
        for octant_indices in octants:
            if len(octant_indices) > partition_size:
                octant_coords = coords[octant_indices]
                octant_center = octant_coords.mean(dim=0)
                self._octree_partition(
                    coords, octant_indices, octant_center, partitions, partition_size
                )
            else:
                partitions.append(octant_indices)

    def _get_neighbor_partitions(
        self, partition_idx: int, partitions: List[torch.Tensor]
    ) -> List[int]:
        """Get neighboring partitions for boundary connections."""
        # Simple heuristic: return adjacent partition indices
        neighbors = []
        if partition_idx > 0:
            neighbors.append(partition_idx - 1)
        if partition_idx < len(partitions) - 1:
            neighbors.append(partition_idx + 1)
        return neighbors

    def build_pyg_data(
        self,
        method: str = "knn",
        k: int = 10,
        r: Optional[float] = None,
        node_features: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Data:
        """
        Build optimized PyG Data object.

        Args:
            method: Graph construction method
            k: Number of neighbors
            r: Radius in parsecs
            node_features: Node feature matrix [N, F]
            edge_features: Edge feature matrix [E, F_edge]
            target: Target labels/values

        Returns:
            PyG Data object optimized for PyG 2025
        """

        # Build edge index
        edge_index = self.build_edge_index(method=method, k=k, r=r, **kwargs)

        # Prepare node features
        if node_features is None:
            node_features = self.coordinates

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

        # Add metadata
        data.meta = {
            "num_nodes": self.coordinates.size(0),
            "num_edges": edge_index.num_edges,
            "graph_method": method,
            "coordinate_system": self["meta"]["coordinate_system"],
            "unit": self["meta"]["unit"],
        }

        return data

    def _calculate_edge_features(self, edge_index: EdgeIndex) -> torch.Tensor:
        """Calculate edge features (distances, directions, etc.)."""

        coords = self.coordinates
        row, col = edge_index.tensor

        # Edge vectors and distances
        edge_vectors = coords[row] - coords[col]
        edge_distances = torch.norm(edge_vectors, dim=1, keepdim=True)

        # Unit direction vectors
        edge_directions = edge_vectors / (edge_distances + 1e-8)

        # Combine into edge features
        edge_features = torch.cat([edge_distances, edge_directions], dim=1)

        return edge_features

    def _fast_clustering_kernel_implementation(
        self, edge_index: Tensor, num_nodes: int, device=None
    ) -> Tensor:
        """Fast connected components using PyTorch geometric utils."""
        if device is None:
            device = edge_index.device
        # Use pure PyTorch Union-Find implementation
        return self._torch_union_find(edge_index, num_nodes, device=device)

    def _torch_union_find(
        self, edge_index: Tensor, num_nodes: int, device=None
    ) -> Tensor:
        """Pure PyTorch Union-Find implementation."""
        if device is None:
            device = edge_index.device
        labels = torch.arange(num_nodes, device=device, dtype=torch.long)

        # union-find without path compression for torch.compile compatibility
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

    def cosmic_web_clustering(
        self,
        method: str = "fof",
        linking_length: float = 10.0,
        min_group_size: int = 5,
        use_pyg_lib: bool = None,
        batch_size: int = 1_000_000,
        **kwargs,
    ) -> torch.Tensor:
        """
        Cosmic web clustering optimized for 50M+ objects.

        Args:
            method: 'fof' (Friends of Friends) or 'hierarchical'
            linking_length: Linking length in parsecs
            min_group_size: Minimum cluster size
            use_pyg_lib: Use native pyg-lib if available
            batch_size: Batch size for large-scale processing

        Returns:
            Cluster labels tensor
        """
        coords = self.coordinates
        n_objects = coords.size(0)

        # For very large datasets, use hierarchical approach
        if n_objects > 10_000_000:
            return self._hierarchical_cosmic_web_clustering(
                coords, linking_length, min_group_size, batch_size
            )

        if method == "fof":
            labels = self._friends_of_friends_clustering(
                coords, linking_length, min_group_size, use_pyg_lib or HAS_PYG_LIB
            )
            return labels.to(coords.device)
        elif method == "hierarchical":
            labels = self._hierarchical_clustering_pyg(
                coords, linking_length, min_group_size
            )
            return labels.to(coords.device)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _hierarchical_cosmic_web_clustering(
        self,
        coords: torch.Tensor,
        linking_length: float,
        min_group_size: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Hierarchical clustering for 50M+ objects using spatial subdivision."""
        n_objects = coords.size(0)
        device = coords.device

        # Initialize labels
        labels = torch.full((n_objects,), -1, dtype=torch.long, device=device)
        current_label = 0

        # Create spatial hierarchy
        logger.info(f"Building spatial hierarchy for {n_objects:,} objects...")

        # Level 1: Coarse partitioning (1 Mpc cells for galaxies, 100 pc for stars)
        coarse_scale = linking_length * 100
        coarse_partitions = self._grid_partition(coords, coarse_scale)

        # Process each coarse partition
        for coarse_idx, coarse_partition in enumerate(coarse_partitions):
            if len(coarse_partition) < min_group_size:
                continue

            # Level 2: Fine partitioning within coarse cells
            fine_coords = coords[coarse_partition]
            fine_partitions = self._grid_partition(fine_coords, linking_length * 10)

            for fine_partition in fine_partitions:
                if len(fine_partition) < min_group_size:
                    continue

                # Get global indices
                global_indices = coarse_partition[fine_partition]
                partition_coords = coords[global_indices]

                # Local clustering within fine partition
                local_labels = self._local_fof_clustering(
                    partition_coords, linking_length, min_group_size
                )

                # Assign global labels
                unique_local = torch.unique(local_labels[local_labels >= 0])
                for local_label in unique_local:
                    mask = local_labels == local_label
                    labels[global_indices[mask]] = current_label
                    current_label += 1

            if coarse_idx % 100 == 0:
                logger.info(
                    f"Processed {coarse_idx}/{len(coarse_partitions)} coarse partitions"
                )

        # Merge clusters across partition boundaries
        labels = self._merge_boundary_clusters(coords, labels, linking_length)

        return labels

    def _grid_partition(
        self, coords: torch.Tensor, cell_size: float
    ) -> List[torch.Tensor]:
        """Partition coordinates into grid cells."""
        # Compute grid indices
        min_coords = coords.min(dim=0)[0]
        grid_indices = ((coords - min_coords) / cell_size).long()

        # Create unique cell identifiers
        grid_shape = grid_indices.max(dim=0)[0] + 1
        cell_ids = (
            grid_indices[:, 0] * grid_shape[1] * grid_shape[2]
            + grid_indices[:, 1] * grid_shape[2]
            + grid_indices[:, 2]
        )

        # Group by cell
        unique_cells = torch.unique(cell_ids)
        partitions = []

        for cell_id in unique_cells:
            mask = cell_ids == cell_id
            indices = torch.where(mask)[0]
            if len(indices) > 0:
                partitions.append(indices)

        return partitions

    def _local_fof_clustering(
        self, coords: torch.Tensor, linking_length: float, min_group_size: int
    ) -> torch.Tensor:
        """Local Friends-of-Friends clustering for a partition."""
        n_points = coords.size(0)

        if n_points == 0:
            return torch.empty(0, dtype=torch.long)

        # Build radius graph
        edge_index = radius_graph(
            coords, r=linking_length, loop=False, max_num_neighbors=256
        )

        # Find connected components using PyG utilities
        if HAS_PYG_LIB:
            try:
                labels = pyg_lib.ops.connected_components(edge_index, n_points)
            except:
                labels = self._torch_connected_components(edge_index, n_points)
        else:
            labels = self._torch_connected_components(edge_index, n_points)

        # Filter small groups
        labels = self._filter_small_clusters(labels, min_group_size)

        return labels

    def _merge_boundary_clusters(
        self, coords: torch.Tensor, labels: torch.Tensor, linking_length: float
    ) -> torch.Tensor:
        """Merge clusters that should be connected across partition boundaries."""
        unique_labels = torch.unique(labels[labels >= 0])

        if len(unique_labels) < 2:
            return labels

        # Sample points from each cluster for boundary checking
        cluster_samples = {}
        for label in unique_labels:
            mask = labels == label
            cluster_coords = coords[mask]

            # Use farthest point sampling for representative points
            if len(cluster_coords) > 100:
                sample_indices = fps(cluster_coords, ratio=0.1, random_start=True)
                cluster_samples[label.item()] = cluster_coords[sample_indices]
            else:
                cluster_samples[label.item()] = cluster_coords

        # Check for connections between clusters
        merge_pairs = []
        cluster_list = list(cluster_samples.keys())

        for i in range(len(cluster_list)):
            for j in range(i + 1, len(cluster_list)):
                label_i, label_j = cluster_list[i], cluster_list[j]
                coords_i = cluster_samples[label_i]
                coords_j = cluster_samples[label_j]

                # Compute minimum distance between clusters
                if self._clusters_connected(coords_i, coords_j, linking_length):
                    merge_pairs.append((label_i, label_j))

        # Merge connected clusters
        if merge_pairs:
            labels = self._merge_labels(labels, merge_pairs)

        return labels

    def _clusters_connected(
        self, coords1: torch.Tensor, coords2: torch.Tensor, threshold: float
    ) -> bool:
        """Check if two clusters are connected within threshold distance."""
        # Efficient distance computation using broadcasting
        # Sample if too large
        if len(coords1) * len(coords2) > 1_000_000:
            n_samples = int(np.sqrt(1_000_000))
            if len(coords1) > n_samples:
                idx1 = torch.randperm(len(coords1))[:n_samples]
                coords1 = coords1[idx1]
            if len(coords2) > n_samples:
                idx2 = torch.randperm(len(coords2))[:n_samples]
                coords2 = coords2[idx2]

        # Compute pairwise distances
        dists = torch.cdist(coords1, coords2)
        min_dist = dists.min()

        return min_dist <= threshold

    def _merge_labels(
        self, labels: torch.Tensor, merge_pairs: List[Tuple[int, int]]
    ) -> torch.Tensor:
        """Merge cluster labels based on merge pairs."""
        # Build equivalence classes
        equivalence = {}
        for label1, label2 in merge_pairs:
            # Find roots
            root1 = label1
            while root1 in equivalence:
                root1 = equivalence[root1]
            root2 = label2
            while root2 in equivalence:
                root2 = equivalence[root2]

            # Union
            if root1 != root2:
                equivalence[max(root1, root2)] = min(root1, root2)

        # Create mapping
        label_mapping = {}
        unique_labels = torch.unique(labels[labels >= 0])

        for label in unique_labels:
            label_val = label.item()
            # Find root
            root = label_val
            while root in equivalence:
                root = equivalence[root]
            label_mapping[label_val] = root

        # Apply mapping
        new_labels = labels.clone()
        for old_label, new_label in label_mapping.items():
            new_labels[labels == old_label] = new_label

        # Renumber consecutively
        unique_new = torch.unique(new_labels[new_labels >= 0])
        for i, label in enumerate(unique_new):
            new_labels[new_labels == label] = i

        return new_labels

    def _friends_of_friends_clustering(
        self,
        coords: torch.Tensor,
        linking_length: float,
        min_group_size: int,
        use_pyg_lib: bool,
    ) -> torch.Tensor:
        """Optimized Friends-of-Friends clustering."""

        # Build radius graph
        edge_index = radius_graph(
            coords, r=linking_length, loop=False, max_num_neighbors=256
        )

        num_nodes = coords.size(0)
        device = coords.device

        if use_pyg_lib and HAS_PYG_LIB:
            # Use native pyg-lib connected components
            try:
                labels = pyg_lib.ops.connected_components(edge_index, num_nodes)
            except ImportError:
                # Fallback to torch implementation
                labels = self._torch_connected_components(
                    edge_index, num_nodes, device=device
                )
        else:
            # Pure PyTorch implementation
            labels = self._torch_connected_components(
                edge_index, num_nodes, device=device
            )

        # Filter small groups and ensure correct device
        filtered_labels = self._filter_small_clusters(labels, min_group_size)
        return filtered_labels.to(device)

    def _torch_connected_components(
        self, edge_index: torch.Tensor, num_nodes: int, device=None
    ) -> torch.Tensor:
        """Pure PyTorch connected components implementation."""
        if device is None:
            device = edge_index.device

        # Disable compilation on Windows to avoid C++ compiler issues
        if (
            self.enable_compilation
            and "_fast_clustering_kernel" not in self._compiled_functions
            and not torch.cuda.is_available()
        ):
            # Create a standalone function for compilation to avoid self parameter issues
            # when using torch.compile in PyTorch 2.0+.
            def clustering_kernel(
                edge_index: torch.Tensor, num_nodes: int
            ) -> torch.Tensor:
                return self._fast_clustering_kernel_implementation(
                    edge_index, num_nodes, device=device
                )

            try:
                self._compiled_functions["_fast_clustering_kernel"] = torch.compile(
                    clustering_kernel, mode="default"
                )
                kernel = self._compiled_functions["_fast_clustering_kernel"]
            except Exception as e:
                logger.warning(f"Compilation failed: {e}. Using uncompiled version.")

                def kernel(ei, nn):
                    return self._fast_clustering_kernel_implementation(
                        ei, nn, device=device
                    )
        else:

            def kernel(ei, nn):
                return self._fast_clustering_kernel_implementation(
                    ei, nn, device=device
                )

        return kernel(edge_index, num_nodes)

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

    def _dbscan_clustering(
        self, coords: torch.Tensor, eps: float, min_samples: int, use_pyg_lib: bool
    ) -> torch.Tensor:
        """DBSCAN clustering with PyG optimizations."""

        # Build radius graph for neighborhoods
        edge_index = radius_graph(coords, r=eps, loop=False, max_num_neighbors=256)

        # Count neighbors efficiently
        row, col = edge_index
        neighbor_counts = degree(row, num_nodes=coords.size(0))

        # Identify core points
        is_core = neighbor_counts >= min_samples

        # Initialize labels
        labels = torch.full(
            (coords.size(0),), -1, device=coords.device, dtype=torch.long
        )
        current_label = 0

        # Process core points using BFS
        for core_idx in torch.where(is_core)[0]:
            if labels[core_idx] != -1:
                continue

            # Start new cluster
            queue = [core_idx.item()]
            labels[core_idx] = current_label

            while queue:
                current = queue.pop(0)

                # Get neighbors
                neighbor_mask = row == current
                neighbors = col[neighbor_mask]

                for neighbor in neighbors:
                    if labels[neighbor] == -1:
                        labels[neighbor] = current_label
                        if is_core[neighbor]:
                            queue.append(neighbor.item())

            current_label += 1

        return labels

    def multi_scale_analysis(
        self,
        scales: List[float] = [5.0, 10.0, 20.0, 50.0],
        methods: List[str] = ["degree", "clustering", "density"],
        use_caching: bool = True,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Multi-scale cosmic web structure analysis.

        Args:
            scales: Analysis scales in parsecs
            methods: Analysis methods to apply
            use_caching: Use cached edge indices

        Returns:
            Nested dictionary with results per scale and method
        """

        coords = self.coordinates
        results = {}

        for scale in scales:
            scale_results = {}

            # Build graph at this scale
            edge_index = self.build_edge_index(method="radius", r=scale)

            if "degree" in methods:
                # Node degree distribution
                degrees = degree(edge_index.tensor[0], num_nodes=coords.size(0))
                scale_results["degree"] = degrees

            if "clustering" in methods:
                # Local clustering coefficient
                clustering_coeff = self._calculate_clustering_coefficient(edge_index)
                scale_results["clustering"] = clustering_coeff

            if "density" in methods:
                # Local density
                neighbor_counts = degree(edge_index.tensor[0], num_nodes=coords.size(0))
                volume = (4 / 3) * torch.pi * (scale**3)
                density = neighbor_counts.float() / volume
                scale_results["density"] = density

            results[f"{scale}pc"] = scale_results

        return results

    def _calculate_clustering_coefficient(self, edge_index: EdgeIndex) -> torch.Tensor:
        """Calculate local clustering coefficient efficiently."""

        edge_tensor = edge_index.tensor
        num_nodes = edge_index.sparse_size[0]

        # Convert to undirected if needed
        if not edge_index.is_undirected:
            edge_tensor = to_undirected(edge_tensor, num_nodes)

        clustering = torch.zeros(
            num_nodes, dtype=torch.float32, device=edge_tensor.device
        )

        # Calculate for each node
        for node in range(num_nodes):
            # Find neighbors
            neighbors = edge_tensor[1][edge_tensor[0] == node]

            if len(neighbors) < 2:
                clustering[node] = 0.0
                continue

            # Count triangles
            triangles = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    # Check if edge exists between n1 and n2
                    if ((edge_tensor[0] == n1) & (edge_tensor[1] == n2)).any():
                        triangles += 1

            # Clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
            clustering[node] = triangles / possible_edges if possible_edges > 0 else 0.0

        return clustering

    def to_frame(self, frame: str, **kwargs) -> "SpatialTensorDict":
        """Transform to different coordinate frame."""

        # Transform using AstroPy
        new_skycoord = self.skycoord.transform_to(frame)

        # Create new SpatialTensorDict
        result = SpatialTensorDict(
            new_skycoord,
            coordinate_system=frame,
            enable_compilation=self.enable_compilation,
            cache_edge_indices=self.cache_edge_indices,
            **kwargs,
        )

        return result

    def cone_search(
        self,
        center: Union[SkyCoord, torch.Tensor, Tuple[float, float]],
        radius: u.Quantity,
    ) -> torch.Tensor:
        """Cone search around a position."""

        if isinstance(center, (tuple, list)):
            center = SkyCoord(
                center[0] * u.Unit("deg"),
                center[1] * u.Unit("deg"),
                frame=self.skycoord.frame,
            )
        elif isinstance(center, torch.Tensor):
            center = SkyCoord(
                center[0] * u.Unit("deg"),
                center[1] * u.Unit("deg"),
                frame=self.skycoord.frame,
            )

        # Use AstroPy separation
        separations = self.skycoord.separation(center)
        mask = separations < radius

        return torch.tensor(mask, dtype=torch.bool)

    def _hierarchical_clustering_pyg(
        self, coords: torch.Tensor, threshold: float, min_group_size: int
    ) -> torch.Tensor:
        """Hierarchical clustering using PyTorch Geometric utilities."""
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
        if HAS_PYG_LIB:
            try:
                labels = pyg_lib.ops.connected_components(edge_index, n_points)
            except:
                labels = self._torch_connected_components(edge_index, n_points)
        else:
            labels = self._torch_connected_components(edge_index, n_points)

        # Filter small groups
        labels = self._filter_small_clusters(labels, min_group_size)

        return labels

    def create_neighbor_loader(
        self,
        batch_size: int = 1024,
        num_neighbors: List[int] = [15, 10, 5],
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> NeighborLoader:
        """
        Create efficient NeighborLoader for large-scale processing.

        Optimized for 50M+ objects with:
        - Memory-efficient sampling
        - Multi-worker data loading
        - Persistent workers for reduced overhead
        """

        # Build base graph data
        data = self.build_pyg_data(**kwargs)

        # Add index for efficient sampling
        if not hasattr(data, "n_id"):
            data.n_id = torch.arange(data.num_nodes)

        # Create NeighborLoader with optimizations
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
            # Additional optimizations for large data
            replace=False,  # No replacement sampling
            directed=True,  # More efficient for large graphs
            time_attr=None,
            transform=None,
            filter_per_worker=True if num_workers > 0 else False,
        )

        return loader

    def create_cluster_loader(
        self,
        num_parts: int = 1000,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        save_dir: Optional[str] = None,
        **kwargs,
    ) -> ClusterLoader:
        """
        Create ClusterLoader for distributed processing of 50M+ objects.

        Uses METIS partitioning for balanced clusters.
        """

        # Build base graph data
        data = self.build_pyg_data(**kwargs)

        # Create cluster data with METIS partitioning
        cluster_data = ClusterData(
            data,
            num_parts=num_parts,
            recursive=False,
            save_dir=save_dir or "./clusters",
            log=True,
        )

        # Create loader
        loader = ClusterLoader(
            cluster_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
        )

        logger.info(f"Created ClusterLoader with {num_parts} partitions")

        return loader

    def subsample_uniform(
        self, n_samples: int, seed: Optional[int] = None
    ) -> "SpatialTensorDict":
        """
        Uniform subsampling for manageable analysis of large datasets.

        Args:
            n_samples: Number of samples to select
            seed: Random seed for reproducibility

        Returns:
            New SpatialTensorDict with subsampled data
        """
        n_total = len(self.coordinates)

        if n_samples >= n_total:
            return self

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # Random sampling
        indices = torch.randperm(n_total)[:n_samples]

        # Create new SpatialTensorDict
        subsampled_coords = self.coordinates[indices]

        # Copy other attributes if they exist
        new_data = {"coordinates": subsampled_coords}
        for key in self.keys():
            if key != "coordinates" and key != "meta":
                if isinstance(self[key], torch.Tensor) and len(self[key]) == n_total:
                    new_data[key] = self[key][indices]

        # Update metadata
        new_data["meta"] = self["meta"].copy()
        new_data["meta"]["n_objects"] = n_samples
        new_data["meta"]["subsampled"] = True
        new_data["meta"]["subsample_fraction"] = n_samples / n_total

        return SpatialTensorDict(
            subsampled_coords,
            coordinate_system=self["meta"]["coordinate_system"],
            unit=self["meta"]["unit"],
            epoch=self["meta"]["epoch"],
            enable_compilation=self.enable_compilation,
            cache_edge_indices=self.cache_edge_indices,
        )

    def subsample_fps(
        self, n_samples: int, seed: Optional[int] = None
    ) -> "SpatialTensorDict":
        """
        Farthest Point Sampling for better spatial coverage.

        Args:
            n_samples: Number of samples to select
            seed: Random seed for initial point

        Returns:
            New SpatialTensorDict with FPS-sampled data
        """
        n_total = len(self.coordinates)

        if n_samples >= n_total:
            return self

        # Use PyG's FPS
        ratio = n_samples / n_total
        indices = fps(
            self.coordinates,
            ratio=ratio,
            random_start=seed is not None,
        )

        # Create subsampled SpatialTensorDict
        subsampled_coords = self.coordinates[indices]

        return SpatialTensorDict(
            subsampled_coords,
            coordinate_system=self["meta"]["coordinate_system"],
            unit=self["meta"]["unit"],
            epoch=self["meta"]["epoch"],
            enable_compilation=self.enable_compilation,
            cache_edge_indices=self.cache_edge_indices,
        )

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""

        coords = self.coordinates

        # Build default graph
        edge_index = self.build_edge_index(method="knn", k=10)

        # Basic statistics
        num_nodes = coords.size(0)
        num_edges = edge_index.num_edges

        # Degree statistics
        degrees = degree(edge_index.tensor[0], num_nodes=coords.size(0))

        # Coordinate bounds
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

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": num_edges / (num_nodes * (num_nodes - 1) / 2),
            "mean_degree": degrees.float().mean().item(),
            "max_degree": degrees.max().item(),
            "coordinate_bounds": bounds,
            "volume_pc3": volume,
            "number_density": num_nodes / volume,
            "coordinate_system": self["meta"]["coordinate_system"],
            "has_cached_indices": len(self._edge_index_cache),
            "pyg_lib_available": HAS_PYG_LIB,
        }

    def validate(self) -> bool:
        """Validate spatial tensor data."""

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

        return True

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._edge_index_cache.clear()
        self._graph_cache.clear()
        self._compiled_functions.clear()
        logger.info("Cleared all caches")

    def to_trimesh(self) -> "TrimeshType":
        """
        Convert the spatial tensor to a trimesh.Trimesh mesh using PyG's to_trimesh utility.
        Returns:
            trimesh.Trimesh object
        Raises:
            ImportError if torch-geometric or trimesh is not installed
        """
        try:
            from torch_geometric.utils import to_trimesh
        except ImportError:
            raise ImportError("torch-geometric >=2.4 required for to_trimesh")
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is not installed. Please install with 'pip install trimesh'."
            )
        data = self.build_pyg_data()
        mesh = to_trimesh(data)
        return mesh

    def delaunay_mesh(self, return_trimesh: bool = True) -> Optional["TrimeshType"]:
        """
        Create a mesh from the coordinates using Delaunay triangulation.
        Args:
            return_trimesh: If True, return trimesh.Trimesh, else (vertices, faces)
        Returns:
            trimesh.Trimesh or (vertices, faces)
        Raises:
            ImportError if scipy or trimesh is not installed
        """
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise ImportError("scipy is required for Delaunay triangulation.")
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for Delaunay triangulation.")
        points = self.coordinates.detach().cpu().numpy()
        if points.shape[1] != 3:
            raise ValueError("Coordinates must be of shape (N, 3)")
        tri = Delaunay(points)
        faces = tri.simplices
        if return_trimesh:
            try:
                import trimesh
            except ImportError:
                raise ImportError("trimesh is required to return a Trimesh object.")
            mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            return mesh
        else:
            return points, faces

    def convex_hull(self, return_trimesh: bool = True) -> Optional["TrimeshType"]:
        """
        Compute the convex hull mesh of the coordinates.
        Args:
            return_trimesh: If True, return trimesh.Trimesh, else (vertices, faces)
        Returns:
            trimesh.Trimesh or (vertices, faces)
        Raises:
            ImportError if scipy or trimesh is not installed
        """
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            raise ImportError("scipy is required for ConvexHull.")
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for ConvexHull.")
        points = self.coordinates.detach().cpu().numpy()
        if points.shape[1] != 3:
            raise ValueError("Coordinates must be of shape (N, 3)")
        hull = ConvexHull(points)
        faces = hull.simplices
        if return_trimesh:
            try:
                import trimesh
            except ImportError:
                raise ImportError("trimesh is required to return a Trimesh object.")
            mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            return mesh
        else:
            return points, faces

    def kdtree(self) -> "KDTreeType":
        """
        Return a cKDTree for fast neighbor queries on the coordinates.
        Returns:
            scipy.spatial.cKDTree object
        Raises:
            ImportError if scipy is not installed
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("scipy is required for KDTree.")
        points = self.coordinates.detach().cpu().numpy()
        return cKDTree(points)

    def alpha_shape(self, alpha: float = 0.03, backend: str = "open3d"):
        """
        Compute the alpha shape (concave hull) mesh of the coordinates using Open3D.
        Args:
            alpha: Alpha value for the shape (smaller = tighter hull)
            backend: Only 'open3d' is supported
        Returns:
            open3d.geometry.TriangleMesh
        """
        import numpy as np
        import open3d as o3d

        points = self.coordinates.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        return mesh

    def knn_graph(self, k: int = 10, batch=None, loop: bool = False):
        """
        Build k-nearest neighbor graph using torch_geometric.nn.knn_graph.
        Args:
            k: Number of neighbors
            batch: Optional batch vector
            loop: Include self-loops
        Returns:
            edge_index (torch.LongTensor)
        """
        from torch_geometric.nn import knn_graph

        coords = self.coordinates
        return knn_graph(coords, k=k, batch=batch, loop=loop)

    def radius_graph(
        self,
        r: float = 10.0,
        batch=None,
        loop: bool = False,
        max_num_neighbors: int = 64,
    ):
        """
        Build radius graph using torch_geometric.nn.radius_graph.
        Args:
            r: Radius
            batch: Optional batch vector
            loop: Include self-loops
            max_num_neighbors: Maximum neighbors per node
        Returns:
            edge_index (torch.LongTensor)
        """
        from torch_geometric.nn import radius_graph

        coords = self.coordinates
        return radius_graph(
            coords, r=r, batch=batch, loop=loop, max_num_neighbors=max_num_neighbors
        )

    def minimum_spanning_tree(self, edge_index=None, edge_weight=None):
        """
        Compute the minimum spanning tree (MST) of the graph using torch_geometric.utils.minimum_spanning_tree.
        Args:
            edge_index: Edge indices (optional, will use knn_graph if None)
            edge_weight: Edge weights (optional, will compute Euclidean if None)
        Returns:
            mst_edge_index (torch.LongTensor)
        """
        import torch
        from torch_geometric.utils import minimum_spanning_tree

        coords = self.coordinates
        if edge_index is None:
            edge_index = self.knn_graph(k=10)
        if edge_weight is None:
            # Compute Euclidean distances for each edge
            src, dst = edge_index
            edge_weight = torch.norm(coords[src] - coords[dst], dim=1)
        mst = minimum_spanning_tree(edge_index, edge_weight)
        return mst
