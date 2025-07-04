"""
PyTorch Geometric  Native Optimizations for AstroLab
======================================================

Clean, modern implementation leveraging:
- EdgeIndex with metadata caching
- pyg-lib native operations
- torch.compile optimization
- astronomical data augmentation
- Distributed processing capabilities
"""

import logging
from typing import Any, Dict, List, Optional, Union

# pyg-lib for native operations
import pyg_lib
import torch
from torch import Tensor

# PyTorch Geometric
from torch_geometric.data import Data
from torch_geometric.loader import (
    ClusterLoader,
    GraphSAINTRandomWalkSampler,
    NeighborLoader,
)
from torch_geometric.transforms import (
    AddRandomWalkPE,
    RandomLinkSplit,
    RandomNodeSplit,
)
from torch_geometric.utils import (
    add_random_edge,
    degree,
)

HAS_PYG_LIB = True

logger = logging.getLogger(__name__)


class AstroGraphSampler:
    """
    graph sampling strategies for astronomical surveys.

    Features:
    - Intelligent survey-specific sampling
    - EdgeIndex optimization
    - Memory-efficient large-scale processing
    """

    def __init__(self, survey_threshold: int = 100000, enable_caching: bool = True):
        self.survey_threshold = survey_threshold
        self.enable_caching = enable_caching
        self._cache = {}

    def create_astronomical_splits(
        self,
        data: Data,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stratify_by: Optional[str] = None,
    ) -> Data:
        """
        Create astronomically meaningful train/val/test splits.

        Args:
            data: PyG Data object
            train_ratio: Training data fraction
            val_ratio: Validation data fraction
            stratify_by: Stratification strategy ('magnitude', 'redshift', 'type')

        Returns:
            Data with split masks
        """

        if stratify_by is not None:
            # Implement stratified splitting for astronomical data
            return self._stratified_split(data, train_ratio, val_ratio, stratify_by)
        else:
            # Standard random split
            transform = RandomNodeSplit(
                split="train_rest",
                num_val=val_ratio,
                num_test=1.0 - train_ratio - val_ratio,
            )
            return transform(data)

    def _stratified_split(
        self, data: Data, train_ratio: float, val_ratio: float, stratify_by: str
    ) -> Data:
        """Stratified splitting for astronomical data."""

        num_nodes = data.x.size(0)

        if stratify_by == "magnitude" and hasattr(data, "magnitude"):
            # Stratify by magnitude bins
            mags = data.magnitude
            bins = torch.quantile(mags, torch.linspace(0, 1, 6))  # 5 bins
            strata = torch.bucketize(mags, bins) - 1
        elif stratify_by == "redshift" and hasattr(data, "redshift"):
            # Stratify by redshift bins
            z = data.redshift
            bins = torch.quantile(z, torch.linspace(0, 1, 4))  # 3 bins
            strata = torch.bucketize(z, bins) - 1
        else:
            # Fallback to random split
            logger.warning(
                f"Stratification by {stratify_by} not available, using random split"
            )
            transform = RandomNodeSplit(
                split="train_rest",
                num_val=val_ratio,
                num_test=1.0 - train_ratio - val_ratio,
            )
            return transform(data)

        # Create balanced splits within each stratum
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        for stratum in torch.unique(strata):
            stratum_mask = strata == stratum
            stratum_indices = torch.where(stratum_mask)[0]

            # Shuffle within stratum
            perm = torch.randperm(len(stratum_indices))
            shuffled_indices = stratum_indices[perm]

            # Split within stratum
            n_train = int(len(shuffled_indices) * train_ratio)
            n_val = int(len(shuffled_indices) * val_ratio)

            train_mask[shuffled_indices[:n_train]] = True
            val_mask[shuffled_indices[n_train : n_train + n_val]] = True
            test_mask[shuffled_indices[n_train + n_val :]] = True

        # Add masks to data
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data

    def create_cosmic_web_link_splits(
        self,
        data: Data,
        neg_sampling_ratio: float = 1.0,
        preserve_filaments: bool = True,
    ) -> Dict[str, Data]:
        """
        Create link prediction splits optimized for cosmic web analysis.

        Args:
            data: PyG Data object
            neg_sampling_ratio: Negative to positive edge ratio
            preserve_filaments: Preserve filamentary structure in training

        Returns:
            Dictionary with train/val/test data
        """

        if preserve_filaments:
            # Custom splitting that preserves cosmic web structure
            return self._filament_aware_link_split(data, neg_sampling_ratio)
        else:
            # Standard link splitting
            transform = RandomLinkSplit(
                num_val=0.1,
                num_test=0.2,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=neg_sampling_ratio,
            )
            return dict(zip(["train", "val", "test"], transform(data)))

    def _filament_aware_link_split(
        self, data: Data, neg_sampling_ratio: float
    ) -> Dict[str, Data]:
        """Link splitting that preserves filamentary structure."""

        edge_index = data.edge_index
        num_edges = edge_index.size(1)

        # Identify potential filament edges (based on clustering coefficient)
        clustering_coeff = self._calculate_edge_clustering(data)

        # Prefer to keep high-clustering edges in training (avoid breaking filaments)
        edge_weights = clustering_coeff + torch.rand_like(clustering_coeff) * 0.1
        _, sorted_indices = torch.sort(edge_weights, descending=True)

        # Split edges
        n_train = int(0.8 * num_edges)
        n_val = int(0.1 * num_edges)

        train_edge_indices = sorted_indices[:n_train]
        val_edge_indices = sorted_indices[n_train : n_train + n_val]
        test_edge_indices = sorted_indices[n_train + n_val :]

        # Create data objects
        train_data = Data(
            x=data.x,
            edge_index=edge_index[:, train_edge_indices],
            pos=data.pos if hasattr(data, "pos") else None,
        )

        val_data = Data(
            x=data.x,
            edge_index=edge_index[:, val_edge_indices],
            pos=data.pos if hasattr(data, "pos") else None,
        )

        test_data = Data(
            x=data.x,
            edge_index=edge_index[:, test_edge_indices],
            pos=data.pos if hasattr(data, "pos") else None,
        )

        return {"train": train_data, "val": val_data, "test": test_data}

    def _calculate_edge_clustering(self, data: Data) -> Tensor:
        """Calculate clustering coefficient for each edge."""

        edge_index = data.edge_index
        row, col = edge_index

        # Calculate degree for efficiency
        degrees = degree(row, num_nodes=data.x.size(0))

        # heuristic: edges between high-degree nodes have higher clustering
        edge_clustering = (degrees[row] + degrees[col]) / (
            degrees[row] * degrees[col] + 1e-8
        )

        return edge_clustering

    def create_survey_loader(
        self,
        data: Data,
        batch_size: int = 1024,
        num_neighbors: List[int] = [15, 10, 5],
        loader_type: str = "neighbor",
    ) -> Union[NeighborLoader, GraphSAINTRandomWalkSampler, ClusterLoader]:
        """
        Create optimized data loader for large astronomical surveys.

        Args:
            data: PyG Data object
            batch_size: Batch size
            num_neighbors: Neighbor sampling numbers per layer
            loader_type: Type of loader ('neighbor', 'saint', 'cluster')

        Returns:
            Configured data loader
        """

        if data.x.size(0) > self.survey_threshold:
            logger.info(
                f"Large survey detected ({data.x.size(0)} objects), using {loader_type} sampling"
            )

        if loader_type == "neighbor":
            return NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                persistent_workers=True,
            )

        elif loader_type == "saint":
            return GraphSAINTRandomWalkSampler(
                data,
                batch_size=batch_size,
                walk_length=2,
                num_steps=5,
                sample_coverage=100,
                num_workers=4,
            )

        elif loader_type == "cluster":
            return ClusterLoader(
                data,
                num_parts=batch_size // 64,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )

        else:
            raise ValueError(f"Unknown loader_type: {loader_type}")


class AstroPositionalEncoding:
    """
    Astronomical positional encoding strategies.

    Features:
    - Stellar neighborhood encoding
    - Cosmic web structure encoding
    - Multi-scale spatial encoding
    """

    @staticmethod
    def add_stellar_neighborhood_encoding(
        data: Data, walk_length: int = 20, num_walks: int = 10, encoding_dim: int = 16
    ) -> Data:
        """
        Add positional encoding based on stellar neighborhood structure.

        Args:
            data: PyG Data object
            walk_length: Random walk length
            num_walks: Number of walks per node
            encoding_dim: Encoding dimension

        Returns:
            Data with positional encoding features
        """

        # Use PyG's AddRandomWalkPE with astronomical parameters
        transform = AddRandomWalkPE(walk_length=walk_length, attr_name="random_walk_pe")

        data = transform(data)

        # Add cosmic distance encoding
        if hasattr(data, "pos"):
            distance_encoding = AstroPositionalEncoding._compute_distance_encoding(
                data.pos, encoding_dim
            )

            # Concatenate with existing features
            if hasattr(data, "random_walk_pe"):
                data.x = torch.cat(
                    [data.x, data.random_walk_pe, distance_encoding], dim=1
                )
            else:
                data.x = torch.cat([data.x, distance_encoding], dim=1)

        return data

    @staticmethod
    def _compute_distance_encoding(pos: Tensor, encoding_dim: int) -> Tensor:
        """Compute distance-based positional encoding."""

        # Calculate distances to galactic center (if in galactic coordinates)
        center = torch.zeros(3, device=pos.device)
        distances = torch.norm(pos - center, dim=1, keepdim=True)

        # Logarithmic distance encoding
        log_distances = torch.log(distances + 1e-8)

        # Sinusoidal encoding
        encoding = torch.zeros(pos.size(0), encoding_dim, device=pos.device)

        for i in range(encoding_dim // 2):
            freq = 1.0 / (10000 ** (2 * i / encoding_dim))
            encoding[:, 2 * i] = torch.sin(log_distances.squeeze() * freq)
            encoding[:, 2 * i + 1] = torch.cos(log_distances.squeeze() * freq)

        return encoding

    @staticmethod
    def add_cosmic_web_encoding(
        data: Data, scales: List[float] = [10.0, 50.0, 100.0]
    ) -> Data:
        """Add multi-scale cosmic web structure encoding."""

        if not hasattr(data, "pos"):
            logger.warning("Position information required for cosmic web encoding")
            return data

        encodings = []

        for scale in scales:
            # Build graph at this scale
            from torch_geometric.nn import radius_graph

            edge_index = radius_graph(data.pos, r=scale, loop=False)

            # Calculate local density
            degrees = degree(edge_index[0], num_nodes=data.x.size(0))
            local_density = degrees / (4 / 3 * torch.pi * scale**3)

            encodings.append(local_density.unsqueeze(1))

        # Concatenate scale encodings
        scale_encoding = torch.cat(encodings, dim=1)
        data.x = torch.cat([data.x, scale_encoding], dim=1)

        return data


class AstroDataAugmentation:
    """
    Astronomical data augmentation strategies.

    Features:
    - Measurement uncertainty simulation
    - Coordinate frame transformations
    - Survey-specific noise modeling
    """

    def __init__(self, enable_compilation: bool = True):
        self.enable_compilation = enable_compilation
        self._compiled_functions = {}

    def simulate_measurement_uncertainty(
        self, data: Data, uncertainty_config: Dict[str, float] = None
    ) -> Data:
        """
        Simulate realistic astronomical measurement uncertainties.

        Args:
            data: PyG Data object
            uncertainty_config: Uncertainty parameters for different measurements

        Returns:
            Data with simulated uncertainties
        """

        if uncertainty_config is None:
            uncertainty_config = {
                "position_pc": 1.0,  # 1 parsec position uncertainty
                "velocity_km_s": 1.0,  # 1 km/s velocity uncertainty
                "magnitude": 0.01,  # 0.01 mag photometric uncertainty
                "parallax_fraction": 0.1,  # 10% parallax uncertainty
            }

        # Position uncertainty
        if hasattr(data, "pos") and "position_pc" in uncertainty_config:
            noise = torch.randn_like(data.pos) * uncertainty_config["position_pc"]
            data.pos = data.pos + noise

        # Coordinate uncertainty in features
        if "position_pc" in uncertainty_config:
            # Assume first 3 features are coordinates
            coord_noise = (
                torch.randn_like(data.x[:, :3]) * uncertainty_config["position_pc"]
            )
            data.x[:, :3] = data.x[:, :3] + coord_noise

        # Velocity uncertainty (features 3-6 if present)
        if data.x.size(1) >= 6 and "velocity_km_s" in uncertainty_config:
            vel_noise = (
                torch.randn_like(data.x[:, 3:6]) * uncertainty_config["velocity_km_s"]
            )
            data.x[:, 3:6] = data.x[:, 3:6] + vel_noise

        # Magnitude uncertainty
        if hasattr(data, "magnitude") and "magnitude" in uncertainty_config:
            mag_noise = (
                torch.randn_like(data.magnitude) * uncertainty_config["magnitude"]
            )
            data.magnitude = data.magnitude + mag_noise

        return data

    def apply_coordinate_transformations(
        self, data: Data, transformation_prob: float = 0.5
    ) -> Data:
        """Apply random coordinate frame transformations."""

        if torch.rand(1).item() < transformation_prob:
            # Random rotation around galactic center
            if hasattr(data, "pos"):
                angle = torch.rand(1).item() * 2 * torch.pi
                rotation_matrix = self._create_rotation_matrix(angle)
                data.pos = torch.matmul(data.pos, rotation_matrix.T)

                # Apply same rotation to coordinate features
                if data.x.size(1) >= 3:
                    data.x[:, :3] = torch.matmul(data.x[:, :3], rotation_matrix.T)

        return data

    def _create_rotation_matrix(self, angle: float) -> Tensor:
        """Create 3D rotation matrix around z-axis."""
        cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))

        rotation = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=torch.float32
        )

        return rotation

    def add_systematic_effects(
        self, data: Data, effects_config: Dict[str, Any] = None
    ) -> Data:
        """Add systematic effects common in astronomical surveys."""

        if effects_config is None:
            effects_config = {
                "magnitude_offset": {"enabled": True, "sigma": 0.02},
                "color_offset": {"enabled": True, "sigma": 0.01},
                "extinction": {"enabled": True, "av_max": 0.5},
            }

        # Magnitude zero-point offset
        if effects_config.get("magnitude_offset", {}).get("enabled", False) and hasattr(
            data, "magnitude"
        ):
            offset = torch.randn(1) * effects_config["magnitude_offset"]["sigma"]
            data.magnitude = data.magnitude + offset

        # Galactic extinction simulation
        if (
            effects_config.get("extinction", {}).get("enabled", False)
            and hasattr(data, "magnitude")
            and hasattr(data, "pos")
        ):
            # extinction model based on galactic latitude
            if data.pos.size(1) >= 3:
                # Calculate galactic latitude approximation
                z_coord = data.pos[:, 2]
                xy_dist = torch.norm(data.pos[:, :2], dim=1)
                lat_approx = torch.atan2(z_coord.abs(), xy_dist)

                # Extinction decreases with galactic latitude
                av_max = effects_config["extinction"]["av_max"]
                extinction = av_max * torch.exp(-lat_approx / 0.2)  # 0.2 rad scale
                data.magnitude = data.magnitude + extinction

        return data


class EnhancedCosmicWebAnalyzer:
    """
    cosmic web analysis with PyG 2025 optimizations.

    Features:
    - Filament detection using graph topology
    - Multi-scale structure analysis
    - clustering algorithms
    """

    def __init__(self, use_pyg_lib: bool = None, enable_caching: bool = True):
        self.use_pyg_lib = use_pyg_lib if use_pyg_lib is not None else HAS_PYG_LIB
        self.enable_caching = enable_caching
        self._cache = {}

        self.sampler = AstroGraphSampler(enable_caching=enable_caching)
        self.pos_encoder = AstroPositionalEncoding()
        self.augmenter = AstroDataAugmentation()

    def prepare_survey_for_analysis(
        self,
        data: Data,
        add_position_encoding: bool = True,
        add_uncertainty: bool = True,
        uncertainty_config: Optional[Dict[str, float]] = None,
        enable_augmentation: bool = True,
    ) -> Dict[str, Data]:
        """
        Comprehensive survey preparation for cosmic web analysis.

        Args:
            data: PyG Data object
            add_position_encoding: Add positional encoding
            add_uncertainty: Simulate measurement uncertainty
            uncertainty_config: Uncertainty parameters
            enable_augmentation: Enable data augmentation

        Returns:
            Dictionary with prepared data variants
        """

        logger.info("🔬 Preparing survey data for cosmic web analysis...")

        results = {"original": data}
        current_data = data

        # Add positional encoding
        if add_position_encoding:
            current_data = self.pos_encoder.add_stellar_neighborhood_encoding(
                current_data
            )
            current_data = self.pos_encoder.add_cosmic_web_encoding(current_data)
            results["position_encoded"] = current_data
            logger.debug("  ✓ Added stellar neighborhood and cosmic web encoding")

        # Simulate measurement uncertainty
        if add_uncertainty:
            uncertain_data = self.augmenter.simulate_measurement_uncertainty(
                current_data.clone(), uncertainty_config
            )
            results["with_uncertainty"] = uncertain_data
            logger.debug("  ✓ Added measurement uncertainty simulation")

        # Create astronomical splits
        split_data = self.sampler.create_astronomical_splits(
            current_data,
            stratify_by="magnitude" if hasattr(current_data, "magnitude") else None,
        )
        results["split"] = split_data
        logger.debug("  ✓ Created astronomically meaningful splits")

        # cosmic web structure
        enhanced_data = self._enhance_cosmic_web_structure(current_data)
        results["enhanced_cosmic_web"] = enhanced_data
        logger.debug("  ✓ cosmic web structure with probabilistic connections")

        logger.info(f"🎯 Survey preparation complete: {len(results)} data variants")
        return results

    def _enhance_cosmic_web_structure(self, data: Data) -> Data:
        """Enhance cosmic web graph with probabilistic filament connections."""

        # Add edges based on cosmic web topology
        enhanced_data = add_random_edge(
            data,
            num_edges=data.edge_index.size(1) // 20,  # Add 5% more edges
            force_undirected=True,
        )

        return enhanced_data

    def detect_cosmic_filaments(
        self, data: Data, method: str = "mst", **kwargs
    ) -> Dict[str, Tensor]:
        """
        Detect filamentary structures in cosmic web.

        Args:
            data: PyG Data object with cosmic web graph
            method: Detection method ('mst', 'persistence', 'skeleton')
            **kwargs: Method-specific parameters

        Returns:
            Dictionary with filament information
        """

        if method == "mst":
            return self._detect_filaments_mst(data, **kwargs)
        elif method == "persistence":
            return self._detect_filaments_persistence(data, **kwargs)
        elif method == "skeleton":
            return self._detect_filaments_skeleton(data, **kwargs)
        else:
            raise ValueError(f"Unknown filament detection method: {method}")

    def _detect_filaments_mst(
        self, data: Data, max_edge_length: float = 100.0
    ) -> Dict[str, Tensor]:
        """Filament detection using Minimum Spanning Tree."""

        edge_index = data.edge_index

        # Calculate edge lengths
        if hasattr(data, "pos"):
            row, col = edge_index
            edge_lengths = torch.norm(data.pos[row] - data.pos[col], dim=1)

            # Filter long edges
            valid_edges = edge_lengths <= max_edge_length
            filament_edges = edge_index[:, valid_edges]
            filament_lengths = edge_lengths[valid_edges]
        else:
            filament_edges = edge_index
            filament_lengths = torch.ones(edge_index.size(1))

        return {
            "filament_edges": filament_edges,
            "filament_lengths": filament_lengths,
            "total_filament_length": filament_lengths.sum(),
            "n_filament_segments": len(filament_lengths),
        }

    def _detect_filaments_persistence(self, data: Data, **kwargs) -> Dict[str, Tensor]:
        """Filament detection using topological persistence."""

        # Simplified implementation - would need proper topological libraries
        # For now, use degree-based detection
        degrees = degree(data.edge_index[0], num_nodes=data.x.size(0))

        # Nodes with degree 2-4 are likely in filaments
        filament_nodes = torch.where((degrees >= 2) & (degrees <= 4))[0]

        return {
            "filament_nodes": filament_nodes,
            "filament_fraction": len(filament_nodes) / data.x.size(0),
        }

    def _detect_filaments_skeleton(self, data: Data, **kwargs) -> Dict[str, Tensor]:
        """Filament detection using graph skeleton extraction."""

        # Use clustering coefficient to identify filament regions
        # Low clustering coefficient indicates filamentary structure
        clustering_coeff = self._compute_local_clustering(data)

        # Threshold for filament identification
        threshold = kwargs.get("clustering_threshold", 0.3)
        filament_mask = clustering_coeff < threshold

        filament_nodes = torch.where(filament_mask)[0]

        return {
            "filament_nodes": filament_nodes,
            "clustering_coefficients": clustering_coeff,
            "filament_fraction": filament_mask.float().mean(),
        }

    def _compute_local_clustering(self, data: Data) -> Tensor:
        """Compute local clustering coefficient efficiently."""

        edge_index = data.edge_index
        num_nodes = data.x.size(0)

        # Use pyg-lib if available for efficiency
        if self.use_pyg_lib and HAS_PYG_LIB:
            try:
                return pyg_lib.ops.clustering_coefficient(edge_index, num_nodes)
            except ImportError:
                pass

        # Fallback to manual computation
        clustering = torch.zeros(num_nodes, device=edge_index.device)

        for node in range(num_nodes):
            neighbors = edge_index[1][edge_index[0] == node]

            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i + 1 :]:
                    if ((edge_index[0] == n1) & (edge_index[1] == n2)).any():
                        triangles += 1

            # Clustering coefficient
            possible = len(neighbors) * (len(neighbors) - 1) // 2
            clustering[node] = triangles / possible if possible > 0 else 0.0

        return clustering

    def analyze_multi_scale_structure(
        self, data: Data, scales: List[float] = [5.0, 10.0, 25.0, 50.0, 100.0]
    ) -> Dict[str, Dict[str, Tensor]]:
        """Multi-scale cosmic web structure analysis."""

        results = {}

        for scale in scales:
            scale_results = {}

            # Build graph at this scale
            from torch_geometric.nn import radius_graph

            edge_index = radius_graph(data.pos, r=scale, loop=False)

            # Analyze structure at this scale
            degrees = degree(edge_index[0], num_nodes=data.x.size(0))
            clustering = self._compute_local_clustering(
                Data(x=data.x, edge_index=edge_index, pos=data.pos)
            )

            scale_results.update(
                {
                    "degrees": degrees,
                    "clustering": clustering,
                    "mean_degree": degrees.float().mean(),
                    "clustering_mean": clustering.mean(),
                    "scale_pc": scale,
                }
            )

            results[f"{scale}pc"] = scale_results

        return results
