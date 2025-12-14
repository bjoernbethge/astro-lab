"""
Base Classes for Graph Building
==============================

Provides base classes and configuration for centralized graph construction.
Implements state-of-the-art best practices for astronomical graph building.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import AddSelfLoops, RemoveIsolatedNodes

from astro_lab.tensors import SurveyTensorDict


@dataclass
class GraphConfig:
    """Configuration for graph building with best practices."""

    # Graph construction method
    method: str = "knn"  # "knn", "radius", "astronomical", "adaptive", "multiscale"

    # KNN parameters
    k_neighbors: int = 16  # Increased default for better connectivity
    k_min: int = 4  # Minimum neighbors for adaptive methods
    k_max: int = 32  # Maximum neighbors for adaptive methods

    # Radius parameters
    radius: float = 1.0
    radius_min: float = 0.1
    radius_max: float = 10.0

    # Astronomical parameters
    use_3d_coordinates: bool = True
    coordinate_system: str = "cartesian"  # "cartesian", "spherical", "galactic"
    distance_metric: str = "euclidean"  # "euclidean", "angular", "mahalanobis"

    # Feature selection
    use_photometry: bool = True
    use_astrometry: bool = True
    use_spectroscopy: bool = False
    use_temporal: bool = False

    # Feature preprocessing
    normalize_features: bool = True
    normalize_method: str = "standardize"  # "standardize", "minmax", "robust"
    handle_nan: str = "median"  # "median", "mean", "zero", "drop"
    outlier_detection: bool = True
    outlier_method: str = "zscore"  # "zscore", "iqr", "isolation_forest"
    outlier_threshold: float = 3.0

    # Graph properties
    directed: bool = False
    self_loops: bool = False
    remove_isolated: bool = True
    ensure_connected: bool = True

    # Multi-scale graph
    use_multiscale: bool = False
    scales: List[int] = field(default_factory=lambda: [8, 16, 32])

    # Heterogeneous graph
    use_hetero: bool = False
    node_types: List[str] = field(default_factory=lambda: ["star", "galaxy"])
    edge_types: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [("star", "near", "star"), ("galaxy", "near", "galaxy")]
    )

    # Performance
    device: Optional[Union[str, torch.device]] = None
    use_gpu_construction: bool = True
    batch_size: Optional[int] = None  # For batch processing large graphs
    num_workers: int = 0
    prefetch_factor: int = 2

    # Caching
    use_cache: bool = True
    cache_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = ["knn", "radius", "astronomical", "adaptive", "multiscale"]
        if self.method not in valid_methods:
            raise ValueError(f"Unknown method: {self.method}. Valid: {valid_methods}")

        if self.coordinate_system not in ["cartesian", "spherical", "galactic"]:
            raise ValueError(f"Unknown coordinate system: {self.coordinate_system}")

        if self.k_neighbors <= 0 or self.k_neighbors > 1000:
            raise ValueError(f"k_neighbors must be in (0, 1000], got {self.k_neighbors}")

        if self.k_min >= self.k_max:
            raise ValueError(f"k_min ({self.k_min}) must be < k_max ({self.k_max})")

        if self.outlier_threshold <= 0:
            raise ValueError(f"outlier_threshold must be positive, got {self.outlier_threshold}")


class BaseGraphBuilder(ABC):
    """Base class for graph builders with best practices."""

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self.device = self._setup_device()
        self._setup_logging()
        self._transforms = self._setup_transforms()

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device:
            return torch.device(self.config.device)
        
        if self.config.use_gpu_construction and torch.cuda.is_available():
            return torch.device("cuda")
        
        return torch.device("cpu")

    def _setup_logging(self):
        """Setup logging configuration."""
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_transforms(self) -> List[Any]:
        """Setup graph transforms."""
        transforms = []
        
        if self.config.self_loops:
            transforms.append(AddSelfLoops())
        
        if self.config.remove_isolated:
            transforms.append(RemoveIsolatedNodes())
        
        return transforms

    @abstractmethod
    def build(self, survey_tensor: SurveyTensorDict) -> Union[Data, HeteroData]:
        """Build graph from SurveyTensorDict."""
        pass

    def validate_input(self, survey_tensor: SurveyTensorDict) -> None:
        """Enhanced input validation."""
        if not isinstance(survey_tensor, SurveyTensorDict):
            raise TypeError(
                f"Input must be SurveyTensorDict, got {type(survey_tensor)}"
            )

        # Check for required spatial data
        if "spatial" not in survey_tensor:
            raise ValueError("SurveyTensorDict must contain 'spatial' data")

        # Validate spatial data has coordinates
        spatial_data = survey_tensor["spatial"]
        if "coordinates" not in spatial_data:
            raise ValueError("Spatial data must have 'coordinates' key")

        # Check coordinate dimensions
        coords = spatial_data["coordinates"]
        if coords.dim() < 2:
            raise ValueError(
                f"Coordinates must be at least 2D tensor, got shape {coords.shape}"
            )

        # Validate k_neighbors parameter
        n_objects = coords.shape[0]
        if self.config.k_neighbors >= n_objects:
            self.logger.warning(
                f"k_neighbors ({self.config.k_neighbors}) >= n_objects ({n_objects}). "
                f"Adjusting to {n_objects - 1}"
            )
            self.config.k_neighbors = min(self.config.k_neighbors, n_objects - 1)

    def extract_coordinates(self, survey_tensor: SurveyTensorDict) -> torch.Tensor:
        """Extract coordinates from SurveyTensorDict structure."""
        spatial_data = survey_tensor["spatial"]
        
        # Get coordinates from TensorDict structure
        coords = spatial_data["coordinates"]
        
        # Ensure proper dimensions
        if coords.dim() == 1:
            coords = coords.unsqueeze(1)
        
        # Handle NaN/Inf
        coords = self._handle_invalid_values(coords, "coordinates")
        
        # Convert coordinate system if needed
        if self.config.coordinate_system == "spherical" and coords.shape[1] >= 2:
            coords = self._convert_to_spherical(coords)
        elif self.config.coordinate_system == "galactic" and coords.shape[1] >= 2:
            coords = self._convert_to_galactic(coords)
        
        return coords.to(self.device)

    def extract_features(self, survey_tensor: SurveyTensorDict) -> torch.Tensor:
        """Extract features from SurveyTensorDict components."""
        features = []
        feature_names = []

        # Spatial features (coordinates and proper motions)
        if self.config.use_astrometry and "spatial" in survey_tensor:
            spatial_data = survey_tensor["spatial"]
            
            # Always include coordinates as features
            coords = spatial_data["coordinates"]
            features.append(coords)
            feature_names.extend([f"coord_{i}" for i in range(coords.shape[1])])
            
            # Proper motions if available in spatial data
            if "pmra" in spatial_data:
                pmra = spatial_data["pmra"]
                if pmra.dim() == 1:
                    pmra = pmra.unsqueeze(1)
                features.append(pmra)
                feature_names.append("pmra")
                
            if "pmdec" in spatial_data:
                pmdec = spatial_data["pmdec"]
                if pmdec.dim() == 1:
                    pmdec = pmdec.unsqueeze(1)
                features.append(pmdec)
                feature_names.append("pmdec")
            
            # Parallax if available
            if "parallax" in spatial_data:
                plx = spatial_data["parallax"]
                if plx.dim() == 1:
                    plx = plx.unsqueeze(1)
                features.append(plx)
                feature_names.append("parallax")

        # Photometric features
        if self.config.use_photometry and "photometric" in survey_tensor:
            phot_data = survey_tensor["photometric"]
            
            if "magnitudes" in phot_data:
                mags = phot_data["magnitudes"]
                if mags.dim() == 1:
                    mags = mags.unsqueeze(1)
                features.append(mags)
                
                # Add band names if available
                if hasattr(phot_data, "bands"):
                    feature_names.extend([f"mag_{band}" for band in phot_data.bands])
                else:
                    feature_names.extend([f"mag_{i}" for i in range(mags.shape[1])])
            
            # Colors if available
            if "colors" in phot_data:
                colors = phot_data["colors"]
                if colors.dim() == 1:
                    colors = colors.unsqueeze(1)
                features.append(colors)
                feature_names.extend([f"color_{i}" for i in range(colors.shape[1])])

        # Spectral features
        if self.config.use_spectroscopy and "spectral" in survey_tensor:
            spec_data = survey_tensor["spectral"]
            
            if "features" in spec_data:
                spec_features = spec_data["features"]
                if spec_features.dim() == 1:
                    spec_features = spec_features.unsqueeze(1)
                features.append(spec_features)
                feature_names.extend([f"spec_{i}" for i in range(spec_features.shape[1])])

        # Additional features if available
        if "features" in survey_tensor:
            extra_features = survey_tensor["features"]
            if extra_features.dim() == 1:
                extra_features = extra_features.unsqueeze(1)
            features.append(extra_features)
            feature_names.extend([f"feat_{i}" for i in range(extra_features.shape[1])])

        if not features:
            # Minimal fallback: use coordinates only
            self.logger.warning("No features found, using coordinates as features")
            coords = self.extract_coordinates(survey_tensor)
            features = [coords]
            feature_names = [f"coord_{i}" for i in range(coords.shape[1])]

        # Concatenate all features
        result = torch.cat(features, dim=-1).to(self.device)
        
        # Handle invalid values
        result = self._handle_invalid_values(result, "features")
        
        # Outlier detection
        if self.config.outlier_detection:
            result, outlier_mask = self._detect_outliers(result)
            self.logger.info(f"Detected {outlier_mask.sum()} outliers")
        
        # Normalize features
        if self.config.normalize_features:
            result = self._normalize_features(result)
        
        # Store feature names for interpretability
        self._feature_names = feature_names
        
        self.logger.info(
            f"Extracted {result.shape[1]} features from {result.shape[0]} objects"
        )
        
        return result

    def _handle_invalid_values(
        self, tensor: torch.Tensor, name: str
    ) -> torch.Tensor:
        """Handle NaN and Inf values."""
        # Count invalid values
        nan_mask = torch.isnan(tensor)
        inf_mask = torch.isinf(tensor)
        invalid_mask = nan_mask | inf_mask
        
        if invalid_mask.any():
            n_invalid = invalid_mask.sum().item()
            self.logger.warning(
                f"Found {n_invalid} invalid values in {name} "
                f"({100 * n_invalid / tensor.numel():.2f}%)"
            )
            
            if self.config.handle_nan == "median":
                # Replace with median
                for i in range(tensor.shape[1]):
                    col = tensor[:, i]
                    valid_mask = ~invalid_mask[:, i]
                    if valid_mask.any():
                        median_val = col[valid_mask].median()
                        col[invalid_mask[:, i]] = median_val
                    else:
                        col[invalid_mask[:, i]] = 0.0
            
            elif self.config.handle_nan == "mean":
                # Replace with mean
                for i in range(tensor.shape[1]):
                    col = tensor[:, i]
                    valid_mask = ~invalid_mask[:, i]
                    if valid_mask.any():
                        mean_val = col[valid_mask].mean()
                        col[invalid_mask[:, i]] = mean_val
                    else:
                        col[invalid_mask[:, i]] = 0.0
            
            elif self.config.handle_nan == "zero":
                # Replace with zero
                tensor[invalid_mask] = 0.0
            
            else:  # "drop"
                # This should be handled at a higher level
                self.logger.warning("'drop' method not implemented at tensor level")
                tensor[invalid_mask] = 0.0
        
        return tensor

    def _detect_outliers(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect outliers in features."""
        n_samples, n_features = features.shape
        outlier_mask = torch.zeros(n_samples, dtype=torch.bool, device=features.device)
        
        if self.config.outlier_method == "zscore":
            # Z-score based outlier detection
            for i in range(n_features):
                col = features[:, i]
                mean = col.mean()
                std = col.std()
                if std > 0:
                    z_scores = torch.abs((col - mean) / std)
                    outlier_mask |= z_scores > self.config.outlier_threshold
        
        elif self.config.outlier_method == "iqr":
            # IQR based outlier detection
            for i in range(n_features):
                col = features[:, i]
                q1 = torch.quantile(col, 0.25)
                q3 = torch.quantile(col, 0.75)
                iqr = q3 - q1
                lower = q1 - self.config.outlier_threshold * iqr
                upper = q3 + self.config.outlier_threshold * iqr
                outlier_mask |= (col < lower) | (col > upper)
        
        # Handle outliers (clip to non-outlier range)
        if outlier_mask.any():
            for i in range(n_features):
                col = features[:, i]
                non_outlier_mask = ~outlier_mask
                if non_outlier_mask.any():
                    min_val = col[non_outlier_mask].min()
                    max_val = col[non_outlier_mask].max()
                    col[outlier_mask] = torch.clamp(col[outlier_mask], min_val, max_val)
        
        return features, outlier_mask

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features."""
        if self.config.normalize_method == "standardize":
            # Standardization (z-score normalization)
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            std = torch.where(std > 0, std, torch.ones_like(std))
            features = (features - mean) / std
        
        elif self.config.normalize_method == "minmax":
            # Min-max normalization
            min_vals = features.min(dim=0, keepdim=True)[0]
            max_vals = features.max(dim=0, keepdim=True)[0]
            range_vals = max_vals - min_vals
            range_vals = torch.where(range_vals > 0, range_vals, torch.ones_like(range_vals))
            features = (features - min_vals) / range_vals
        
        elif self.config.normalize_method == "robust":
            # Robust normalization (using median and MAD)
            median = features.median(dim=0, keepdim=True)[0]
            mad = (features - median).abs().median(dim=0, keepdim=True)[0]
            mad = torch.where(mad > 0, mad, torch.ones_like(mad))
            features = (features - median) / (1.4826 * mad)  # 1.4826 ≈ 1/Φ^(-1)(0.75)
        
        return features

    def _convert_to_spherical(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert Cartesian to spherical coordinates."""
        if coords.shape[1] < 3:
            # Already spherical (RA, Dec)
            return coords
        
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        # Radius
        r = torch.sqrt(x**2 + y**2 + z**2)
        
        # Azimuth (RA)
        phi = torch.atan2(y, x)
        
        # Elevation (Dec)
        theta = torch.asin(z / (r + 1e-8))
        
        return torch.stack([phi, theta, r], dim=-1)

    def _convert_to_galactic(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert to galactic coordinates (placeholder)."""
        self.logger.warning("Galactic conversion not implemented, returning original")
        return coords

    def create_data_object(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        **kwargs,
    ) -> Data:
        """Create PyG Data object with enhanced validation."""
        # Validate dimensions
        n_nodes = features.size(0)
        if coords.size(0) != n_nodes:
            raise ValueError(
                f"Size mismatch: features {n_nodes} != coords {coords.size(0)}"
            )

        # Validate edge indices
        if edge_index.size(0) != 2:
            raise ValueError(
                f"Edge index must have shape [2, E], got {edge_index.shape}"
            )
        
        if edge_index.size(1) > 0:
            max_idx = edge_index.max()
            if max_idx >= n_nodes:
                # Filter invalid edges
                valid_mask = (edge_index[0] < n_nodes) & (edge_index[1] < n_nodes)
                edge_index = edge_index[:, valid_mask]
                self.logger.warning(
                    f"Filtered {(~valid_mask).sum()} invalid edges"
                )

        # Create data object
        data = Data(
            x=features,
            edge_index=edge_index,
            pos=coords,
            num_nodes=n_nodes,
            **kwargs
        )

        # Apply transforms
        for transform in self._transforms:
            data = transform(data)

        # Add metadata
        data.num_edges = edge_index.size(1)
        data.graph_method = self.config.method
        data.device = str(self.device)
        
        # Add feature names if available
        if hasattr(self, "_feature_names"):
            data.feature_names = self._feature_names

        # Ensure connectivity if requested
        if self.config.ensure_connected:
            data = self._ensure_connected(data)

        self.logger.info(
            f"Created graph: {data.num_nodes} nodes, {data.num_edges} edges, "
            f"{features.shape[1]} features"
        )

        return data

    def _ensure_connected(self, data: Data) -> Data:
        """Ensure graph is connected (placeholder)."""
        # TODO: Implement connected component analysis and connection
        return data

    def create_hetero_data_object(
        self,
        node_features: Dict[str, torch.Tensor],
        edge_indices: Dict[Tuple[str, str, str], torch.Tensor],
        node_coords: Dict[str, torch.Tensor],
        **kwargs,
    ) -> HeteroData:
        """Create heterogeneous graph data object."""
        data = HeteroData(**kwargs)
        
        # Add node features
        for node_type, features in node_features.items():
            data[node_type].x = features
            if node_type in node_coords:
                data[node_type].pos = node_coords[node_type]
        
        # Add edges
        for edge_type, edge_index in edge_indices.items():
            data[edge_type].edge_index = edge_index
        
        return data
