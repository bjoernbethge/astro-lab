"""
Exoplanet Datasets
=================

Datasets for exoplanet data including:
- NASA Exoplanet Archive datasets
- Planetary system graphs
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
import time

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset

from astro_lab.data.datasets.base import get_device, to_device, gpu_knn_graph, ASTRO_LAB_TENSORS_AVAILABLE

# Import tensors only for type annotations
if TYPE_CHECKING:
    from astro_lab.tensors import SurveyTensor, Spatial3DTensor


class ExoplanetGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for Exoplanet data.

    Creates spatial graphs from exoplanet systems with connections
    based on stellar distances and system properties.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        k_neighbors: int = 5,
        max_distance: float = 100.0,  # parsecs
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize Exoplanet graph dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        max_distance : float, default 100.0
            Maximum distance for connections (parsecs)
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

        if root is None:
            root = "data/processed/exoplanet_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["confirmed_exoplanets.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"exoplanet_graph_k{self.k_neighbors}_all.pt"]

    def download(self):
        """Download exoplanet data if needed."""
        raw_path = Path("data/processed/exoplanet_graphs/raw") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"📂 Exoplanet data should be available at: {raw_path}")
            if raw_path.exists():
                print(f"✅ Found exoplanet data")
            else:
                print(f"❌ Exoplanet data not found. Please ensure data is available.")
                raise FileNotFoundError("Exoplanet data not available")

    def process(self):
        """Process exoplanet catalog into graph format."""
        raw_path = Path("data/processed/exoplanet_graphs/raw") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing exoplanet catalog: {raw_path.name}")

        # Load catalog
        df = pl.read_parquet(raw_path)
        print(f"   Loaded {len(df)} exoplanets")
        print(f"   Available columns: {df.columns}")

        # Group by stellar system (same host star)
        systems = df.group_by("hostname").agg([
            pl.col("ra").first(),
            pl.col("dec").first(), 
            pl.col("sy_dist").first(),
            pl.col("pl_rade").mean().alias("avg_radius"),
            pl.col("pl_masse").mean().alias("avg_mass"),
            pl.col("disc_year").first().alias("discovery_year"),
            pl.count().alias("num_planets")
        ]).filter(pl.col("num_planets") >= 1)

        print(f"   Processing {len(systems)} planetary systems")

        # Convert to numpy for processing
        coords = systems.select(["ra", "dec"]).to_numpy()
        
        # Remove NaN coordinates
        valid_coords = ~np.isnan(coords).any(axis=1)
        coords = coords[valid_coords]
        systems_filtered = systems.filter(pl.Series(valid_coords))

        # Get k-nearest neighbors based on sky position
        print(f"   Computing {self.k_neighbors}-NN graph...")
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)

        # Extract features (system properties) - using available columns
        feature_cols = ["sy_dist", "avg_radius", "avg_mass", "discovery_year", "num_planets"]
        features = systems_filtered.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Create edge connections based on spatial proximity
        edge_index = []
        system_data = systems_filtered.to_dicts()  # Convert to list of dicts for easier access
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                # Check distance constraint (convert angular to physical distance)
                sys_i_dist = system_data[i].get("sy_dist")
                sys_j_dist = system_data[j].get("sy_dist")
                
                if sys_i_dist is not None and sys_j_dist is not None and not np.isnan(sys_i_dist) and not np.isnan(sys_j_dist):
                    # Approximate physical separation
                    angular_sep = distances[i][np.where(neighbors == j)[0][0]]  # radians
                    avg_distance = (sys_i_dist + sys_j_dist) / 2
                    physical_sep = angular_sep * avg_distance * 206265  # parsecs
                    
                    if physical_sep <= self.max_distance:
                        edge_index.append([i, j])
                else:
                    # If no distance info, use angular separation only
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_systems = len(systems_filtered)
        data.k_neighbors = self.k_neighbors
        data.max_distance = self.max_distance
        
        data_list = [data]

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed exoplanet graph with {len(systems_filtered)} systems and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with exoplanet system data
        survey_data = {
            'coordinates': data.pos,  # RA, Dec of host stars
            'distances': data.x[:, 0:1],  # System distances
            'periods': data.x[:, 1:2],  # Average orbital periods
            'radii': data.x[:, 2:3],  # Average planet radii
            'masses': data.x[:, 3:4],  # Average planet masses
            'num_planets': data.x[:, 4:5],  # Number of planets per system
            'num_objects': getattr(data, 'num_systems', data.num_nodes),
            'survey_name': 'NASA Exoplanet Archive'
        }
        
        try:
            # Import at runtime to avoid circular imports
            from astro_lab.tensors import SurveyTensor
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='NASA Exoplanet Archive'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_spatial_tensor(self) -> Optional["Spatial3DTensor"]:
        """Extract spatial coordinates as Spatial3DTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Convert RA/Dec + distance to 3D Cartesian coordinates
        ra_rad = torch.deg2rad(data.pos[:, 0])
        dec_rad = torch.deg2rad(data.pos[:, 1])
        distance = data.x[:, 0]  # parsecs
        
        # Convert to 3D Cartesian
        x = distance * torch.cos(dec_rad) * torch.cos(ra_rad)
        y = distance * torch.cos(dec_rad) * torch.sin(ra_rad)
        z = distance * torch.sin(dec_rad)
        
        coords_3d = torch.stack([x, y, z], dim=1)
        
        spatial_data = {
            'coordinates': coords_3d,
            'coordinate_system': 'cartesian',
            'units': 'parsec'
        }
        
        try:
            return Spatial3DTensor(data=spatial_data)
        except Exception as e:
            print(f"⚠️  Failed to create Spatial3DTensor: {e}")
            return None 