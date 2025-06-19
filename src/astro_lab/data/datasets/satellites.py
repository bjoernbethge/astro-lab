"""
Satellite Datasets
=================

Datasets for satellite orbital data including:
- Satellite orbit datasets
- Orbital mechanics datasets
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


class SatelliteOrbitDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for satellite orbital data.

    Creates spatial graphs from satellite orbits with connections
    based on orbital similarity and proximity.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_satellites: int = 100,
        k_neighbors: int = 5,
        orbital_similarity_threshold: float = 1000.0,  # km
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize Satellite orbit dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        max_satellites : int, default 100
            Maximum number of satellites to include
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        orbital_similarity_threshold : float, default 1000.0
            Orbital similarity threshold for connections (km)
        """
        self.max_satellites = max_satellites
        self.k_neighbors = k_neighbors
        self.orbital_similarity_threshold = orbital_similarity_threshold

        if root is None:
            root = "data/processed/satellite_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["orbits.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"satellite_graph_k{self.k_neighbors}_n{self.max_satellites}.pt"]

    def download(self):
        """Download satellite orbital data if needed."""
        from astro_lab.data.manager import download_satellite_data
        
        raw_path = Path("data/processed/satellites") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"🛰️ Downloading satellite orbital data automatically...")
            try:
                download_satellite_data()
                print(f"✅ Satellite data downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download satellite data: {e}")
                raise

    def process(self):
        """Process satellite orbital data into graph format."""
        raw_path = Path("data/processed/satellites") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing satellite orbital data: {raw_path.name}")

        # Load orbital catalog
        df = pl.read_parquet(raw_path)
        
        # Sample if too many satellites
        if len(df) > self.max_satellites:
            df = df.sample(self.max_satellites, seed=42)
            
        print(f"   Processing {len(df)} satellite orbits")

        # Extract orbital elements and convert to 3D positions
        # Assuming columns: semi_major_axis, eccentricity, inclination, longitude_of_ascending_node, argument_of_periapsis, mean_anomaly
        orbital_elements = df.select([
            "semi_major_axis", "eccentricity", "inclination", 
            "longitude_of_ascending_node", "argument_of_periapsis", "mean_anomaly"
        ]).to_numpy().astype(np.float32)

        # Convert orbital elements to 3D Cartesian coordinates (simplified)
        # This is a basic conversion - real implementation would use proper orbital mechanics
        coords_3d = []
        for i, orbit in enumerate(orbital_elements):
            a, e, inc, lan, aop, M = orbit
            
            # Convert to radians
            inc_rad = np.radians(inc)
            lan_rad = np.radians(lan)
            aop_rad = np.radians(aop)
            M_rad = np.radians(M)
            
            # Simplified position calculation (assuming circular orbit for simplicity)
            r = a * (1 - e)  # Approximate radius
            
            # Position in orbital plane
            x_orb = r * np.cos(M_rad)
            y_orb = r * np.sin(M_rad)
            z_orb = 0.0
            
            # Rotate to Earth-centered coordinates
            # Simplified rotation (proper implementation would use full rotation matrices)
            x = x_orb * np.cos(lan_rad) - y_orb * np.sin(lan_rad) * np.cos(inc_rad)
            y = x_orb * np.sin(lan_rad) + y_orb * np.cos(lan_rad) * np.cos(inc_rad)
            z = y_orb * np.sin(inc_rad)
            
            coords_3d.append([x, y, z])

        coords_3d = np.array(coords_3d, dtype=np.float32)

        # Create k-NN graph based on 3D spatial proximity
        print(f"   Computing {self.k_neighbors}-NN graph in 3D space...")
        
        # Use simple Euclidean distance for 3D coordinates
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean')
        nbrs.fit(coords_3d)
        distances, indices = nbrs.kneighbors(coords_3d)

        # Extract features (orbital elements)
        features = orbital_elements

        # Create edge connections based on orbital similarity
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                # Check orbital similarity (difference in semi-major axis)
                sma_diff = abs(features[i, 0] - features[j, 0])  # Semi-major axis difference
                if sma_diff <= self.orbital_similarity_threshold:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords_3d, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_satellites = len(df)
        data.k_neighbors = self.k_neighbors
        data.orbital_threshold = self.orbital_similarity_threshold
        
        data_list = [data]

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed satellite graph with {len(df)} orbits and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with satellite orbital data
        survey_data = {
            'coordinates': data.pos,  # 3D Cartesian positions
            'semi_major_axis': data.x[:, 0:1],  # Orbital semi-major axis
            'eccentricity': data.x[:, 1:2],     # Orbital eccentricity
            'inclination': data.x[:, 2:3],      # Orbital inclination
            'longitude_of_ascending_node': data.x[:, 3:4],  # RAAN
            'argument_of_periapsis': data.x[:, 4:5],        # Argument of periapsis
            'mean_anomaly': data.x[:, 5:6],     # Mean anomaly
            'num_objects': data.num_satellites,
            'survey_name': 'Satellite Orbits'
        }
        
        try:
            return SurveyTensor(
                data=survey_data,
                coordinate_system='cartesian',
                survey_name='Satellite Orbits'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_spatial_tensor(self) -> Optional["Spatial3DTensor"]:
        """Extract spatial coordinates as Spatial3DTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Use the 3D Cartesian coordinates directly
        spatial_data = {
            'coordinates': data.pos,  # Already in 3D Cartesian
            'coordinate_system': 'cartesian',
            'units': 'km'
        }
        
        try:
            return Spatial3DTensor(data=spatial_data)
        except Exception as e:
            print(f"⚠️  Failed to create Spatial3DTensor: {e}")
            return None

    def get_orbital_elements(self) -> Dict[str, torch.Tensor]:
        """Extract orbital elements as separate tensors."""
        if len(self) == 0:
            return {}
            
        data = self[0]
        
        return {
            'semi_major_axis': data.x[:, 0],
            'eccentricity': data.x[:, 1],
            'inclination': data.x[:, 2],
            'longitude_of_ascending_node': data.x[:, 3],
            'argument_of_periapsis': data.x[:, 4],
            'mean_anomaly': data.x[:, 5],
        } 