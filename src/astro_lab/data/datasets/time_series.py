"""
Time Series Datasets
===================

Datasets for astronomical time series data including:
- LINEAR lightcurve datasets
- RR Lyrae variable star datasets
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
    from astro_lab.tensors import SurveyTensor, LightcurveTensor


class LINEARLightcurveDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for LINEAR lightcurve data.

    Creates spatial graphs from variable star lightcurves with connections
    based on sky coordinates and period similarity.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_objects: int = 500,
        k_neighbors: int = 5,
        min_observations: int = 50,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize LINEAR lightcurve dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        max_objects : int, default 500
            Maximum number of objects to include
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        min_observations : int, default 50
            Minimum number of observations per object
        """
        self.max_objects = max_objects
        self.k_neighbors = k_neighbors
        self.min_observations = min_observations

        if root is None:
            root = "data/processed/linear_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["linear_lightcurves.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"linear_graph_k{self.k_neighbors}_n{self.max_objects}.pt"]

    def download(self):
        """Download LINEAR data if needed."""
        from astro_lab.data.manager import download_linear_data
        
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⭐ Downloading LINEAR lightcurve data automatically...")
            try:
                download_linear_data()
                print(f"✅ LINEAR data downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download LINEAR data: {e}")
                raise

    def process(self):
        """Process LINEAR lightcurve data into graph format."""
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing LINEAR lightcurve data: {raw_path.name}")

        # Load lightcurve catalog
        df = pl.read_parquet(raw_path)
        
        # Filter by minimum observations
        object_counts = df.group_by("object_id").count()
        valid_objects = object_counts.filter(pl.col("count") >= self.min_observations)["object_id"]
        df = df.filter(pl.col("object_id").is_in(valid_objects))
        
        # Sample objects if too many
        unique_objects = df["object_id"].unique()
        if len(unique_objects) > self.max_objects:
            sampled_objects = unique_objects.sample(self.max_objects, seed=42)
            df = df.filter(pl.col("object_id").is_in(sampled_objects))
            unique_objects = sampled_objects
            
        print(f"   Processing {len(unique_objects)} objects with lightcurves")

        # Compute lightcurve features for each object
        object_features = df.group_by("object_id").agg([
            pl.col("ra").first(),
            pl.col("dec").first(),
            pl.col("magnitude").std().alias("magnitude_std"),
            pl.col("magnitude").mean().alias("magnitude_mean"),
            pl.col("magnitude").max().alias("magnitude_max"),
            pl.col("magnitude").min().alias("magnitude_min"),
            pl.count().alias("num_observations")
        ])

        # Convert to numpy for processing
        coords = object_features.select(["ra", "dec"]).to_numpy()
        
        # Get k-nearest neighbors based on sky position
        print(f"   Computing {self.k_neighbors}-NN graph...")
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)

        # Extract features (lightcurve statistics)
        feature_cols = ["magnitude_mean", "magnitude_std", "magnitude_max", "magnitude_min", "num_observations"]
        features = object_features.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Create edge connections
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_objects = len(unique_objects)
        data.k_neighbors = self.k_neighbors
        data.min_observations = self.min_observations
        
        data_list = [data]

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed LINEAR graph with {len(unique_objects)} objects and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with LINEAR lightcurve data
        survey_data = {
            'coordinates': data.pos,  # RA, Dec
            'magnitudes': data.x[:, 0:1],  # Mean magnitudes
            'variability': data.x[:, 1:2],  # Magnitude std
            'amplitude': data.x[:, 2:3] - data.x[:, 3:4],  # max - min
            'num_observations': data.x[:, 4:5],
            'num_objects': data.num_objects,
            'survey_name': 'LINEAR'
        }
        
        try:
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='LINEAR'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_lightcurve_tensor(self) -> Optional["LightcurveTensor"]:
        """Extract lightcurve data as LightcurveTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Create synthetic lightcurve tensor from statistics
        # In a real implementation, this would load the full time series
        lightcurve_data = {
            'times': torch.arange(0, 100, 1.0).unsqueeze(0).repeat(data.num_objects, 1),  # Dummy times
            'magnitudes': data.x[:, 0:1].repeat(1, 100),  # Constant magnitude (simplified)
            'errors': torch.full((data.num_objects, 100), 0.1),  # Dummy errors
            'bands': ['V'] * data.num_objects,
            'periods': torch.full((data.num_objects,), 1.0),  # Dummy periods
        }
        
        try:
            return LightcurveTensor(data=lightcurve_data)
        except Exception as e:
            print(f"⚠️  Failed to create LightcurveTensor: {e}")
            return None


class RRLyraeDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for RR Lyrae variable star data.

    Creates spatial graphs from RR Lyrae stars with connections
    based on sky coordinates and period similarity.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_stars: int = 300,
        k_neighbors: int = 5,
        period_similarity_threshold: float = 0.1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize RR Lyrae dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        max_stars : int, default 300
            Maximum number of stars to include
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        period_similarity_threshold : float, default 0.1
            Period similarity threshold for connections (days)
        """
        self.max_stars = max_stars
        self.k_neighbors = k_neighbors
        self.period_similarity_threshold = period_similarity_threshold

        if root is None:
            root = "data/processed/rrlyrae_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["rrlyrae_real_data_cleaned.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"rrlyrae_graph_k{self.k_neighbors}_n{self.max_stars}.pt"]

    def download(self):
        """Download RR Lyrae data if needed."""
        from astro_lab.data.manager import download_rrlyrae_data
        
        raw_path = Path("data/processed") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"🌟 Downloading RR Lyrae data automatically...")
            try:
                download_rrlyrae_data()
                print(f"✅ RR Lyrae data downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download RR Lyrae data: {e}")
                raise

    def process(self):
        """Process RR Lyrae data into graph format."""
        raw_path = Path("data/processed") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing RR Lyrae data: {raw_path.name}")

        # Load catalog
        df = pl.read_parquet(raw_path)
        
        # Sample if too many stars
        if len(df) > self.max_stars:
            df = df.sample(self.max_stars, seed=42)
            
        print(f"   Processing {len(df)} RR Lyrae stars")

        # Convert to numpy for processing
        coords = df.select(["ra", "dec"]).to_numpy()
        
        # Get k-nearest neighbors based on sky position
        print(f"   Computing {self.k_neighbors}-NN graph...")
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)

        # Extract features (periods, amplitudes, colors)
        feature_cols = ["period", "amplitude_v", "mean_v", "color_gr", "metallicity"]
        features = df.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Create edge connections based on both spatial and period similarity
        edge_index = []
        periods = features[:, 0]  # Period is first feature
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                # Check period similarity
                period_diff = abs(periods[i] - periods[j])
                if period_diff <= self.period_similarity_threshold:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_stars = len(df)
        data.k_neighbors = self.k_neighbors
        data.period_threshold = self.period_similarity_threshold
        
        data_list = [data]

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed RR Lyrae graph with {len(df)} stars and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with RR Lyrae data
        survey_data = {
            'coordinates': data.pos,  # RA, Dec
            'periods': data.x[:, 0:1],  # Pulsation periods
            'amplitudes': data.x[:, 1:2],  # V-band amplitudes
            'magnitudes': data.x[:, 2:3],  # Mean V magnitudes
            'colors': data.x[:, 3:4],  # g-r colors
            'metallicity': data.x[:, 4:5],  # [Fe/H]
            'num_objects': data.num_stars,
            'survey_name': 'RR Lyrae Catalog'
        }
        
        try:
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='RR Lyrae Catalog'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_lightcurve_tensor(self) -> Optional["LightcurveTensor"]:
        """Extract lightcurve data as LightcurveTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Create synthetic RR Lyrae lightcurves based on period and amplitude
        num_points = 100
        phases = torch.linspace(0, 1, num_points)
        
        lightcurves = []
        for i in range(data.num_stars):
            period = data.x[i, 0].item()
            amplitude = data.x[i, 1].item()
            mean_mag = data.x[i, 2].item()
            
            # Simple sinusoidal RR Lyrae template
            magnitudes = mean_mag + amplitude * torch.sin(2 * np.pi * phases)
            lightcurves.append(magnitudes)
        
        lightcurve_data = {
            'times': phases.unsqueeze(0).repeat(data.num_stars, 1),
            'magnitudes': torch.stack(lightcurves),
            'errors': torch.full((data.num_stars, num_points), 0.05),
            'bands': ['V'] * data.num_stars,
            'periods': data.x[:, 0],
        }
        
        try:
            return LightcurveTensor(data=lightcurve_data)
        except Exception as e:
            print(f"⚠️  Failed to create LightcurveTensor: {e}")
            return None 