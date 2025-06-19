"""
Astronomical Datasets
====================

Datasets for stellar and galaxy data including:
- Gaia DR3 stellar catalogs
- NSA (NASA Sloan Atlas) galaxy datasets  
- TNG50 simulation data
- AstroPhot galaxy fitting datasets
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import time

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data, InMemoryDataset

from astro_lab.data.datasets.base import get_device, to_device, gpu_knn_graph, ASTRO_LAB_TENSORS_AVAILABLE

# Import tensors only for type annotations
if TYPE_CHECKING:
    from astro_lab.tensors import SurveyTensor, Spatial3DTensor, PhotometricTensor



class GaiaGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for Gaia DR3 stellar data.

    Creates spatial graphs from stellar catalogs with k-nearest neighbor
    connections based on sky coordinates.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        magnitude_limit: float = 12.0,
        k_neighbors: int = 8,
        max_distance: float = 1.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize Gaia graph dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        magnitude_limit : float, default 12.0
            Magnitude limit for catalog selection
        k_neighbors : int, default 8
            Number of nearest neighbors for graph construction
        max_distance : float, default 1.0
            Maximum distance for connections (degrees)
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.magnitude_limit = magnitude_limit
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

        # Set default root if not provided
        if root is None:
            root = "data/processed/gaia_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        if self.magnitude_limit <= 10.0:
            return ["gaia_dr3_bright_all_sky_mag10.0.parquet"]
        else:
            return ["gaia_dr3_bright_all_sky_mag12.0.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        mag_str = f"mag{self.magnitude_limit:.1f}"
        k_str = f"k{self.k_neighbors}"
        return [f"gaia_graph_{mag_str}_{k_str}.pt"]

    def download(self):
        """Download raw data if needed."""
        from astro_lab.data.manager import download_bright_all_sky
        
        raw_path = Path("data/raw/gaia") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"🌟 Downloading Gaia data automatically...")
            try:
                download_bright_all_sky(magnitude_limit=self.magnitude_limit)
                print(f"✅ Gaia data downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download Gaia data: {e}")
                raise

    def process(self):
        """Process raw data into graph format."""
        # Load raw catalog
        raw_path = Path("data/raw/gaia") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing Gaia catalog: {raw_path.name}")

        # Load catalog
        df = pl.read_parquet(raw_path)
        print(f"   Loaded {len(df)} stars")

        # Convert to numpy for processing
        coords = df.select(["ra", "dec"]).to_numpy()
        
        # Get k-nearest neighbors
        print(f"   Computing {self.k_neighbors}-NN graph...")
        start_time = time.time()
        distances, indices = gpu_knn_graph(coords, self.k_neighbors, device=get_device())
        print(f"   k-NN computation took {time.time() - start_time:.1f}s")

        # Create graph data
        data_list = []
        
        # Extract features (magnitudes and colors)
        feature_cols = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "parallax"]
        features = df.select(feature_cols).to_numpy().astype(np.float32)
        
        # Handle missing values
        features = np.nan_to_num(features, nan=99.0)
        
        # Create edge index from k-NN results
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                if distances[i][np.where(neighbors == j)[0][0]] <= np.radians(self.max_distance):
                    edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_stars = len(df)
        data.magnitude_limit = self.magnitude_limit
        data.k_neighbors = self.k_neighbors
        
        data_list.append(data)

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed Gaia graph with {len(df)} stars and {edge_index.shape[1]} edges")

    def to_survey_tensor(self, include_photometry: bool = True, include_spatial: bool = True) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with stellar data
        survey_data = {
            'coordinates': data.pos,  # RA, Dec
            'magnitudes': data.x[:, :3],  # G, BP, RP
            'parallax': data.x[:, 3:4],
            'num_objects': getattr(data, 'num_stars', data.num_nodes),
            'survey_name': 'Gaia DR3',
            'magnitude_limit': getattr(data, 'magnitude_limit', self.magnitude_limit)
        }
        
        try:
            # Import at runtime to avoid circular imports
            from astro_lab.tensors import SurveyTensor
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='Gaia DR3'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_photometric_tensor(self) -> Optional["PhotometricTensor"]:
        """Extract photometric measurements as PhotometricTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Gaia photometry: G, BP, RP bands
        photometry_data = {
            'magnitudes': data.x[:, :3],  # G, BP, RP
            'bands': ['G', 'BP', 'RP'],
            'survey': 'Gaia DR3'
        }
        
        try:
            # Import at runtime to avoid circular imports
            from astro_lab.tensors import PhotometricTensor
            return PhotometricTensor(data=photometry_data)
        except Exception as e:
            print(f"⚠️  Failed to create PhotometricTensor: {e}")
            return None

    def get_spatial_tensor(self) -> Optional["Spatial3DTensor"]:
        """Extract spatial coordinates as Spatial3DTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Convert RA/Dec to 3D Cartesian (unit sphere for stars without distance)
        ra_rad = torch.deg2rad(data.pos[:, 0])
        dec_rad = torch.deg2rad(data.pos[:, 1])
        
        # Use parallax to estimate distance (with fallback to unit sphere)
        parallax = data.x[:, 3]
        distance = torch.where(parallax > 0, 1000.0 / parallax, 1.0)  # pc, or unit sphere
        
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
            # Import at runtime to avoid circular imports
            from astro_lab.tensors import Spatial3DTensor
            return Spatial3DTensor(data=spatial_data)
        except Exception as e:
            print(f"⚠️  Failed to create Spatial3DTensor: {e}")
            return None


class NSAGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for NSA (NASA Sloan Atlas) galaxy data.

    Creates spatial graphs from galaxy catalogs with k-nearest neighbor
    connections based on sky coordinates and physical properties.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_galaxies: int = 10000,
        k_neighbors: int = 8,
        distance_threshold: float = 50.0,  # Mpc
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize NSA graph dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        max_galaxies : int, default 10000
            Maximum number of galaxies to include
        k_neighbors : int, default 8
            Number of nearest neighbors for graph construction
        distance_threshold : float, default 50.0
            Maximum distance for connections (Mpc)
        """
        self.max_galaxies = max_galaxies
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

        if root is None:
            root = "data/processed/nsa_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["nsa_catalog.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"nsa_graph_k{self.k_neighbors}_n{self.max_galaxies}.pt"]

    def download(self):
        """Download NSA catalog if needed."""
        raw_path = Path("data/processed/nsa") / self.raw_file_names[0]
        if not raw_path.exists():
            # Check for alternative NSA files
            alternative_paths = [
                Path("data/nsa_processed.parquet"),
                Path("data/datasets/nsa/catalog_sample_50.parquet"),
                Path("data/nsa_v0_1_2.fits")
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    print(f"📂 Using existing NSA data: {alt_path}")
                    # Create symlink or copy to expected location
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    import shutil
                    shutil.copy2(alt_path, raw_path)
                    return
                    
            print(f"❌ NSA catalog not found. Please ensure NSA data is available.")
            raise FileNotFoundError("NSA catalog data not available")

    def process(self):
        """Process NSA catalog into graph format."""
        raw_path = Path("data/processed/nsa") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing NSA catalog: {raw_path.name}")

        # Load catalog
        df = pl.read_parquet(raw_path)
        
        # Filter and sample
        if self.max_galaxies and len(df) > self.max_galaxies:
            df = df.sample(self.max_galaxies, seed=42)
            
        print(f"   Processing {len(df)} galaxies")

        # Convert to numpy
        coords = df.select(["ra", "dec"]).to_numpy()
        
        # Get k-nearest neighbors based on sky position
        print(f"   Computing {self.k_neighbors}-NN graph...")
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)

        # Extract features (magnitudes, sizes, redshifts)
        feature_cols = ["elpetro_mag_r", "elpetro_th50_r", "z", "elpetro_ba", "elpetro_phi"]
        features = df.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=99.0)
        
        # Create edge connections
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                # Add distance-based filtering here if needed
                edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_galaxies = len(df)
        data.k_neighbors = self.k_neighbors
        
        data_list = [data]

        # Apply transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed NSA graph with {len(df)} galaxies and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        survey_data = {
            'coordinates': data.pos,
            'magnitudes': data.x[:, :1],  # r-band
            'redshifts': data.x[:, 2:3],
            'sizes': data.x[:, 1:2],  # th50_r
            'num_objects': getattr(data, 'num_galaxies', data.num_nodes),
            'survey_name': 'NSA'
        }
        
        try:
            # Import at runtime to avoid circular imports
            from astro_lab.tensors import SurveyTensor
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='NSA'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_photometric_tensor(self) -> Optional["PhotometricTensor"]:
        """Extract photometric data."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        photometry_data = {
            'magnitudes': data.x[:, :1],  # r-band magnitude
            'bands': ['r'],
            'survey': 'NSA'
        }
        
        try:
            return PhotometricTensor(data=photometry_data)
        except Exception as e:
            return None

    def get_spatial_tensor(self) -> Optional["Spatial3DTensor"]:
        """Extract spatial coordinates."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Convert RA/Dec + redshift to 3D coordinates
        # Simple cosmological distance approximation
        c = 299792.458  # km/s
        H0 = 70.0  # km/s/Mpc
        
        ra_rad = torch.deg2rad(data.pos[:, 0])
        dec_rad = torch.deg2rad(data.pos[:, 1])
        z = data.x[:, 2]
        
        # Comoving distance approximation
        distance = c * z / H0  # Mpc
        
        # Convert to 3D Cartesian
        x = distance * torch.cos(dec_rad) * torch.cos(ra_rad)
        y = distance * torch.cos(dec_rad) * torch.sin(ra_rad)
        z_coord = distance * torch.sin(dec_rad)
        
        coords_3d = torch.stack([x, y, z_coord], dim=1)
        
        spatial_data = {
            'coordinates': coords_3d,
            'coordinate_system': 'cartesian',
            'units': 'Mpc'
        }
        
        try:
            return Spatial3DTensor(data=spatial_data)
        except Exception as e:
            return None


class TNG50GraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for TNG50 simulation data.

    Creates spatial graphs from simulation particle data with k-nearest
    neighbor connections based on 3D positions.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        snapshot_file: Optional[str] = None,
        particle_type: str = "PartType0",
        radius: float = 1.0,
        max_particles: int = 10000,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize TNG50 dataset.

        Parameters
        ----------
        root : str, optional
            Root directory
        snapshot_file : str, optional
            Path to TNG50 snapshot HDF5 file
        particle_type : str, default "PartType0"
            Type of particles to load
        radius : float, default 1.0
            Radius for spatial connections (Mpc)
        max_particles : int, default 10000
            Maximum particles to load
        """
        self.particle_type = particle_type
        self.radius = radius
        self.max_particles = max_particles
        self.snapshot_file = snapshot_file

        if root is None:
            root = "data/processed/tng50_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        if self.snapshot_file:
            return [Path(self.snapshot_file).name]
        return ["snap_099.0.hdf5"]  # Default

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        snap_name = Path(self.raw_file_names[0]).stem
        return [f"tng50_graph_{snap_name}_{self.particle_type.lower()}_r{self.radius:.1f}_n{self.max_particles}.pt"]

    def download(self):
        """Download TNG50 data."""
        # Check if snapshot file exists
        if self.snapshot_file and Path(self.snapshot_file).exists():
            return
        
        raw_path = Path("data/raw/TNG50-4/output/snapdir_099") / self.raw_file_names[0]
        if not raw_path.exists():
            print("⚠️  TNG50 data must be downloaded manually from IllustrisTNG")
            print("   Visit: https://www.tng-project.org/data/")
            raise FileNotFoundError(f"TNG50 snapshot not found: {raw_path}")

    def process(self):
        """Process TNG50 data into graph format."""
        # Use provided snapshot file or default path
        if self.snapshot_file:
            raw_path = Path(self.snapshot_file)
        else:
            raw_path = Path("data/raw/TNG50-4/output/snapdir_099") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing TNG50 snapshot: {raw_path.name}")
        print(f"   Particle type: {self.particle_type}")

        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for TNG50 data processing")

        with h5py.File(raw_path, 'r') as f:
            # Check if particle type exists
            if self.particle_type not in f:
                available_types = [key for key in f.keys() if key.startswith('PartType')]
                raise ValueError(f"Particle type {self.particle_type} not found. Available: {available_types}")
            
            # Load particle data
            coords = f[f'{self.particle_type}/Coordinates'][:]
            
            # Load features - handle missing Masses field for dark matter
            features = []
            feature_names = []
            
            # Try to load masses - not all particle types have individual masses
            if 'Masses' in f[self.particle_type]:
                masses = f[f'{self.particle_type}/Masses'][:]
                features.append(masses.reshape(-1, 1))
                feature_names.append('mass')
            else:
                # For particle types without individual masses (like dark matter),
                # use a constant mass from header or create dummy feature
                if 'Header' in f:
                    header = f['Header']
                    # TNG50 stores particle masses in header
                    if 'MassTable' in header.attrs:
                        mass_table = header.attrs['MassTable']
                        ptype_idx = int(self.particle_type.replace('PartType', ''))
                        if ptype_idx < len(mass_table) and mass_table[ptype_idx] > 0:
                            # Use mass from header for all particles
                            constant_mass = mass_table[ptype_idx]
                            masses = np.full(len(coords), constant_mass)
                            features.append(masses.reshape(-1, 1))
                            feature_names.append('mass_from_header')
                        else:
                            # Create dummy mass feature
                            masses = np.ones(len(coords))
                            features.append(masses.reshape(-1, 1))
                            feature_names.append('mass_dummy')
                    else:
                        # Fallback: dummy mass
                        masses = np.ones(len(coords))
                        features.append(masses.reshape(-1, 1))
                        feature_names.append('mass_dummy')
                else:
                    # No header available, use dummy mass
                    masses = np.ones(len(coords))
                    features.append(masses.reshape(-1, 1))
                    feature_names.append('mass_dummy')
            
            # Try to load potential
            if 'Potential' in f[self.particle_type]:
                potential = f[f'{self.particle_type}/Potential'][:]
                features.append(potential.reshape(-1, 1))
                feature_names.append('potential')
            
            # Sample if too many particles
            if len(coords) > self.max_particles:
                indices = np.random.choice(len(coords), self.max_particles, replace=False)
                coords = coords[indices]
                features = [feat[indices] for feat in features]

        print(f"   Processing {len(coords):,} particles with {len(features)} features")

        # Create spatial graph using radius-based connections
        print(f"   Creating spatial graph with radius {self.radius} Mpc...")
        
        # For efficiency with large datasets, use KDTree for neighbor search
        from sklearn.neighbors import NearestNeighbors
        
        # Find neighbors within radius
        nbrs = NearestNeighbors(radius=self.radius, algorithm='kd_tree')
        nbrs.fit(coords)
        
        # Get all neighbors within radius
        distances, indices = nbrs.radius_neighbors(coords)
        
        # Build edge list
        edge_index = []
        edge_weights = []
        total_connections = 0
        
        for i, (dists, neighs) in enumerate(zip(distances, indices)):
            for j, (dist, neigh) in enumerate(zip(dists, neighs)):
                if neigh != i:  # Skip self-connections
                    edge_index.append([i, neigh])
                    edge_weights.append(dist)
                    total_connections += 1
        
        if total_connections > 0:
            print(f"   📊 Found {total_connections:,} radius-based connections")

        if len(edge_index) == 0:
            print(f"   ⚠️  No connections found with radius {self.radius} Mpc. Using k-NN instead...")
            # Fallback to k-NN if no radius connections
            k_neighbors = min(10, len(coords) - 1)  # Ensure k < n_samples
            nbrs_knn = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree')  # +1 for self
            nbrs_knn.fit(coords)
            distances, indices = nbrs_knn.kneighbors(coords)
            
            edge_index = []
            edge_weights = []
            for i, (dists, neighs) in enumerate(zip(distances, indices)):
                for j, (dist, neigh) in enumerate(zip(dists[1:], neighs[1:])):  # Skip self
                    edge_index.append([i, neigh])
                    edge_weights.append(dist)
            
            print(f"   📊 Created k-NN graph with k={k_neighbors}")

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        # Combine features
        x = torch.tensor(np.concatenate(features, axis=1), dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            pos=pos
        )
        data.num_particles = len(coords)
        data.particle_type = self.particle_type
        data.feature_names = feature_names
        data.snapshot_file = raw_path.name
        
        data_list = [data]

        # Apply transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed TNG50 graph: {len(coords):,} particles, {edge_index.shape[1]:,} edges")


class AstroPhotDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for AstroPhot galaxy fitting.

    Creates graph datasets from galaxy catalogs with image cutouts
    and morphological features.
    """

    def __init__(
        self,
        catalog_path: Union[str, Path],
        root: Optional[str] = None,
        cutout_size: int = 128,
        pixel_scale: float = 0.262,
        magnitude_range: Tuple[float, float] = (10.0, 18.0),
        k_neighbors: int = 5,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize AstroPhot dataset.

        Parameters
        ----------
        catalog_path : str or Path
            Path to galaxy catalog
        root : str, optional
            Root directory
        cutout_size : int, default 128
            Size of image cutouts in pixels
        pixel_scale : float, default 0.262
            Pixel scale in arcsec/pixel
        magnitude_range : tuple, default (10.0, 18.0)
            Magnitude range for galaxy selection
        k_neighbors : int, default 5
            Number of nearest neighbors
        """
        self.catalog_path = Path(catalog_path)
        self.cutout_size = cutout_size
        self.pixel_scale = pixel_scale
        self.magnitude_range = magnitude_range
        self.k_neighbors = k_neighbors

        if root is None:
            root = f"data/processed/astrophot_{self.catalog_path.stem}"

        super().__init__(root, transform, pre_transform, pre_filter)

        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return [self.catalog_path.name]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        mag_str = f"mag{self.magnitude_range[0]:.1f}-{self.magnitude_range[1]:.1f}"
        return [f"astrophot_dataset_{mag_str}_k{self.k_neighbors}.pt"]

    def download(self):
        """Download not needed - using local catalog."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")

    def process(self):
        """Process galaxy catalog for AstroPhot fitting."""
        print(f"🔄 Processing AstroPhot catalog: {self.catalog_path.name}")

        # Load catalog
        if self.catalog_path.suffix == ".parquet":
            df = pl.read_parquet(self.catalog_path)
        else:
            raise ValueError(f"Unsupported format: {self.catalog_path.suffix}")

        # Filter by magnitude
        df = df.filter(
            (pl.col("r_mag") >= self.magnitude_range[0]) &
            (pl.col("r_mag") <= self.magnitude_range[1])
        )
        
        print(f"   Processing {len(df)} galaxies in magnitude range {self.magnitude_range}")

        # Extract features for graph
        coords = df.select(["ra", "dec"]).to_numpy()
        features = df.select(["r_mag", "g_r_color", "r_i_color", "petro_r50"]).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=99.0)

        # Create k-NN graph
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)
        
        edge_index = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:
                edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_galaxies = len(df)
        data.cutout_size = self.cutout_size
        data.pixel_scale = self.pixel_scale
        
        data_list = [data]

        # Apply transforms
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed AstroPhot dataset with {len(df)} galaxies") 