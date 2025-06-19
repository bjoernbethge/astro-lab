"""
Spectroscopy Datasets
====================

Datasets for astronomical spectroscopic data including:
- SDSS spectral datasets
- Spectroscopic classification datasets
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
    from astro_lab.tensors import SurveyTensor, SpectralTensor


class SDSSSpectralDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for SDSS spectroscopic data.

    Creates spatial graphs from spectroscopic objects with connections
    based on sky coordinates and spectral similarity.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_spectra: int = 1000,
        k_neighbors: int = 5,
        spectral_similarity_threshold: float = 0.8,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize SDSS spectral dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        max_spectra : int, default 1000
            Maximum number of spectra to include
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        spectral_similarity_threshold : float, default 0.8
            Spectral similarity threshold for connections
        """
        self.max_spectra = max_spectra
        self.k_neighbors = k_neighbors
        self.spectral_similarity_threshold = spectral_similarity_threshold

        if root is None:
            root = "data/processed/sdss_spectral_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files."""
        return ["sdss_spectra.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"sdss_spectral_graph_k{self.k_neighbors}_n{self.max_spectra}.pt"]

    def download(self):
        """Download SDSS spectral data if needed."""
        from astro_lab.data.manager import download_sdss_spectra
        
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"🔭 Downloading SDSS spectral data automatically...")
            try:
                download_sdss_spectra()
                print(f"✅ SDSS spectral data downloaded successfully")
            except Exception as e:
                print(f"❌ Failed to download SDSS spectral data: {e}")
                raise

    def process(self):
        """Process SDSS spectral data into graph format."""
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing SDSS spectral data: {raw_path.name}")

        # Load spectral catalog
        df = pl.read_parquet(raw_path)
        
        # Sample if too many spectra
        if len(df) > self.max_spectra:
            df = df.sample(self.max_spectra, seed=42)
            
        print(f"   Processing {len(df)} SDSS spectra")

        # Convert to numpy for processing
        coords = df.select(["ra", "dec"]).to_numpy()
        
        # Get k-nearest neighbors based on sky position
        print(f"   Computing {self.k_neighbors}-NN graph...")
        distances, indices = gpu_knn_graph(coords, self.k_neighbors)

        # Extract features (spectroscopic properties)
        feature_cols = ["z", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "spec_class"]
        features = df.select(feature_cols).to_numpy().astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        # Create edge connections based on spatial and spectral similarity
        edge_index = []
        redshifts = features[:, 0]  # Redshift is first feature
        spec_classes = features[:, -1]  # Spectral class is last feature
        
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self-connection
                # Check redshift similarity (crude spectral similarity proxy)
                z_diff = abs(redshifts[i] - redshifts[j])
                same_class = spec_classes[i] == spec_classes[j]
                
                # Connect if similar redshift or same spectral class
                if z_diff <= 0.1 or same_class:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create data object
        x = torch.tensor(features, dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, pos=pos)
        data.num_spectra = len(df)
        data.k_neighbors = self.k_neighbors
        data.spectral_threshold = self.spectral_similarity_threshold
        
        data_list = [data]

        # Apply pre-filter and pre-transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"✅ Processed SDSS spectral graph with {len(df)} spectra and {edge_index.shape[1]} edges")

    def to_survey_tensor(self) -> Optional["SurveyTensor"]:
        """Convert dataset to SurveyTensor format."""
        if not ASTRO_LAB_TENSORS_AVAILABLE:
            print("⚠️  AstroLab tensors not available")
            return None
            
        if len(self) == 0:
            print("⚠️  Dataset is empty")
            return None
        
        data = self[0]
        
        # Create SurveyTensor with SDSS spectral data
        survey_data = {
            'coordinates': data.pos,  # RA, Dec
            'redshifts': data.x[:, 0:1],  # Spectroscopic redshifts
            'magnitudes': data.x[:, 1:6],  # ugriz magnitudes
            'spec_classes': data.x[:, 6:7],  # Spectral classifications
            'num_objects': data.num_spectra,
            'survey_name': 'SDSS Spectroscopy'
        }
        
        try:
            return SurveyTensor(
                data=survey_data,
                coordinate_system='icrs',
                survey_name='SDSS Spectroscopy'
            )
        except Exception as e:
            print(f"⚠️  Failed to create SurveyTensor: {e}")
            return None

    def get_spectral_tensor(self) -> Optional["SpectralTensor"]:
        """Extract spectral data as SpectralTensor."""
        if not ASTRO_LAB_TENSORS_AVAILABLE or len(self) == 0:
            return None
            
        data = self[0]
        
        # Create synthetic spectral data based on redshift and magnitudes
        # In a real implementation, this would load actual spectra
        wavelengths = torch.linspace(3800, 9200, 1000)  # SDSS wavelength range (Å)
        
        # Create simple synthetic spectra based on object properties
        spectra = []
        for i in range(data.num_spectra):
            z = data.x[i, 0].item()  # Redshift
            g_mag = data.x[i, 2].item()  # g-band magnitude
            spec_class = data.x[i, 6].item()  # Spectral class
            
            # Simple synthetic spectrum (blackbody + emission lines)
            # Redshift the wavelengths
            rest_wavelengths = wavelengths / (1 + z)
            
            # Simple blackbody-like continuum
            flux = torch.exp(-((rest_wavelengths - 5500) / 2000)**2) * torch.exp(-g_mag/5)
            
            # Add emission lines based on spectral class
            if spec_class == 1:  # QSO-like
                # Add broad emission lines
                flux += 0.5 * torch.exp(-((rest_wavelengths - 6563) / 50)**2)  # H-alpha
                flux += 0.3 * torch.exp(-((rest_wavelengths - 4861) / 40)**2)  # H-beta
            elif spec_class == 2:  # Galaxy-like
                # Add narrow emission lines
                flux += 0.2 * torch.exp(-((rest_wavelengths - 6563) / 10)**2)  # H-alpha
                flux += 0.1 * torch.exp(-((rest_wavelengths - 5007) / 8)**2)   # [OIII]
            
            spectra.append(flux)
        
        spectral_data = {
            'wavelengths': wavelengths.unsqueeze(0).repeat(data.num_spectra, 1),
            'fluxes': torch.stack(spectra),
            'errors': torch.full((data.num_spectra, len(wavelengths)), 0.1),
            'redshifts': data.x[:, 0],
            'units': 'angstrom',
        }
        
        try:
            return SpectralTensor(data=spectral_data)
        except Exception as e:
            print(f"⚠️  Failed to create SpectralTensor: {e}")
            return None 