"""
AstroLab Core Data Module - Clean Polars-First Implementation with Full Tensor Integration
==========================================================================================

Eliminiert Wrapper-Chaos und bietet direkte Polars→PyTorch→AstroTensor Pipeline.
Ersetzt die komplexe Manager/DataModule/Transform Kette durch einfache,
performante Klassen mit nativer SurveyTensor Integration.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import lightning as L
import numpy as np
import polars as pl
import torch
import torch_geometric.transforms as T
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

# 🌟 TENSOR INTEGRATION - Import all tensor types
try:
    from astro_lab.tensors import (
        LightcurveTensor,
        PhotometricTensor,
        SimulationTensor,
        Spatial3DTensor,
        SpectralTensor,
        SurveyTensor,
    )

    TENSOR_INTEGRATION_AVAILABLE = True
except ImportError:
    TENSOR_INTEGRATION_AVAILABLE = False
    SurveyTensor = None

# Survey configurations - DRY principle with TENSOR METADATA
SURVEY_CONFIGS = {
    "gaia": {
        "name": "Gaia DR3",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
        "extra_cols": ["parallax", "pmra", "pmdec"],
        "color_pairs": [
            ("phot_g_mean_mag", "phot_bp_mean_mag"),
            ("phot_bp_mean_mag", "phot_rp_mean_mag"),
        ],
        "default_limit": 12.0,
        "url": "gaia",
        # 🌟 TENSOR METADATA
        "filter_system": "gaia",
        "data_release": "DR3",
        "coordinate_system": "icrs",
        "photometric_bands": ["G", "BP", "RP"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "parallax": 2,
            "pmra": 3,
            "pmdec": 4,
            "phot_g_mean_mag": 5,
            "phot_bp_mean_mag": 6,
            "phot_rp_mean_mag": 7,
        },
    },
    "sdss": {
        "name": "SDSS DR17",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": [
            "modelMag_u",
            "modelMag_g",
            "modelMag_r",
            "modelMag_i",
            "modelMag_z",
        ],
        "extra_cols": ["petroRad_r", "fracDeV_r"],
        "color_pairs": [("modelMag_g", "modelMag_r"), ("modelMag_r", "modelMag_i")],
        "default_limit": 20.0,
        "url": "sdss",
        # 🌟 TENSOR METADATA
        "filter_system": "sdss",
        "data_release": "DR17",
        "coordinate_system": "icrs",
        "photometric_bands": ["u", "g", "r", "i", "z"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "z": 2,
            "modelMag_u": 3,
            "modelMag_g": 4,
            "modelMag_r": 5,
            "modelMag_i": 6,
            "modelMag_z": 7,
            "petroRad_r": 8,
            "fracDeV_r": 9,
        },
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": ["mag_g", "mag_r", "mag_i"],
        "extra_cols": ["sersic_n", "sersic_ba", "mass"],
        "color_pairs": [("mag_g", "mag_r"), ("mag_r", "mag_i")],
        "default_limit": 18.0,
        "url": "nsa",
        # 🌟 TENSOR METADATA
        "filter_system": "sdss",
        "data_release": "v1_0_1",
        "coordinate_system": "icrs",
        "photometric_bands": ["g", "r", "i"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "z": 2,
            "mag_g": 3,
            "mag_r": 4,
            "mag_i": 5,
            "sersic_n": 6,
            "sersic_ba": 7,
            "mass": 8,
        },
    },
    "linear": {
        "name": "LINEAR Lightcurves",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean", "mag_amp"],
        "extra_cols": ["period", "period_error"],
        "color_pairs": [],
        "default_limit": 16.0,
        "url": "linear",
        # 🌟 TENSOR METADATA
        "filter_system": "johnson",
        "data_release": "final",
        "coordinate_system": "icrs",
        "photometric_bands": ["V"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "mag_mean": 2,
            "mag_amp": 3,
            "period": 4,
            "period_error": 5,
        },
    },
}


def _polars_to_survey_tensor(
    df: pl.DataFrame, survey: str, survey_metadata: Optional[Dict[str, Any]] = None
) -> "SurveyTensor":
    """
    Convert Polars DataFrame to SurveyTensor with survey-specific metadata.

    Args:
        df: Polars DataFrame with astronomical data
        survey: Survey name ('gaia', 'sdss', etc.)
        survey_metadata: Additional metadata

    Returns:
        SurveyTensor with properly configured metadata
    """
    if not TENSOR_INTEGRATION_AVAILABLE:
        raise ImportError("Tensor integration not available. Install astro_lab.tensors")

    if survey not in SURVEY_CONFIGS:
        raise ValueError(f"Survey {survey} not supported")

    config = SURVEY_CONFIGS[survey]

    # Convert to PyTorch tensor
    tensor_data = torch.tensor(df.to_numpy(), dtype=torch.float32)

    # Prepare metadata
    metadata = {
        "survey_name": survey,
        "data_release": config["data_release"],
        "filter_system": config["filter_system"],
        "column_mapping": config["tensor_column_mapping"],
        "coordinate_system": config["coordinate_system"],
        "photometric_bands": config["photometric_bands"],
        "n_objects": len(df),
        "survey_metadata": {
            "original_columns": df.columns,
            "data_shape": df.shape,
            "coord_cols": config["coord_cols"],
            "mag_cols": config["mag_cols"],
            "extra_cols": config["extra_cols"],
        },
    }

    if survey_metadata:
        metadata["survey_metadata"].update(survey_metadata)

    return SurveyTensor(data=tensor_data, **metadata)


class AstroDataset(InMemoryDataset):
    """
    Clean, unified astronomical dataset using Polars with native SurveyTensor support.

    Replaces all the wrapper classes with one flexible implementation.
    Enhanced with automatic tensor conversion capabilities.
    """

    def __init__(
        self,
        survey: str,
        data_path: Optional[Union[str, Path]] = None,
        root: Optional[str] = None,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        force_reload: bool = False,
        transform=None,
        return_tensor: bool = False,  # 🌟 NEW: Auto-convert to SurveyTensor
        **kwargs,
    ):
        """
        Initialize unified astronomical dataset.

        Args:
            survey: Survey type ('gaia', 'sdss', 'nsa', 'linear')
            data_path: Path to data file (optional, will auto-download)
            k_neighbors: Number of nearest neighbors for graph
            max_samples: Limit number of samples
            force_reload: Force reprocessing
            return_tensor: Return SurveyTensor instead of PyG Data objects
        """
        if survey not in SURVEY_CONFIGS:
            raise ValueError(
                f"Survey {survey} not supported. Available: {list(SURVEY_CONFIGS.keys())}"
            )

        self.survey = survey
        self.config = SURVEY_CONFIGS[survey]
        self.data_path = Path(data_path) if data_path else None
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples
        self.return_tensor = return_tensor

        # Set default root
        if root is None:
            tensor_suffix = "_tensor" if return_tensor else ""
            root = f"data/processed/{survey}_k{k_neighbors}{tensor_suffix}"

        super().__init__(root, transform, force_reload=force_reload)

        # Load if available
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names."""
        if self.data_path:
            return [self.data_path.name]
        return [f"{self.survey}_catalog.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        suffix = f"_n{self.max_samples}" if self.max_samples else ""
        tensor_suffix = "_tensor" if self.return_tensor else ""
        return [f"{self.survey}_graph_k{self.k_neighbors}{suffix}{tensor_suffix}.pt"]

    def download(self):
        """Download or prepare raw data."""
        if self.data_path and self.data_path.exists():
            return

        # Auto-generate demo data if no path provided
        if not self.data_path:
            print(f"🔄 Generating demo {self.config['name']} data...")
            self._generate_demo_data()

    def _load_polars_dataframe(self) -> pl.DataFrame:
        """Load data as Polars DataFrame - used for tensor conversion."""
        if not hasattr(self, "_cached_df"):
            self._generate_demo_data()
            # Cache the DataFrame for reuse
            self._cached_df = self._df_cache
        return self._cached_df

    def _generate_demo_data(self):
        """Generate realistic demo data for testing."""
        np.random.seed(42)
        n_objects = self.max_samples or 5000

        # Generate coordinates
        if self.survey == "gaia":
            # Stellar data - concentrated around galactic plane
            ra = np.random.uniform(0, 360, n_objects)
            dec = np.random.normal(0, 30, n_objects)
            dec = np.clip(dec, -90.0, 90.0)

            # Stellar magnitudes
            g_mag = np.random.gamma(2, 2, n_objects) + 8
            bp_mag = g_mag + np.random.normal(0.2, 0.3, n_objects)
            rp_mag = g_mag + np.random.normal(0.5, 0.4, n_objects)

            # Astrometric data
            parallax = np.random.exponential(2, n_objects)
            pmra = np.random.normal(0, 10, n_objects)
            pmdec = np.random.normal(0, 10, n_objects)

            df = pl.DataFrame(
                {
                    "ra": ra.astype(np.float32),
                    "dec": dec.astype(np.float32),
                    "phot_g_mean_mag": g_mag.astype(np.float32),
                    "phot_bp_mean_mag": bp_mag.astype(np.float32),
                    "phot_rp_mean_mag": rp_mag.astype(np.float32),
                    "parallax": parallax.astype(np.float32),
                    "pmra": pmra.astype(np.float32),
                    "pmdec": pmdec.astype(np.float32),
                }
            )

        elif self.survey in ["sdss", "nsa"]:
            # Galaxy data
            ra = np.random.uniform(0, 360, n_objects)
            dec = np.random.uniform(-30, 60, n_objects)
            z = np.random.gamma(2, 0.3, n_objects)

            # Galaxy magnitudes
            r_mag = 16 + 3 * z + np.random.normal(0, 1, n_objects)
            g_mag = r_mag + np.random.normal(0.5, 0.3, n_objects)
            i_mag = r_mag - np.random.normal(0.3, 0.2, n_objects)

            if self.survey == "sdss":
                u_mag = g_mag + np.random.normal(1.0, 0.5, n_objects)
                z_mag = i_mag - np.random.normal(0.2, 0.2, n_objects)

                df = pl.DataFrame(
                    {
                        "ra": ra.astype(np.float32),
                        "dec": dec.astype(np.float32),
                        "z": z.astype(np.float32),
                        "modelMag_u": u_mag.astype(np.float32),
                        "modelMag_g": g_mag.astype(np.float32),
                        "modelMag_r": r_mag.astype(np.float32),
                        "modelMag_i": i_mag.astype(np.float32),
                        "modelMag_z": z_mag.astype(np.float32),
                        "petroRad_r": np.random.lognormal(1, 0.5, n_objects).astype(
                            np.float32
                        ),
                        "fracDeV_r": np.random.beta(2, 5, n_objects).astype(np.float32),
                    }
                )
            else:  # NSA
                mass = 9 + 2 * np.random.random(n_objects)
                sersic_n = np.random.gamma(2, 1, n_objects)
                sersic_ba = np.random.beta(5, 2, n_objects)

                df = pl.DataFrame(
                    {
                        "ra": ra.astype(np.float32),
                        "dec": dec.astype(np.float32),
                        "z": z.astype(np.float32),
                        "mag_g": g_mag.astype(np.float32),
                        "mag_r": r_mag.astype(np.float32),
                        "mag_i": i_mag.astype(np.float32),
                        "mass": mass.astype(np.float32),
                        "sersic_n": sersic_n.astype(np.float32),
                        "sersic_ba": sersic_ba.astype(np.float32),
                    }
                )

        elif self.survey == "linear":
            # Variable star data
            ra = np.random.uniform(0, 360, n_objects)
            dec = np.random.uniform(-30, 60, n_objects)

            # Periods (log-normal distribution)
            period = np.random.lognormal(0, 1, n_objects)
            period_error = period * np.random.uniform(0.01, 0.1, n_objects)

            # Lightcurve properties
            mag_mean = np.random.uniform(12, 18, n_objects)
            mag_amp = np.random.exponential(0.5, n_objects)

            df = pl.DataFrame(
                {
                    "ra": ra.astype(np.float32),
                    "dec": dec.astype(np.float32),
                    "period": period.astype(np.float32),
                    "period_error": period_error.astype(np.float32),
                    "mag_mean": mag_mean.astype(np.float32),
                    "mag_amp": mag_amp.astype(np.float32),
                }
            )

        # Add colors using Polars expressions
        color_exprs = []
        for mag1, mag2 in self.config["color_pairs"]:
            if mag1 in df.columns and mag2 in df.columns:
                # Create more specific color names to avoid duplicates
                mag1_short = (
                    mag1.replace("phot_", "")
                    .replace("_mean_mag", "")
                    .replace("modelMag_", "")
                )
                mag2_short = (
                    mag2.replace("phot_", "")
                    .replace("_mean_mag", "")
                    .replace("modelMag_", "")
                )
                color_name = f"{mag1_short}_{mag2_short}_color"
                color_exprs.append((pl.col(mag1) - pl.col(mag2)).alias(color_name))

        if color_exprs:
            df = df.with_columns(color_exprs)

        # Cache DataFrame for tensor conversion
        self._df_cache = df

        # Save to raw directory
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(raw_path)

        print(f"✅ Generated {len(df)} {self.config['name']} objects")

    def process(self):
        """Process raw data into graph format using Polars."""
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing {self.config['name']} data...")
        start_time = time.time()

        # Load with Polars (fastest)
        df = pl.read_parquet(raw_path)

        # Sample if requested
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(self.max_samples, seed=42)
            print(f"   Sampled {self.max_samples} from {len(df)} objects")

        # Extract coordinates for k-NN
        coord_cols = self.config["coord_cols"]
        coords = df.select(coord_cols).to_numpy()

        # Build k-NN graph
        print(f"   Computing {self.k_neighbors}-NN graph...")
        if len(coord_cols) == 2:  # RA, Dec only
            # Use haversine distance for sky coordinates
            coords_rad = np.radians(coords)
            nbrs = NearestNeighbors(
                n_neighbors=self.k_neighbors + 1, metric="haversine"
            )
            nbrs.fit(coords_rad)
            distances, indices = nbrs.kneighbors(coords_rad)
        else:  # Include redshift/distance
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
            nbrs.fit(coords)
            distances, indices = nbrs.kneighbors(coords)

        # Build edge index
        edges = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Prepare features using Polars
        feature_cols = []

        # Add magnitudes
        available_mags = [col for col in self.config["mag_cols"] if col in df.columns]
        feature_cols.extend(available_mags)

        # Add colors (only new color columns, not existing magnitude columns)
        color_cols = [col for col in df.columns if col.endswith("_color")]
        feature_cols.extend(color_cols)

        # Add extra features
        available_extra = [
            col for col in self.config["extra_cols"] if col in df.columns
        ]
        feature_cols.extend(available_extra)

        # Remove duplicates while preserving order
        feature_cols = list(dict.fromkeys(feature_cols))

        # Convert to PyTorch tensors using Polars (fastest path)
        features = df.select(feature_cols).to_torch(dtype=pl.Float32)
        positions = df.select(coord_cols).to_torch(dtype=pl.Float32)

        # Handle NaN values
        features = torch.nan_to_num(features, nan=0.0)
        positions = torch.nan_to_num(positions, nan=0.0)

        # Create graph data
        data = Data(x=features, edge_index=edge_index, pos=positions, num_nodes=len(df))

        # Add metadata
        data.survey_name = self.config["name"]
        data.feature_names = feature_cols
        data.coord_names = coord_cols
        data.k_neighbors = self.k_neighbors

        # Apply transforms
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save
        self.save([data], self.processed_paths[0])

        elapsed = time.time() - start_time
        print(
            f"✅ Processed {len(df)} objects, {edge_index.shape[1]} edges in {elapsed:.1f}s"
        )

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if len(self) == 0:
            return {"error": "Dataset not loaded"}

        data = self[0]
        return {
            "survey": self.survey,
            "survey_name": data.survey_name,
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "num_features": data.x.shape[1],
            "feature_names": data.feature_names,
            "coordinate_names": data.coord_names,
            "k_neighbors": data.k_neighbors,
            "avg_degree": data.edge_index.shape[1] / data.num_nodes,
        }


class AstroDataModule(L.LightningDataModule):
    """
    Clean Lightning DataModule for astronomical data.

    Eliminates complex setup logic with simple, direct approach.
    """

    def __init__(
        self,
        survey: str,
        data_path: Optional[str] = None,
        batch_size: int = 1,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,
        force_reload: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create dataset
        self.dataset = AstroDataset(
            survey=survey,
            data_path=data_path,
            k_neighbors=k_neighbors,
            max_samples=max_samples,
            force_reload=force_reload,
            **kwargs,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup with automatic train/val/test splits."""
        if len(self.dataset) == 0:
            return

        data = self.dataset[0]
        num_nodes = data.num_nodes

        # Create random splits
        indices = torch.randperm(num_nodes)

        train_size = int(self.hparams.train_ratio * num_nodes)
        val_size = int(self.hparams.val_ratio * num_nodes)

        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size : train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size :]] = True

        print(
            f"📊 Split {data.survey_name}: "
            f"Train={data.train_mask.sum()}, "
            f"Val={data.val_mask.sum()}, "
            f"Test={data.test_mask.sum()}"
        )

    def train_dataloader(self):
        return DataLoader([self.dataset[0]], batch_size=1, num_workers=0)

    def val_dataloader(self):
        return DataLoader([self.dataset[0]], batch_size=1, num_workers=0)

    def test_dataloader(self):
        return DataLoader([self.dataset[0]], batch_size=1, num_workers=0)


# Factory function - replaces all the individual create_* functions
def create_astro_dataloader(survey: str, batch_size: int = 1, **kwargs) -> DataLoader:
    """
    Universal factory for astronomical data loaders.

    Replaces create_gaia_dataloader, create_sdss_dataloader, etc.
    """
    dataset = AstroDataset(survey=survey, **kwargs)
    return DataLoader(dataset, batch_size=batch_size)


def create_astro_datamodule(survey: str, **kwargs) -> AstroDataModule:
    """
    Universal factory for astronomical data modules.

    Replaces GaiaDataModule, ExoplanetDataModule, etc.
    """
    return AstroDataModule(survey=survey, **kwargs)


# 🌟 ENHANCED LOAD FUNCTIONS with automatic tensor conversion


def load_gaia_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load Gaia DR3 stellar catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with Gaia data or AstroDataset for legacy use
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        # Direct tensor loading - bypass PyG overhead for pure tensor usage
        dataset = AstroDataset(
            survey="gaia", max_samples=max_samples, return_tensor=False, **kwargs
        )
        dataset.download()  # Ensure data is generated

        # Get DataFrame and convert to tensor
        df = dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "gaia", {"max_samples": max_samples})

    return AstroDataset(
        survey="gaia",
        k_neighbors=8,
        max_samples=max_samples,
        return_tensor=return_tensor,
        **kwargs,
    )


def load_sdss_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load SDSS DR17 galaxy catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with SDSS data or AstroDataset for legacy use
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        dataset = AstroDataset(
            survey="sdss", max_samples=max_samples, return_tensor=False, **kwargs
        )
        dataset.download()
        df = dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "sdss", {"max_samples": max_samples})

    return AstroDataset(
        survey="sdss",
        k_neighbors=5,
        max_samples=max_samples,
        return_tensor=return_tensor,
        **kwargs,
    )


def load_nsa_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load NASA Sloan Atlas galaxy catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with NSA data or AstroDataset for legacy use
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        dataset = AstroDataset(
            survey="nsa", max_samples=max_samples, return_tensor=False, **kwargs
        )
        dataset.download()
        df = dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "nsa", {"max_samples": max_samples})

    return AstroDataset(
        survey="nsa",
        k_neighbors=5,
        max_samples=max_samples,
        return_tensor=return_tensor,
        **kwargs,
    )


def load_lightcurve_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "LightcurveTensor"]:
    """
    Load LINEAR lightcurve data.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return LightcurveTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        LightcurveTensor with lightcurve data or AstroDataset for legacy use
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        dataset = AstroDataset(
            survey="linear", max_samples=max_samples, return_tensor=False, **kwargs
        )
        dataset.download()
        df = dataset._load_polars_dataframe()

        # Convert to LightcurveTensor for time series data
        n_samples = len(df)
        times = torch.arange(n_samples, dtype=torch.float32)  # Demo time array
        magnitudes = torch.tensor(df["mag_mean"].to_numpy(), dtype=torch.float32)

        return LightcurveTensor(
            data=torch.stack([times, magnitudes], dim=1),
            bands=["V"],
            time_format="sequential",
            coordinate_system="icrs",
            survey_name="linear",
        )

    return AstroDataset(
        survey="linear",
        k_neighbors=8,
        max_samples=max_samples,
        return_tensor=return_tensor,
        **kwargs,
    )
