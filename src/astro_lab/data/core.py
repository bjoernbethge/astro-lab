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

# PyTorch Geometric integration
try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

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
) -> Any:
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

        # Set default root using new config system
        if root is None:
            from .config import data_config

            # Ensure survey directories exist before using them
            data_config.ensure_survey_directories(survey)
            root = str(data_config.get_survey_processed_dir(survey))

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

        # Create labels for classification
        if self.survey == "gaia" and "bp_rp" in df.columns:
            # Stellar classification based on B-R color
            bp_rp = df["bp_rp"].to_numpy()
            bp_rp = np.nan_to_num(bp_rp, nan=0.0)
            labels = (
                np.digitize(
                    bp_rp, bins=np.array([-0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0])
                )
                - 1
            )
            labels = np.clip(labels, 0, 7)
            y = torch.tensor(labels, dtype=torch.long)
        else:
            # Default: random labels for demo
            y = torch.randint(0, 8, (len(df),), dtype=torch.long)

        # Create graph data
        data = Data(
            x=features, edge_index=edge_index, pos=positions, y=y, num_nodes=len(df)
        )

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
        num_workers: int = 4,  # Better default for performance
        force_reload: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store attributes for easy access
        self.num_workers = num_workers
        self.batch_size = batch_size

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
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


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


def detect_survey_type(dataset_name: str, df: pl.DataFrame) -> str:
    """
    Detect survey type from filename and columns.

    Args:
        dataset_name: Name of the dataset file
        df: Polars DataFrame with astronomical data

    Returns:
        Survey type string
    """
    name_lower = dataset_name.lower()
    columns = [col.lower() for col in df.columns]

    if "nsa" in name_lower or any("petromag" in col for col in columns):
        return "nsa"
    elif "gaia" in name_lower or "phot_g_mean_mag" in columns:
        return "gaia"
    elif "sdss" in name_lower or any("modelmag" in col for col in columns):
        return "sdss"
    else:
        return "generic"


def create_graph_from_dataframe(
    df: pl.DataFrame,
    survey_type: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    output_path: Optional[Path] = None,
    **kwargs,
) -> Optional[Data]:
    """
    Create PyTorch Geometric graph from Polars DataFrame.

    Args:
        df: Input DataFrame
        survey_type: Type of survey ('nsa', 'gaia', 'sdss', 'generic')
        k_neighbors: Number of neighbors for graph construction
        distance_threshold: Distance threshold for edges
        output_path: Optional path to save graph
        **kwargs: Additional arguments

    Returns:
        PyTorch Geometric Data object or None if PyG not available
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("⚠️  PyTorch Geometric not available for graph creation")
        return None

    try:
        if survey_type == "nsa":
            return _create_nsa_graph(df, k_neighbors, distance_threshold, **kwargs)
        elif survey_type == "gaia":
            return _create_gaia_graph(df, k_neighbors, distance_threshold, **kwargs)
        elif survey_type == "sdss":
            return _create_sdss_graph(df, k_neighbors, distance_threshold, **kwargs)
        else:
            return _create_generic_graph(df, k_neighbors, distance_threshold, **kwargs)

    except Exception as e:
        print(f"❌ Error creating {survey_type} graph: {e}")
        return None


def create_graph_datasets_from_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_path: Path,
    dataset_name: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    **kwargs,
) -> Dict[str, Optional[Data]]:
    """
    Create graph datasets from train/val/test splits.

    Args:
        train_df, val_df, test_df: Split DataFrames
        output_path: Output directory
        dataset_name: Name of the dataset
        k_neighbors: Number of neighbors for graph construction
        distance_threshold: Distance threshold for edges
        **kwargs: Additional arguments

    Returns:
        Dictionary with 'train', 'val', 'test' graph data objects
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("⚠️  PyTorch Geometric not available - skipping graph creation")
        return {"train": None, "val": None, "test": None}

    print("\n🔗 Creating PyTorch Geometric Graphs (.pt) - Standard for GNNs")

    # Detect survey type
    survey_type = detect_survey_type(dataset_name, train_df)
    print(f"📊 Detected survey type: {survey_type}")

    results = {}

    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"   🔄 Creating {split_name} graph...")

        graph_data = create_graph_from_dataframe(
            df, survey_type, k_neighbors, distance_threshold, **kwargs
        )

        if graph_data is not None:
            # Save graph if output path provided
            if output_path:
                graph_dir = output_path / f"graphs_{split_name}"
                graph_dir.mkdir(exist_ok=True)

                graph_file = graph_dir / f"{dataset_name}_{split_name}.pt"
                torch.save(graph_data, graph_file)
                print(f"   💾 Saved to: {graph_file}")

            print(
                f"   📊 {split_name.title()}: {graph_data.num_nodes:,} nodes, {graph_data.num_edges:,} edges"
            )

        results[split_name] = graph_data

    print("✅ Graph datasets created successfully!")
    return results


def _create_nsa_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create NSA galaxy graph using SurveyTensor spatial integration."""
    # 🌟 Create SurveyTensor first
    survey_tensor = _polars_to_survey_tensor(df, "nsa")

    # 🌟 Get spatial tensor with 3D coordinates
    try:
        spatial_tensor = survey_tensor.get_spatial_tensor(include_distances=True)
        coords_3d = spatial_tensor.cartesian.numpy()  # [N, 3] Cartesian coordinates

        print(
            f"✅ Created NSA spatial tensor: {spatial_tensor.coordinate_system}, {coords_3d.shape}"
        )
    except Exception as e:
        print(f"⚠️ NSA spatial tensor failed, falling back to manual coordinates: {e}")
        # Fallback to manual 3D calculation from z (redshift)
        coords = df.select(["ra", "dec"]).to_numpy()
        z_values = (
            df.select("z").to_numpy().flatten()
            if "z" in df.columns
            else np.ones(len(coords)) * 0.1
        )

        # Convert to 3D using simple cosmology (c*z/H0 approximation)
        c_over_H0 = 3000.0  # Mpc, rough approximation
        distances = c_over_H0 * z_values

        # Convert spherical to Cartesian
        ra_rad = np.radians(coords[:, 0])
        dec_rad = np.radians(coords[:, 1])

        x = distances * np.cos(dec_rad) * np.cos(ra_rad)
        y = distances * np.cos(dec_rad) * np.sin(ra_rad)
        z = distances * np.sin(dec_rad)

        coords_3d = np.column_stack([x, y, z])

    # Create feature matrix with NSA-specific columns
    feature_cols = ["ra", "dec", "z"]

    # Add NSA photometry and morphology
    nsa_cols = [
        "PETROMAG_R",
        "PETROMAG_G",
        "PETROMAG_I",
        "MASS",
        "ELPETRO_MASS",
        "SERSIC_MASS",
    ]
    available_cols = feature_cols + [col for col in nsa_cols if col in df.columns]

    features = df.select(available_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0)

    # Add 3D coordinates to features
    features = np.column_stack([features, coords_3d])

    # Create k-nearest neighbor graph using 3D coordinates
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nbrs.fit(coords_3d)

    distances_3d, indices = nbrs.kneighbors(coords_3d)

    # Convert to edge list
    edge_list = []
    edge_weights = []

    for i, (dist_row, idx_row) in enumerate(zip(distances_3d, indices)):
        for j, (dist, idx) in enumerate(zip(dist_row[1:], idx_row[1:])):  # Skip self
            if dist <= distance_threshold:
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(features, dtype=torch.float)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords_3d, dtype=torch.float),
        num_nodes=len(features),
    )


def _create_gaia_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create Gaia stellar graph using SurveyTensor spatial integration."""
    # 🌟 Create SurveyTensor first
    survey_tensor = _polars_to_survey_tensor(df, "gaia")

    # 🌟 Get spatial tensor with 3D coordinates
    try:
        spatial_tensor = survey_tensor.get_spatial_tensor(include_distances=True)
        coords_3d = spatial_tensor.cartesian.numpy()  # [N, 3] Cartesian coordinates

        print(
            f"✅ Created spatial tensor: {spatial_tensor.coordinate_system}, {coords_3d.shape}"
        )
    except Exception as e:
        print(f"⚠️ Spatial tensor failed, falling back to manual coordinates: {e}")
        # Fallback to manual calculation
        coords = df.select(["ra", "dec"]).to_numpy()
        coords_3d = coords  # Just use 2D for fallback

    # Create feature matrix
    feature_cols = ["ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax"]
    available_cols = [col for col in feature_cols if col in df.columns]

    features = df.select(available_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0)

    # Add 3D coordinates to features if available
    if coords_3d.shape[1] == 3:
        features = np.column_stack([features, coords_3d])

    # Create k-nearest neighbor graph using 3D coordinates
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")

    if coords_3d.shape[1] == 3:
        # Use 3D Cartesian coordinates
        nbrs.fit(coords_3d)
        distances, indices = nbrs.kneighbors(coords_3d)
        distance_threshold_3d = distance_threshold  # Already in proper units
    else:
        # Fallback to sky coordinates
        coords_sky = df.select(["ra", "dec"]).to_numpy()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords_sky))
        distances, indices = nbrs.kneighbors(np.radians(coords_sky))
        distance_threshold_3d = np.radians(distance_threshold / 3600.0)

    # Convert to edge list
    edge_list = []
    edge_weights = []

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for j, (dist, idx) in enumerate(zip(dist_row[1:], idx_row[1:])):  # Skip self
            if dist <= distance_threshold_3d:
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(features, dtype=torch.float)

    # Create labels for Gaia stellar classification based on color
    if "bp_rp" in df.columns:
        bp_rp = df["bp_rp"].to_numpy()
        bp_rp = np.nan_to_num(bp_rp, nan=0.0)

        # Stellar classification based on B-R color:
        # 0: Very blue (hot stars), 1: Blue, 2: Blue-white, 3: White
        # 4: Yellow-white, 5: Yellow, 6: Orange, 7: Red (cool stars)
        labels = (
            np.digitize(bp_rp, bins=np.array([-0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0]))
            - 1
        )
        labels = np.clip(labels.astype(int), 0, 7)  # Ensure labels are in range [0, 7]
        y = torch.tensor(labels, dtype=torch.long)
    else:
        # Fallback: random labels for demo
        y = torch.randint(0, 8, (len(features),), dtype=torch.long)

    # Use proper 3D positions if available
    pos_tensor = (
        torch.tensor(coords_3d, dtype=torch.float) if coords_3d.shape[1] == 3 else None
    )

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        pos=pos_tensor,
        num_nodes=len(features),
    )


def _create_sdss_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create SDSS galaxy graph using SurveyTensor spatial integration."""
    # 🌟 Create SurveyTensor first
    survey_tensor = _polars_to_survey_tensor(df, "sdss")

    # 🌟 Get spatial tensor with 3D coordinates
    try:
        spatial_tensor = survey_tensor.get_spatial_tensor(include_distances=True)
        coords_3d = spatial_tensor.cartesian.numpy()  # [N, 3] Cartesian coordinates
        use_3d = True

        print(
            f"✅ Created SDSS spatial tensor: {spatial_tensor.coordinate_system}, {coords_3d.shape}"
        )
    except Exception as e:
        print(f"⚠️ SDSS spatial tensor failed, using sky coordinates: {e}")
        # Use sky coordinates for SDSS
        coords_3d = df.select(["ra", "dec"]).to_numpy()
        use_3d = False

    # Create feature matrix
    feature_cols = ["ra", "dec"]
    if "z" in df.columns:
        feature_cols.append("z")

    # Add SDSS photometry
    sdss_mags = ["modelmag_u", "modelmag_g", "modelmag_r", "modelmag_i", "modelmag_z"]
    available_cols = feature_cols + [col for col in sdss_mags if col in df.columns]

    features = df.select(available_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0)

    # Add coordinates to features if 3D
    if use_3d and coords_3d.shape[1] == 3:
        features = np.column_stack([features, coords_3d])

    # Create k-nearest neighbor graph
    if use_3d and coords_3d.shape[1] == 3:
        # Use 3D Cartesian coordinates
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
        nbrs.fit(coords_3d)
        distances, indices = nbrs.kneighbors(coords_3d)
        max_distance = distance_threshold
    else:
        # Use sky coordinates with haversine metric
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords_3d))
        distances, indices = nbrs.kneighbors(np.radians(coords_3d))
        max_distance = np.radians(
            distance_threshold / 3600.0
        )  # Convert arcsec to radians

    # Convert to edge list
    edge_list = []
    edge_weights = []

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for j, (dist, idx) in enumerate(zip(dist_row[1:], idx_row[1:])):  # Skip self
            if dist <= max_distance:
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(features, dtype=torch.float)

    # Use proper positions
    pos_tensor = (
        torch.tensor(coords_3d, dtype=torch.float)
        if use_3d and coords_3d.shape[1] == 3
        else None
    )

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos_tensor,
        num_nodes=len(features),
    )


def _create_generic_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create generic astronomical graph."""
    # Use first two columns as coordinates
    coords = df.to_numpy()[:, :2]
    features = df.to_numpy()
    features = np.nan_to_num(features, nan=0.0)

    # Create k-nearest neighbor graph
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    nbrs.fit(coords)

    distances, indices = nbrs.kneighbors(coords)

    # Convert to edge list
    edge_list = []
    edge_weights = []

    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for j, (dist, idx) in enumerate(zip(dist_row[1:], idx_row[1:])):  # Skip self
            if dist <= distance_threshold:
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

    # Convert to tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    node_features = torch.tensor(features, dtype=torch.float)

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(features),
    )
