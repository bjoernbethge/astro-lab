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
    # 🌟 NEW: TNG50 als vollwertiger Survey
    "tng50": {
        "name": "TNG50 Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": [],  # Keine Magnituden für Simulation
        "extra_cols": ["masses", "velocities_0", "velocities_1", "velocities_2"],
        "color_pairs": [],
        "default_limit": None,  # Keine Magnitude-Limits
        "url": "tng50",
        # 🌟 TENSOR METADATA für Simulation
        "filter_system": "none",
        "data_release": "TNG50-4",
        "coordinate_system": "cartesian",
        "photometric_bands": [],
        "particle_types": ["PartType0", "PartType1", "PartType4", "PartType5"],
        "tensor_column_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "masses": 3,
            "velocities_0": 4,
            "velocities_1": 5,
            "velocities_2": 6,
        },
        # Simulation-spezifische Metadaten
        "simulation_metadata": {
            "box_size": 35.0,  # Mpc/h
            "redshift": 0.0,
            "snapshot": 99,
            "cosmology": "Planck2018",
        },
    },
    # 🌟 NEW: TNG50-Temporal als vollwertiger Survey
    "tng50_temporal": {
        "name": "TNG50 Temporal Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": [],  # Keine Magnituden für Simulation
        "extra_cols": [
            "mass",
            "velocity_0",
            "velocity_1",
            "velocity_2",
            "particle_type",
            "snapshot_id",
            "redshift",
            "time_gyr",
            "scale_factor",
        ],
        "color_pairs": [],
        "default_limit": None,  # Keine Magnitude-Limits
        "url": "tng50_temporal",
        # 🌟 TENSOR METADATA für temporale Simulation
        "filter_system": "none",
        "data_release": "TNG50-4-Temporal",
        "coordinate_system": "cartesian",
        "photometric_bands": [],
        "particle_types": ["PartType0", "PartType1", "PartType4", "PartType5"],
        "tensor_column_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "mass": 3,
            "velocity_0": 4,
            "velocity_1": 5,
            "velocity_2": 6,
            "particle_type": 7,
            "snapshot_id": 8,
            "redshift": 9,
            "time_gyr": 10,
            "scale_factor": 11,
        },
        # Temporale Simulation-spezifische Metadaten
        "simulation_metadata": {
            "box_size": 35.0,  # Mpc/h
            "num_snapshots": 11,
            "redshift_range": [0.0, 1.0],
            "time_range": [0.0, 7.8],  # Gyr
            "cosmology": "Planck2018",
            "temporal_evolution": True,
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

        elif self.survey == "tng50":
            # TNG50 simulation data - load existing parquet files if available
            if hasattr(self, "data_path") and self.data_path:
                # Use provided data path
                parquet_path = Path(self.data_path)
                if parquet_path.exists():
                    print(f"📊 Loading existing TNG50 data from {parquet_path}")
                    df = pl.read_parquet(parquet_path)
                    if self.max_samples and len(df) > self.max_samples:
                        df = df.sample(self.max_samples, seed=42)
                    self._df_cache = df
                    return

            # Try to load from standard locations
            data_dir = Path("data/raw/tng50")
            particle_files = {
                "PartType0": "tng50_parttype0.parquet",
                "PartType1": "tng50_parttype1.parquet",
                "PartType4": "tng50_parttype4.parquet",
                "PartType5": "tng50_parttype5.parquet",
            }

            # Use PartType0 (gas) as default
            default_file = data_dir / particle_files["PartType0"]

            if default_file.exists():
                print(f"📊 Loading TNG50 PartType0 data from {default_file}")
                df = pl.read_parquet(default_file)
                if self.max_samples and len(df) > self.max_samples:
                    df = df.sample(self.max_samples, seed=42)
                self._df_cache = df
                return

            # Fallback: generate synthetic TNG50-like data
            print("⚠️ No TNG50 data found, generating synthetic simulation data")

            # 3D coordinates in a box (Mpc/h units)
            box_size = 35.0  # TNG50 box size
            x = np.random.uniform(0, box_size, n_objects)
            y = np.random.uniform(0, box_size, n_objects)
            z = np.random.uniform(0, box_size, n_objects)

            # Masses (log-normal distribution)
            masses = np.random.lognormal(8, 2, n_objects)

            # Velocities (Gaussian with cosmic expansion)
            velocities_0 = np.random.normal(0, 100, n_objects)  # km/s
            velocities_1 = np.random.normal(0, 100, n_objects)
            velocities_2 = np.random.normal(0, 100, n_objects)

            # Density for gas particles
            density = np.random.lognormal(-2, 3, n_objects)

            df = pl.DataFrame(
                {
                    "x": x.astype(np.float32),
                    "y": y.astype(np.float32),
                    "z": z.astype(np.float32),
                    "masses": masses.astype(np.float32),
                    "velocities_0": velocities_0.astype(np.float32),
                    "velocities_1": velocities_1.astype(np.float32),
                    "velocities_2": velocities_2.astype(np.float32),
                    "density": density.astype(np.float32),
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
        # Create temporary dataset to get DataFrame
        temp_dataset = AstroDataset(
            survey="gaia", max_samples=max_samples, return_tensor=False, **kwargs
        )
        temp_dataset.download()  # Ensure data is generated
        df = temp_dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "gaia", {"max_samples": max_samples})

    # Create AstroDataset with survey parameter and set DataFrame
    dataset = AstroDataset(
        survey="gaia", max_samples=max_samples, return_tensor=False, **kwargs
    )
    dataset.download()  # Ensure data is generated

    # Set the DataFrame as the data attribute
    dataset.data = dataset._load_polars_dataframe()

    return dataset


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
        # Create temporary dataset to get DataFrame
        temp_dataset = AstroDataset(
            survey="sdss", max_samples=max_samples, return_tensor=False, **kwargs
        )
        temp_dataset.download()
        df = temp_dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "sdss", {"max_samples": max_samples})

    # Create AstroDataset with survey parameter and set DataFrame
    dataset = AstroDataset(
        survey="sdss", max_samples=max_samples, return_tensor=False, **kwargs
    )
    dataset.download()

    # Set the DataFrame as the data attribute
    dataset.data = dataset._load_polars_dataframe()

    return dataset


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


def load_tng50_data(max_samples: Optional[int] = None) -> AstroDataset:
    """Load TNG50 simulation data as a survey-like dataset.

    Args:
        max_samples: Maximum number of particles to load (None = all)

    Returns:
        AstroDataset with TNG50 simulation data
    """
    config = SURVEY_CONFIGS["tng50"]

    # Load TNG50 data from processed files
    data_path = Path("data/processed/tng50_combined.parquet")
    if not data_path.exists():
        print(f"⚠️ TNG50 data not found at {data_path}. Generating demo data...")
        # Create AstroDataset with survey parameter and generate demo data
        dataset = AstroDataset(
            survey="tng50", max_samples=max_samples, return_tensor=False
        )
        dataset.download()  # Generate demo data
        dataset.data = dataset._load_polars_dataframe()
        return dataset

    # Load with Polars
    df = pl.read_parquet(data_path)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=42)

    # Create AstroDataset
    dataset = AstroDataset(
        name=config["name"],
        data=df,
        coord_cols=config["coord_cols"],
        mag_cols=config["mag_cols"],
        extra_cols=config["extra_cols"],
        color_pairs=config["color_pairs"],
        tensor_metadata=config,
    )

    return dataset


def load_tng50_temporal_data(
    max_samples: Optional[int] = None, snapshot_id: Optional[int] = None
) -> AstroDataset:
    """Load TNG50 temporal simulation data as a survey-like dataset.

    Args:
        max_samples: Maximum number of particles to load (None = all)
        snapshot_id: Specific snapshot to load (None = all snapshots)

    Returns:
        AstroDataset with TNG50 temporal simulation data
    """
    config = SURVEY_CONFIGS["tng50_temporal"]

    # Load TNG50 temporal data from processed files
    data_path = Path(
        "data/processed/tng50_temporal_100mb/processed/tng50_temporal_graphs_r1.0.pt"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"TNG50 temporal data not found at {data_path}. Run preprocessing first."
        )

    # Load PyTorch tensors
    import torch

    temporal_data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Extract data from temporal structure
    if snapshot_id is not None:
        # Load specific snapshot
        if snapshot_id >= len(temporal_data):
            raise ValueError(
                f"Snapshot {snapshot_id} not available. Max: {len(temporal_data) - 1}"
            )

        graph_data = temporal_data[snapshot_id]

        # Handle dict format (actual format of the data)
        if isinstance(graph_data, dict):
            positions = graph_data["x"][:, :3].numpy()  # x, y, z
            masses = graph_data["x"][:, 3].numpy()  # mass
            velocities = graph_data["x"][:, 4:7].numpy()  # vx, vy, vz
            particle_types = graph_data["x"][:, 7].numpy()  # particle_type

            # Get cosmological parameters
            redshifts = graph_data.get("redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = graph_data.get("time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = graph_data.get("scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        elif hasattr(graph_data, "x"):
            # Standard PyTorch Geometric format
            positions = graph_data.x[:, :3].numpy()  # x, y, z
            masses = graph_data.x[:, 3].numpy()  # mass
            velocities = graph_data.x[:, 4:7].numpy()  # vx, vy, vz
            particle_types = graph_data.x[:, 7].numpy()  # particle_type

            # Get cosmological parameters
            redshifts = getattr(graph_data, "redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = getattr(graph_data, "time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = getattr(graph_data, "scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        elif hasattr(graph_data, "pos"):
            # Alternative format with pos attribute
            positions = graph_data.pos[:, :3].numpy()
            masses = (
                graph_data.mass.numpy()
                if hasattr(graph_data, "mass")
                else np.ones(len(positions))
            )
            velocities = (
                graph_data.vel.numpy()
                if hasattr(graph_data, "vel")
                else np.zeros((len(positions), 3))
            )
            particle_types = (
                graph_data.particle_type.numpy()
                if hasattr(graph_data, "particle_type")
                else np.zeros(len(positions))
            )

            # Get cosmological parameters
            redshifts = getattr(graph_data, "redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = getattr(graph_data, "time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = getattr(graph_data, "scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        else:
            # Fallback: assume it's a dictionary or other format
            raise ValueError(
                f"Unknown TNG50 temporal data format for snapshot {snapshot_id}"
            )

        # Create DataFrame for single snapshot
        df_data = {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "mass": masses,
            "velocity_0": velocities[:, 0],
            "velocity_1": velocities[:, 1],
            "velocity_2": velocities[:, 2],
            "particle_type": particle_types,
            "snapshot_id": snapshot_id,
            "redshift": redshifts,
            "time_gyr": time_gyr,
            "scale_factor": scale_factor,
        }
        df = pl.DataFrame(df_data)

    else:
        # Load all snapshots combined
        all_data = []
        for i, graph_data in enumerate(temporal_data):
            # Handle dict format (actual format of the data)
            if isinstance(graph_data, dict):
                positions = graph_data["x"][:, :3].numpy()
                masses = graph_data["x"][:, 3].numpy()
                velocities = graph_data["x"][:, 4:7].numpy()
                particle_types = graph_data["x"][:, 7].numpy()

                # Get cosmological parameters
                redshifts = graph_data.get("redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = graph_data.get("time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = graph_data.get("scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            elif hasattr(graph_data, "x"):
                positions = graph_data.x[:, :3].numpy()
                masses = graph_data.x[:, 3].numpy()
                velocities = graph_data.x[:, 4:7].numpy()
                particle_types = graph_data.x[:, 7].numpy()

                # Get cosmological parameters
                redshifts = getattr(graph_data, "redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = getattr(graph_data, "time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = getattr(graph_data, "scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            elif hasattr(graph_data, "pos"):
                positions = graph_data.pos[:, :3].numpy()
                masses = (
                    graph_data.mass.numpy()
                    if hasattr(graph_data, "mass")
                    else np.ones(len(positions))
                )
                velocities = (
                    graph_data.vel.numpy()
                    if hasattr(graph_data, "vel")
                    else np.zeros((len(positions), 3))
                )
                particle_types = (
                    graph_data.particle_type.numpy()
                    if hasattr(graph_data, "particle_type")
                    else np.zeros(len(positions))
                )

                # Get cosmological parameters
                redshifts = getattr(graph_data, "redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = getattr(graph_data, "time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = getattr(graph_data, "scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            else:
                continue  # Skip unknown format

            snapshot_data = {
                "x": positions[:, 0],
                "y": positions[:, 1],
                "z": positions[:, 2],
                "mass": masses,
                "velocity_0": velocities[:, 0],
                "velocity_1": velocities[:, 1],
                "velocity_2": velocities[:, 2],
                "particle_type": particle_types,
                "snapshot_id": i,
                "redshift": redshifts,
                "time_gyr": time_gyr,
                "scale_factor": scale_factor,
            }
            all_data.append(pl.DataFrame(snapshot_data))

        df = pl.concat(all_data)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=42)

    # Create AstroDataset
    dataset = AstroDataset(
        survey="tng50_temporal",
        data_path=None,  # We already have the data
        k_neighbors=8,
        max_samples=max_samples,
        force_reload=False,
        return_tensor=True,  # Return as SurveyTensor
    )

    # Set the processed data directly
    dataset._df_cache = df
    dataset._processed_data = [df]  # Ensure it's treated as processed

    return dataset


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

    # TNG50 simulation data
    if "tng50" in name_lower or "parttype" in name_lower:
        return "tng50"
    elif (
        "x" in columns
        and "y" in columns
        and "z" in columns
        and any("velocities" in col for col in columns)
    ):
        return "tng50"
    # Observational surveys
    elif "nsa" in name_lower or any("petromag" in col for col in columns):
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
        elif survey_type == "tng50":
            return _create_tng50_graph(df, k_neighbors, distance_threshold, **kwargs)
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


def _create_tng50_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create TNG50 simulation graph from DataFrame with 3D coordinates."""
    print(f"🌌 Creating TNG50 graph: {len(df):,} particles, k={k_neighbors}")

    # Extract 3D coordinates (x, y, z)
    coord_cols = ["x", "y", "z"]
    missing_coords = [col for col in coord_cols if col not in df.columns]
    if missing_coords:
        raise ValueError(f"Missing coordinate columns for TNG50: {missing_coords}")

    coords = df.select(coord_cols).to_numpy()
    all_features = df.to_numpy()

    print(
        f"   📊 Coordinate range: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]"
    )
    print(
        f"   📊 Coordinate range: Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]"
    )
    print(
        f"   📊 Coordinate range: Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]"
    )

    # For TNG50, use 3D distance in comoving coordinates
    # Distance threshold is in simulation units (Mpc/h)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df)), metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Create edge list (exclude self-connections)
    edge_list = []
    edge_distances = []
    edge_velocities = []  # Store relative velocities as edge features

    # Extract velocity columns if available
    vel_cols = ["velocities_0", "velocities_1", "velocities_2"]
    has_velocities = all(col in df.columns for col in vel_cols)

    if has_velocities:
        velocities = df.select(vel_cols).to_numpy()

    for i, (dists, neighs) in enumerate(zip(distances, indices)):
        for dist, neigh in zip(dists[1:], neighs[1:]):  # Skip self
            if dist <= distance_threshold:
                edge_list.append([i, neigh])
                edge_distances.append(dist)

                # Add relative velocity as edge feature if available
                if has_velocities:
                    rel_velocity = np.linalg.norm(velocities[i] - velocities[neigh])
                    edge_velocities.append(rel_velocity)

    if not edge_list:
        print(f"   ⚠️ No edges found with distance threshold {distance_threshold}")
        # Create a minimal graph with self-loops
        edge_list = [[i, i] for i in range(min(10, len(df)))]
        edge_distances = [0.0] * len(edge_list)
        edge_velocities = [0.0] * len(edge_list) if has_velocities else []

    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

    # Create edge attributes (distance + relative velocity if available)
    if has_velocities and edge_velocities:
        edge_attr = torch.tensor(
            np.column_stack([edge_distances, edge_velocities]), dtype=torch.float
        )
    else:
        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)

    node_features = torch.tensor(all_features, dtype=torch.float)

    print(f"   ✅ TNG50 Graph: {len(all_features):,} nodes, {len(edge_list):,} edges")
    if has_velocities:
        print("   📊 Edge features: distance + relative velocity")
    else:
        print("   📊 Edge features: distance only")

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float),  # 3D positions
        num_nodes=len(all_features),
    )
