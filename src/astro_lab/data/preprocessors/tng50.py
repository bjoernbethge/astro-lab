"""
TNG50 Data Preprocessor
======================

Preprocessor for TNG50 simulation data to PyTorch Geometric format.
Handles multiple HDF5 files and merges them during preprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import astropy.units as u
import polars as pl
import torch
from torch_geometric.data import Data

from astro_lab.config.data import get_data_config
from astro_lab.data.preprocessors.base import BaseSurveyProcessor
from astro_lab.tensors import SpatialTensorDict

logger = logging.getLogger(__name__)


class TNG50Preprocessor(BaseSurveyProcessor):
    """Preprocessor for TNG50 simulation data."""

    def __init__(self, survey_name: str = "tng50", data_config: Optional[Dict] = None):
        # Handle case where data_config is passed as first argument (old API)
        if isinstance(survey_name, dict):
            data_config = survey_name
            survey_name = "tng50"
        elif not isinstance(survey_name, str):
            # Fallback for any other type
            survey_name = "tng50"

        super().__init__(survey_name, data_config)

    def has_raw_data(self) -> bool:
        """Check if TNG50 raw data exists."""
        # Check for TNG50-4 directory structure
        tng50_4_dir = self.raw_dir.parent / "TNG50-4"
        if tng50_4_dir.exists():
            snapdir = tng50_4_dir / "output" / "snapdir_099"
            if snapdir.exists():
                hdf5_files = list(snapdir.glob("*.hdf5"))
                return len(hdf5_files) > 0

        # Fallback to old location
        if self.raw_dir.exists():
            hdf5_files = list(self.raw_dir.glob("*.h5")) + list(
                self.raw_dir.glob("*.hdf5")
            )
            return len(hdf5_files) > 0

        return False

    def _find_hdf5_files(self) -> List[Path]:
        """Find all HDF5 files in the TNG50-4 directory."""
        # Look in TNG50-4 directory first
        tng50_4_dir = self.raw_dir.parent / "TNG50-4"
        if tng50_4_dir.exists():
            snapdir = tng50_4_dir / "output" / "snapdir_099"
            if snapdir.exists():
                hdf5_files = list(snapdir.glob("*.hdf5"))
                if hdf5_files:
                    logger.info(
                        f"Found {len(hdf5_files)} HDF5 files in TNG50-4: {[f.name for f in hdf5_files]}"
                    )
                    return sorted(hdf5_files)

        # Fallback to old location
        hdf5_files = list(self.raw_dir.glob("*.h5")) + list(self.raw_dir.glob("*.hdf5"))

        if not hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in TNG50-4 or {self.raw_dir}")

        logger.info(
            f"Found {len(hdf5_files)} HDF5 files: {[f.name for f in hdf5_files]}"
        )
        return sorted(hdf5_files)

    def _load_hdf5_file(
        self, file_path: Path, dataset: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Load single HDF5 file as DataFrame.

        Args:
            file_path: Path to HDF5 file
            dataset: Specific dataset name (optional)

        Returns:
            DataFrame with file data
        """
        logger.info(f"Loading HDF5 file: {file_path.name}")

        try:
            import h5py

            with h5py.File(file_path, "r") as f:
                particle_types = ["PartType0", "PartType1", "PartType4", "PartType5"]
                all_data = []
                all_columns = set()

                # First pass: collect all columns
                for ptype in particle_types:
                    if ptype in f and "Coordinates" in f[ptype]:
                        ptype_data = f[ptype]
                        n_particles = ptype_data["Coordinates"].shape[0]
                        data_dict = {
                            "x": ptype_data["Coordinates"][:, 0],
                            "y": ptype_data["Coordinates"][:, 1],
                            "z": ptype_data["Coordinates"][:, 2],
                            "particle_type": [ptype] * n_particles,
                            "source_file": [file_path.stem] * n_particles,
                        }
                        if "Velocities" in ptype_data:
                            data_dict["vx"] = ptype_data["Velocities"][:, 0]
                            data_dict["vy"] = ptype_data["Velocities"][:, 1]
                            data_dict["vz"] = ptype_data["Velocities"][:, 2]
                        if "Masses" in ptype_data:
                            data_dict["mass"] = ptype_data["Masses"][:]
                        if "ParticleIDs" in ptype_data:
                            data_dict["particle_id"] = ptype_data["ParticleIDs"][:]
                        if ptype == "PartType0":
                            if "Density" in ptype_data:
                                data_dict["density"] = ptype_data["Density"][:]
                            if "InternalEnergy" in ptype_data:
                                data_dict["internal_energy"] = ptype_data[
                                    "InternalEnergy"
                                ][:]
                            if "GFM_Metallicity" in ptype_data:
                                data_dict["metallicity"] = ptype_data[
                                    "GFM_Metallicity"
                                ][:]
                        if ptype == "PartType4":
                            if "GFM_InitialMass" in ptype_data:
                                data_dict["initial_mass"] = ptype_data[
                                    "GFM_InitialMass"
                                ][:]
                            if "GFM_StellarFormationTime" in ptype_data:
                                data_dict["formation_time"] = ptype_data[
                                    "GFM_StellarFormationTime"
                                ][:]
                        if ptype == "PartType5":
                            if "BH_Mass" in ptype_data:
                                data_dict["bh_mass"] = ptype_data["BH_Mass"][:]
                            if "BH_Mdot" in ptype_data:
                                data_dict["bh_mdot"] = ptype_data["BH_Mdot"][:]
                        all_columns.update(data_dict.keys())
                        all_data.append(data_dict)

                if not all_data:
                    raise ValueError(f"No valid particle types found in {file_path}")

                # Second pass: align all DataFrames to the same columns
                all_columns = sorted(all_columns)
                dfs = []
                for data_dict in all_data:
                    # Fill missing columns with None
                    for col in all_columns:
                        if col not in data_dict:
                            data_dict[col] = [None] * len(data_dict["x"])
                    # Ensure column order
                    ordered = {col: data_dict[col] for col in all_columns}
                    dfs.append(pl.DataFrame(ordered))

                combined_df = pl.concat(dfs, how="vertical")
                logger.info(f"  Total particles: {len(combined_df)}")

        except ImportError:
            logger.error("h5py not available for HDF5 loading")
            raise
        except Exception as e:
            logger.error(f"Failed to load HDF5 file {file_path}: {e}")
            raise

        return combined_df

    def _merge_hdf5_files(
        self, files: List[Path], datasets: Optional[List[str]] = None
    ) -> Dict[str, pl.DataFrame]:
        """
        Merge multiple HDF5 files into separate DataFrames per particle type.

        Args:
            files: List of HDF5 file paths
            datasets: List of dataset names (optional)

        Returns:
            Dict mapping particle type to merged DataFrame
        """
        logger.info(f"Merging {len(files)} HDF5 files (per particle type)")

        # Mapping from TNG type to output name
        type_map = {
            "PartType0": "gas",
            "PartType1": "dm",
            "PartType4": "stars",
            "PartType5": "bh",
        }
        # Collect DataFrames for each type
        type_dfs = {k: [] for k in type_map.values()}
        all_columns = {k: set() for k in type_map.values()}

        for file_path in files:
            try:
                import h5py

                with h5py.File(file_path, "r") as f:
                    for tng_type, out_type in type_map.items():
                        if tng_type in f and "Coordinates" in f[tng_type]:
                            ptype_data = f[tng_type]
                            n_particles = ptype_data["Coordinates"].shape[0]
                            data_dict = {
                                "x": ptype_data["Coordinates"][:, 0],
                                "y": ptype_data["Coordinates"][:, 1],
                                "z": ptype_data["Coordinates"][:, 2],
                                "particle_type": [out_type] * n_particles,
                                "source_file": [file_path.stem] * n_particles,
                            }
                            if "Velocities" in ptype_data:
                                data_dict["vx"] = ptype_data["Velocities"][:, 0]
                                data_dict["vy"] = ptype_data["Velocities"][:, 1]
                                data_dict["vz"] = ptype_data["Velocities"][:, 2]
                            if "Masses" in ptype_data:
                                data_dict["mass"] = ptype_data["Masses"][:]
                            if "ParticleIDs" in ptype_data:
                                data_dict["particle_id"] = ptype_data["ParticleIDs"][:]
                            if tng_type == "PartType0":
                                if "Density" in ptype_data:
                                    data_dict["density"] = ptype_data["Density"][:]
                                if "InternalEnergy" in ptype_data:
                                    data_dict["internal_energy"] = ptype_data[
                                        "InternalEnergy"
                                    ][:]
                                if "GFM_Metallicity" in ptype_data:
                                    data_dict["metallicity"] = ptype_data[
                                        "GFM_Metallicity"
                                    ][:]
                            if tng_type == "PartType4":
                                if "GFM_InitialMass" in ptype_data:
                                    data_dict["initial_mass"] = ptype_data[
                                        "GFM_InitialMass"
                                    ][:]
                                if "GFM_StellarFormationTime" in ptype_data:
                                    data_dict["formation_time"] = ptype_data[
                                        "GFM_StellarFormationTime"
                                    ][:]
                            if tng_type == "PartType5":
                                if "BH_Mass" in ptype_data:
                                    data_dict["bh_mass"] = ptype_data["BH_Mass"][:]
                                if "BH_Mdot" in ptype_data:
                                    data_dict["bh_mdot"] = ptype_data["BH_Mdot"][:]
                            all_columns[out_type].update(data_dict.keys())
                            type_dfs[out_type].append(data_dict)
                            logger.info(
                                f"  Loaded {n_particles} {out_type} particles from {file_path.name}"
                            )
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
                continue

        # Align and merge DataFrames for each type
        merged = {}
        for out_type, dicts in type_dfs.items():
            if not dicts:
                continue
            cols = sorted(all_columns[out_type])
            dfs = []
            for data_dict in dicts:
                for col in cols:
                    if col not in data_dict:
                        data_dict[col] = [None] * len(data_dict["x"])
                ordered = {col: data_dict[col] for col in cols}
                dfs.append(pl.DataFrame(ordered))
            merged[out_type] = pl.concat(dfs, how="vertical")
            logger.info(f"  Total {out_type} particles: {len(merged[out_type])}")
        return merged

    def get_coordinate_columns(self) -> List[str]:
        """Get coordinate column names for TNG50."""
        # TNG50 uses simple column names: x, y, z
        return ["x", "y", "z"]

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract coordinates from TNG50 data."""
        # Check for the expected coordinate columns first
        expected_cols = ["x", "y", "z"]
        if all(col in df.columns for col in expected_cols):
            coords = torch.stack(
                [
                    torch.tensor(df["x"].to_numpy(), dtype=torch.float32),
                    torch.tensor(df["y"].to_numpy(), dtype=torch.float32),
                    torch.tensor(df["z"].to_numpy(), dtype=torch.float32),
                ],
                dim=1,
            )
            return coords

        # Fallback: Check for coordinate columns with different naming conventions
        coord_cols = []
        for prefix in ["Coordinates", "Pos", "Position"]:
            for axis in ["x", "y", "z", "X", "Y", "Z"]:
                col_name = f"{prefix}_{axis}"
                if col_name in df.columns:
                    coord_cols.append(col_name)
                elif f"{prefix}{axis}" in df.columns:
                    coord_cols.append(f"{prefix}{axis}")

        if len(coord_cols) < 3:
            # Final fallback: look for any 3D coordinate-like columns
            numeric_cols = [
                col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32]
            ]
            coord_cols = numeric_cols[:3]
            logger.warning(f"Using fallback coordinate columns: {coord_cols}")

        # Spatial coordinates (kpc/h)
        coords = torch.stack(
            [
                torch.tensor(df[coord_cols[0]].to_numpy(), dtype=torch.float32),
                torch.tensor(df[coord_cols[1]].to_numpy(), dtype=torch.float32),
                torch.tensor(df[coord_cols[2]].to_numpy(), dtype=torch.float32),
            ],
            dim=1,
        )

        return coords

    def _extract_features(self, data: pl.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Extract particle features from TNG50 data.

        Args:
            data: Raw TNG50 data

        Returns:
            Feature tensors
        """
        # Extract coordinates
        coords = self.extract_coordinates(data)

        # Extract velocities if available
        velocities = None
        if "velocities" in data.columns:
            velocities = torch.tensor(
                data["velocities"].to_numpy(), dtype=torch.float32
            )
        elif "vx" in data.columns and "vy" in data.columns and "vz" in data.columns:
            # Extract velocity components
            vx = torch.tensor(data["vx"].to_numpy(), dtype=torch.float32)
            vy = torch.tensor(data["vy"].to_numpy(), dtype=torch.float32)
            vz = torch.tensor(data["vz"].to_numpy(), dtype=torch.float32)
            velocities = torch.stack([vx, vy, vz], dim=1)
        else:
            # No velocity data available
            logger.warning("No velocity data available for TNG50 simulation")

        # Mass features (log scale)
        mass_col = None
        for col in ["Masses", "Mass", "M"]:
            if col in data.columns:
                mass_col = col
                break

        if mass_col:
            masses = torch.log10(
                torch.tensor(data[mass_col].to_numpy(), dtype=torch.float32)
            )
        else:
            masses = torch.ones(len(data), dtype=torch.float32)
            logger.warning("Mass data not found, using ones")

        # Basic features - handle None velocities
        if velocities is not None:
            basic_features = torch.cat([coords, velocities, masses.unsqueeze(1)], dim=1)
        else:
            # Create zero velocities if not available
            zero_velocities = torch.zeros(len(data), 3, dtype=torch.float32)
            basic_features = torch.cat(
                [coords, zero_velocities, masses.unsqueeze(1)], dim=1
            )

        # Try to extract additional features based on available columns
        additional_features = []

        # Density
        for col in ["Density", "Rho"]:
            if col in data.columns:
                additional_features.append(
                    torch.log10(torch.tensor(data[col].to_numpy(), dtype=torch.float32))
                )
                break

        # Temperature
        for col in ["Temperature", "Temp", "T"]:
            if col in data.columns:
                additional_features.append(
                    torch.log10(torch.tensor(data[col].to_numpy(), dtype=torch.float32))
                )
                break

        # Metallicity
        for col in ["Metallicity", "Z", "Metals"]:
            if col in data.columns:
                additional_features.append(
                    torch.tensor(data[col].to_numpy(), dtype=torch.float32)
                )
                break

        if additional_features:
            # Ensure all tensors are valid before concatenation
            valid_features = [basic_features] + [
                f for f in additional_features if f is not None
            ]
            node_features = torch.cat(valid_features, dim=1)
        else:
            node_features = basic_features

        result = {
            "x": node_features,
            "coords": coords,
            "masses": masses,
        }

        # Only add velocities if they exist
        if velocities is not None:
            result["velocities"] = velocities

        return result

    def _create_knsn_graph(self, coords: torch.Tensor, k: int = 10) -> torch.Tensor:
        """Create k-nearest neighbors graph from coordinates."""
        from sklearn.neighbors import NearestNeighbors

        if len(coords) <= k:
            # For small datasets, create fully connected graph
            edge_list = []
            for i in range(len(coords)):
                for j in range(len(coords)):
                    if i != j:
                        edge_list.append([i, j])
            return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # Build k-nearest neighbors graph
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Remove self-loops
        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip first neighbor (self)
                edge_list.append([i, j])
                edge_list.append([j, i])  # Undirected graph

        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    def _create_geometric_data(
        self, features: Dict[str, torch.Tensor], data: pl.DataFrame
    ) -> Data:
        """
        Create PyTorch Geometric Data object.

        Args:
            features: Extracted features
            data: Original data for metadata

        Returns:
            PyTorch Geometric Data object
        """
        # Create spatial graph based on 3D coordinates
        edge_index = self._create_knsn_graph(features["coords"], k=10)

        # Particle IDs (if available)
        particle_ids = None
        for col in ["ParticleIDs", "ID", "ParticleID"]:
            if col in data.columns:
                particle_ids = torch.tensor(data[col].to_numpy(), dtype=torch.long)
                break

        if particle_ids is None:
            particle_ids = torch.arange(len(data), dtype=torch.long)

        # Snapshot info
        snapshot = (
            data.get_column("source_file")[0]
            if "source_file" in data.columns
            else "unknown"
        )
        particle_type = (
            data.get_column("dataset_name")[0]
            if "dataset_name" in data.columns
            else "unknown"
        )

        return Data(
            x=features["x"],
            edge_index=edge_index,
            coords=features["coords"],
            velocities=features["velocities"],
            masses=features["masses"],
            particle_ids=particle_ids,
            snapshot=snapshot,
            particle_type=particle_type,
            survey="tng50",
            num_nodes=len(data),
        )

    def _save_processed_data(self, df: pl.DataFrame, output_file: Path):
        """Save processed data to Parquet file with TNG50-specific columns."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Select key columns for processed file
        key_columns = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "mass",
            "particle_id",
            "particle_type",
            "source_file",
            "density",
            "internal_energy",
            "metallicity",
            "initial_mass",
            "formation_time",
            "bh_mass",
            "bh_mdot",
        ]

        # Only include columns that exist in the DataFrame
        existing_columns = [col for col in key_columns if col in df.columns]
        processed_df = df.select(existing_columns)

        processed_df.write_parquet(output_file)
        logger.info(f"Saved processed data: {output_file}")

    def _load_processed_data(self, processed_file: Path) -> Dict[str, Any]:
        """Load existing processed data."""
        df = pl.read_parquet(processed_file)

        # Recreate tensors
        spatial_tensor = self.create_spatial_tensor(df)

        return {
            "spatial_tensor": spatial_tensor,
            "metadata": {
                "survey": "tng50",
                "data_release": "TNG50-4",
                "n_sources": len(df),
                "processed_file": str(processed_file),
                "particle_types": df["particle_type"].unique().to_list()
                if "particle_type" in df.columns
                else [],
            },
        }

    def preprocess(self, df: pl.DataFrame = None) -> None:
        """
        Preprocess TNG50 data from HDF5 files.

        This method is deprecated. Use the unified preprocess() API instead.
        """
        logger.warning(
            "TNG50Preprocessor.preprocess() is deprecated. Use the unified preprocess() API."
        )

        # Load data if not provided
        if df is None:
            hdf5_files = self._find_hdf5_files()
            merged_data = self._merge_hdf5_files(hdf5_files)

            # Combine all particle types
            all_data = []
            for ptype, data in merged_data.items():
                all_data.append(data)

            if all_data:
                df = pl.concat(all_data, how="vertical")
            else:
                logger.error("No data loaded from HDF5 files")
                return

        # Apply preprocessing
        df = self.preprocess_dataframe(df)

        # Create tensors
        spatial_tensor = self.create_spatial_tensor(df)

        # Save processed data
        processed_file = self.processed_dir / f"{self.survey_name}_processed.parquet"
        self._save_processed_data(df, processed_file)

        logger.info(f"TNG50 preprocessing complete: {len(df)} particles")

    def create_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Create SpatialTensorDict for TNG50 with cartesian coordinates."""
        coordinates = self.extract_coordinates(df)

        # TNG50 uses cartesian coordinates, no need for astronomical coordinate conversion
        return SpatialTensorDict(
            coordinates=coordinates,
            coordinate_system="cartesian",  # Use cartesian instead of astronomical
            unit=u.Unit("kpc"),  # TNG50 units are typically kpc/h
            epoch=2020.0,  # Use a default epoch for simulation data
        )

    def _setup_paths(self):
        """Setup data paths for TNG50 - override to avoid creating empty raw directory."""
        data_config = get_data_config()
        self.raw_dir = data_config.get_survey_raw_dir(self.survey_name)
        self.processed_dir = data_config.get_survey_processed_dir(self.survey_name)

        # Only create processed directory - raw directory is not used for TNG50
        # since we load from TNG50-4 directory structure
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Don't create raw_dir since TNG50 uses TNG50-4 structure
        # self.raw_dir.mkdir(parents=True, exist_ok=True)  # Commented out

    def preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply TNG50-specific preprocessing to DataFrame.
        This includes particle type filtering and feature extraction.
        """
        logger.info("Applying TNG50-specific preprocessing...")

        # TNG50-specific processing can be added here
        # For now, just return the DataFrame as-is
        return df
