"""
PyTorch Geometric Dataset Classes for Astronomical Data
======================================================

Dataset implementations using PyTorch Geometric's InMemoryDataset:
- Gaia DR3 stellar catalogs as graph datasets
- TNG50 simulation data as graph datasets
- AstroPhot galaxy fitting datasets
- NSA (NASA Sloan Atlas) galaxy datasets
- Exoplanet datasets from NASA Exoplanet Archive
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset

# Optional astro-lab tensor integration
try:
    from astro_lab.tensors import GraphFeatureTensor, Spatial3DTensor

    ASTRO_LAB_TENSORS_AVAILABLE = True
except ImportError:
    ASTRO_LAB_TENSORS_AVAILABLE = False


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
        # Data should already be available from data manager
        raw_path = Path("data/raw/gaia") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  Raw data not found: {raw_path}")
            print("   Please download using the datasets tab first.")

    def process(self):
        """Process raw data into graph format."""
        # Load raw catalog
        raw_path = Path("data/raw/gaia") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing Gaia catalog: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Filter by magnitude if needed
        if self.magnitude_limit < 20.0:
            df = df.filter(pl.col("phot_g_mean_mag") <= self.magnitude_limit)

        # Convert to numpy for processing
        coords = df.select(["ra", "dec"]).to_numpy()
        features = df.select(
            ["ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax"]
        ).to_numpy()

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} nodes...")

        # Create k-nearest neighbor graph
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.kneighbors(np.radians(coords))

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                if dist <= np.radians(self.max_distance):
                    edge_list.append([i, idx])
                    edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class TNG50GraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for TNG50 simulation data.

    Creates a single large spatial graph from ALL particle types across ALL snapshots.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        radius: float = 1.0,
        max_particles: int = 10000,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize TNG50 graph dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        radius : float, default 1.0
            Connection radius in simulation units
        max_particles : int, default 10000
            Maximum number of particles to load (total across all types/snapshots)
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.radius = radius
        self.max_particles = max_particles

        # Set default root if not provided
        if root is None:
            root = "data/processed/tng50_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["tng50_combined.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"tng50_graph_all_particles_r{self.radius:.1f}.pt"]

    def download(self):
        """Ensure raw Parquet data is available, else extract from ALL HDF5 snapshots."""
        raw_path = Path("data/raw/tng50") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  Raw data not found: {raw_path}")
            print("   Trying to extract from ALL HDF5 snapshots...")
            try:
                from astro_torch.data.manager import AstroDataManager

                manager = AstroDataManager()

                # Find all snapshots (snap_099.0.hdf5 to snap_099.10.hdf5)
                snapdir = Path("data/raw/TNG50-4/output/snapdir_099")
                hdf5_files = sorted(snapdir.glob("snap_099.*.hdf5"))

                if not hdf5_files:
                    raise FileNotFoundError(f"No HDF5 files found in {snapdir}")

                print(f"🌌 Found {len(hdf5_files)} snapshots to process")

                # Process all snapshots and all particle types
                all_particle_types = [
                    "PartType0",
                    "PartType1",
                    "PartType4",
                    "PartType5",
                ]
                combined_data = []

                # Define the complete set of columns we want in the final dataset
                # This ensures all particle types have the same structure
                all_columns = [
                    "x",
                    "y",
                    "z",  # Coordinates (always present)
                    "mass",  # Mass (normalized from 'masses')
                    "velocity_0",
                    "velocity_1",
                    "velocity_2",  # Velocities (normalized from 'velocities_*')
                    "density",  # Density (only in gas)
                    "temperature",  # Temperature (only in gas)
                    "metallicity",  # Metallicity (only in gas/stars)
                    "particle_type",  # Particle type identifier
                    "snapshot_id",  # Snapshot identifier
                ]

                for snap_idx, hdf5_file in enumerate(hdf5_files):
                    print(
                        f"📊 Processing snapshot {snap_idx + 1}/{len(hdf5_files)}: {hdf5_file.name}"
                    )

                    for p_type in all_particle_types:
                        try:
                            # Import this particle type from this snapshot
                            temp_parquet = manager.import_tng50_hdf5(
                                hdf5_file,
                                dataset_name=p_type,
                                max_particles=2000000,  # 2,000,000 particles per snapshot per particle type for ~100MB
                            )

                            # Load the temporary parquet and normalize columns
                            df = pl.read_parquet(temp_parquet)

                            # Normalize column names to match our standard format
                            column_mapping = {}
                            for col in df.columns:
                                if col == "masses":
                                    column_mapping[col] = "mass"
                                elif col.startswith("velocities_"):
                                    # velocities_0 -> velocity_0
                                    new_name = col.replace("velocities_", "velocity_")
                                    column_mapping[col] = new_name
                                else:
                                    # Keep other columns as is
                                    column_mapping[col] = col

                            # Rename columns
                            df = df.rename(column_mapping)

                            # Add particle type and snapshot info
                            df = df.with_columns(
                                [
                                    pl.lit(p_type).alias("particle_type"),
                                    pl.lit(snap_idx).alias("snapshot_id"),
                                ]
                            )

                            # Add missing columns with default values
                            missing_columns = [
                                col for col in all_columns if col not in df.columns
                            ]
                            for col in missing_columns:
                                if col in ["density", "temperature", "metallicity"]:
                                    # Use 0.0 for missing physical properties
                                    df = df.with_columns(pl.lit(0.0).alias(col))
                                elif col == "mass":
                                    # Use 1.0 for missing mass (normalized)
                                    df = df.with_columns(pl.lit(1.0).alias(col))
                                elif col.startswith("velocity_"):
                                    # Use 0.0 for missing velocities
                                    df = df.with_columns(pl.lit(0.0).alias(col))

                            # Ensure all columns are in the correct order
                            df = df.select(all_columns)

                            # Normalize data types to Float32 for all numeric columns
                            # except particle_type (string) and snapshot_id (integer)
                            numeric_columns = [
                                col
                                for col in all_columns
                                if col not in ["particle_type", "snapshot_id"]
                            ]
                            for col in numeric_columns:
                                df = df.with_columns(pl.col(col).cast(pl.Float32))

                            # Ensure snapshot_id is Int32
                            df = df.with_columns(pl.col("snapshot_id").cast(pl.Int32))

                            combined_data.append(df)
                            print(
                                f"   ✅ {p_type}: {len(df)} particles, {len(df.columns)} columns"
                            )

                        except Exception as e:
                            print(f"   ⚠️ {p_type}: {e}")
                            continue

                if not combined_data:
                    raise FileNotFoundError(
                        "No particle data could be extracted from snapshots!"
                    )

                # Combine all data
                print("🔄 Combining all particle data...")
                combined_df = pl.concat(combined_data)

                # Save combined data
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.write_parquet(raw_path)

                print(f"✅ Combined data saved: {len(combined_df):,} total particles")

                # Clean up temporary files
                for temp_file in Path("data/raw/tng50").glob("tng50_parttype*.parquet"):
                    temp_file.unlink()

            except Exception as e:
                print(f"❌ TNG50 HDF5→Parquet-Import fehlgeschlagen: {e}")
                raise
        else:
            print(f"✅ Raw data found: {raw_path}")

    def process(self):
        """Process raw data into graph format."""
        # Load raw catalog
        raw_path = Path("data/raw/tng50") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing TNG50 combined data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Limit number of particles
        if len(df) > self.max_particles:
            df = df.sample(n=self.max_particles, shuffle=True)

        # Extract coordinates and features dynamically
        coords = df.select(["x", "y", "z"]).to_numpy()

        # Define potential features and select only those present in the dataframe
        potential_features = [
            "x",
            "y",
            "z",
            "mass",
            "density",
            "temperature",
            "velocity_0",
            "velocity_1",
            "velocity_2",
            "metallicity",
        ]
        available_features = [f for f in potential_features if f in df.columns]

        if not available_features:
            raise ValueError("No feature columns found in the processed Parquet file.")

        print(f"   Found available features: {available_features}")
        features = df.select(available_features).to_numpy()

        # Handle missing values
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} particles...")

        # Create radius-based graph
        from sklearn.neighbors import radius_neighbors_graph

        # Use radius neighbors
        adjacency = radius_neighbors_graph(
            coords, radius=self.radius, mode="connectivity"
        )

        # Convert to edge list
        edge_list = np.array(adjacency.nonzero()).T
        edge_index = torch.tensor(edge_list.T, dtype=torch.long)

        # Calculate edge distances
        edge_distances = []
        for i, j in edge_list:
            dist = np.linalg.norm(coords[i] - coords[j])
            edge_distances.append(dist)

        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class TNG50TemporalDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for TNG50 simulation data as temporal graphs.

    Creates a sequence of spatial graphs from ALL particle types across ALL snapshots,
    preserving the temporal evolution and cosmological redshift information.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        radius: float = 1.0,
        max_particles: int = 10000,
        enable_temporal_edges: bool = True,
        temporal_edge_weight: float = 1.0,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """
        Initialize TNG50 temporal graph dataset.

        Parameters
        ----------
        root : str, optional
            Root directory for dataset files
        radius : float, default 1.0
            Connection radius in simulation units
        max_particles : int, default 10000
            Maximum number of particles per snapshot (total across all types)
        enable_temporal_edges : bool, default True
            Whether to add temporal edges between consecutive snapshots
        temporal_edge_weight : float, default 1.0
            Weight for temporal edges
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.radius = radius
        self.max_particles = max_particles
        self.enable_temporal_edges = enable_temporal_edges
        self.temporal_edge_weight = temporal_edge_weight

        # Set default root if not provided
        if root is None:
            root = "data/processed/tng50_temporal"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["tng50_temporal_combined.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"tng50_temporal_graphs_r{self.radius:.1f}.pt"]

    def download(self):
        """Ensure raw Parquet data is available, else extract from ALL HDF5 snapshots."""
        raw_path = Path("data/raw/tng50") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  Raw data not found: {raw_path}")
            print("   Trying to extract from ALL HDF5 snapshots with temporal info...")
            try:
                from astro_torch.data.manager import AstroDataManager

                manager = AstroDataManager()

                # Find all snapshots (snap_099.0.hdf5 to snap_099.10.hdf5)
                snapdir = Path("data/raw/TNG50-4/output/snapdir_099")
                hdf5_files = sorted(snapdir.glob("snap_099.*.hdf5"))

                if not hdf5_files:
                    raise FileNotFoundError(f"No HDF5 files found in {snapdir}")

                print(
                    f"🌌 Found {len(hdf5_files)} snapshots to process as temporal sequence"
                )

                # Define redshift values for each snapshot (approximate)
                # These are approximate redshifts for TNG50 snapshots
                redshift_map = {
                    "snap_099.0.hdf5": 0.0,  # Today
                    "snap_099.1.hdf5": 0.1,  # ~1.3 Gyr ago
                    "snap_099.2.hdf5": 0.2,  # ~2.5 Gyr ago
                    "snap_099.3.hdf5": 0.3,  # ~3.4 Gyr ago
                    "snap_099.4.hdf5": 0.4,  # ~4.2 Gyr ago
                    "snap_099.5.hdf5": 0.5,  # ~5.0 Gyr ago
                    "snap_099.6.hdf5": 0.6,  # ~5.7 Gyr ago
                    "snap_099.7.hdf5": 0.7,  # ~6.4 Gyr ago
                    "snap_099.8.hdf5": 0.8,  # ~7.0 Gyr ago
                    "snap_099.9.hdf5": 0.9,  # ~7.5 Gyr ago
                    "snap_099.10.hdf5": 1.0,  # ~7.8 Gyr ago
                }

                # Process all snapshots and all particle types
                all_particle_types = [
                    "PartType0",
                    "PartType1",
                    "PartType4",
                    "PartType5",
                ]
                combined_data = []

                # Define the complete set of columns we want in the final dataset
                all_columns = [
                    "x",
                    "y",
                    "z",  # Coordinates (always present)
                    "mass",  # Mass (normalized from 'masses')
                    "velocity_0",
                    "velocity_1",
                    "velocity_2",  # Velocities (normalized from 'velocities_*')
                    "density",  # Density (only in gas)
                    "temperature",  # Temperature (only in gas)
                    "metallicity",  # Metallicity (only in gas/stars)
                    "particle_type",  # Particle type identifier
                    "snapshot_id",  # Snapshot identifier
                    "redshift",  # Cosmological redshift
                    "time_gyr",  # Time in Gyr ago
                    "scale_factor",  # Cosmological scale factor
                ]

                for snap_idx, hdf5_file in enumerate(hdf5_files):
                    redshift = redshift_map.get(hdf5_file.name, 0.0)
                    time_gyr = self._redshift_to_time(redshift)
                    scale_factor = 1.0 / (1.0 + redshift)

                    print(
                        f"📊 Processing snapshot {snap_idx + 1}/{len(hdf5_files)}: {hdf5_file.name} (z={redshift:.2f}, {time_gyr:.1f} Gyr ago)"
                    )

                    for p_type in all_particle_types:
                        try:
                            # Import this particle type from this snapshot
                            temp_parquet = manager.import_tng50_hdf5(
                                hdf5_file,
                                dataset_name=p_type,
                                max_particles=2000000,  # 2,000,000 particles per snapshot per particle type for ~100MB
                            )

                            # Load the temporary parquet and normalize columns
                            df = pl.read_parquet(temp_parquet)

                            # Normalize column names to match our standard format
                            column_mapping = {}
                            for col in df.columns:
                                if col == "masses":
                                    column_mapping[col] = "mass"
                                elif col.startswith("velocities_"):
                                    # velocities_0 -> velocity_0
                                    new_name = col.replace("velocities_", "velocity_")
                                    column_mapping[col] = new_name
                                else:
                                    # Keep other columns as is
                                    column_mapping[col] = col

                            # Rename columns
                            df = df.rename(column_mapping)

                            # Add temporal and particle type info
                            df = df.with_columns(
                                [
                                    pl.lit(p_type).alias("particle_type"),
                                    pl.lit(snap_idx).alias("snapshot_id"),
                                    pl.lit(redshift).alias("redshift"),
                                    pl.lit(time_gyr).alias("time_gyr"),
                                    pl.lit(scale_factor).alias("scale_factor"),
                                ]
                            )

                            # Add missing columns with default values
                            missing_columns = [
                                col for col in all_columns if col not in df.columns
                            ]
                            for col in missing_columns:
                                if col in ["density", "temperature", "metallicity"]:
                                    # Use 0.0 for missing physical properties
                                    df = df.with_columns(pl.lit(0.0).alias(col))
                                elif col == "mass":
                                    # Use 1.0 for missing mass (normalized)
                                    df = df.with_columns(pl.lit(1.0).alias(col))
                                elif col.startswith("velocity_"):
                                    # Use 0.0 for missing velocities
                                    df = df.with_columns(pl.lit(0.0).alias(col))

                            # Ensure all columns are in the correct order
                            df = df.select(all_columns)

                            # Normalize data types to Float32 for all numeric columns
                            # except particle_type (string) and snapshot_id (integer)
                            numeric_columns = [
                                col
                                for col in all_columns
                                if col not in ["particle_type", "snapshot_id"]
                            ]
                            for col in numeric_columns:
                                df = df.with_columns(pl.col(col).cast(pl.Float32))

                            # Ensure snapshot_id is Int32
                            df = df.with_columns(pl.col("snapshot_id").cast(pl.Int32))

                            combined_data.append(df)
                            print(
                                f"   ✅ {p_type}: {len(df)} particles, {len(df.columns)} columns"
                            )

                        except Exception as e:
                            print(f"   ⚠️ {p_type}: {e}")
                            continue

                if not combined_data:
                    raise FileNotFoundError(
                        "No particle data could be extracted from snapshots!"
                    )

                # Combine all data
                print("🔄 Combining all particle data with temporal information...")
                combined_df = pl.concat(combined_data)

                # Save combined data
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                combined_df.write_parquet(raw_path)

                print(
                    f"✅ Combined temporal data saved: {len(combined_df):,} total particles"
                )

                # Clean up temporary files
                for temp_file in Path("data/raw/tng50").glob("tng50_parttype*.parquet"):
                    temp_file.unlink()

            except Exception as e:
                print(f"❌ TNG50 HDF5→Parquet-Import fehlgeschlagen: {e}")
                raise
        else:
            print(f"✅ Raw temporal data found: {raw_path}")

    def _redshift_to_time(self, redshift: float) -> float:
        """
        Convert redshift to time in Gyr ago.

        This is an approximation using a simple cosmology.
        """
        if redshift == 0:
            return 0.0

        # Simple approximation: t ≈ 13.8 * (1 - 1/(1+z)^1.5) Gyr
        # This is a rough estimate for a flat universe with Ωm=0.3
        return 13.8 * (1.0 - 1.0 / ((1.0 + redshift) ** 1.5))

    def process(self):
        """Process raw data into temporal graph format."""
        # Load raw catalog
        raw_path = Path("data/raw/tng50") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        print(f"🔄 Processing TNG50 temporal data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Get unique snapshot IDs and sort them
        snapshot_ids = sorted(df["snapshot_id"].unique().to_list())
        print(f"   Creating {len(snapshot_ids)} temporal graphs...")

        temporal_graphs = []

        for snap_idx in snapshot_ids:
            # Filter data for this snapshot
            snap_df = df.filter(pl.col("snapshot_id") == snap_idx)

            # Limit number of particles per snapshot
            if len(snap_df) > self.max_particles:
                snap_df = snap_df.sample(n=self.max_particles, shuffle=True)

            # Extract coordinates and features
            coords = snap_df.select(["x", "y", "z"]).to_numpy()

            # Define potential features and select only those present in the dataframe
            potential_features = [
                "x",
                "y",
                "z",
                "mass",
                "density",
                "temperature",
                "velocity_0",
                "velocity_1",
                "velocity_2",
                "metallicity",
                "redshift",
                "time_gyr",
                "scale_factor",
            ]
            available_features = [f for f in potential_features if f in snap_df.columns]

            if not available_features:
                raise ValueError(
                    "No feature columns found in the processed Parquet file."
                )

            print(
                f"   Snapshot {snap_idx}: {len(snap_df)} particles, {len(available_features)} features"
            )
            features = snap_df.select(available_features).to_numpy()

            # Handle missing values
            features = np.nan_to_num(features, nan=0.0)

            # Create radius-based graph for this snapshot
            from sklearn.neighbors import radius_neighbors_graph

            # Use radius neighbors
            adjacency = radius_neighbors_graph(
                coords, radius=self.radius, mode="connectivity"
            )

            # Convert to edge list
            edge_list = np.array(adjacency.nonzero()).T
            edge_index = torch.tensor(edge_list.T, dtype=torch.long)

            # Calculate edge distances
            edge_distances = []
            for i, j in edge_list:
                dist = np.linalg.norm(coords[i] - coords[j])
                edge_distances.append(dist)

            edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)
            node_features = torch.tensor(features, dtype=torch.float)

            # Create graph for this snapshot
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=len(features),
                # Add temporal metadata
                snapshot_id=torch.tensor([snap_idx], dtype=torch.long),
                redshift=torch.tensor([snap_df["redshift"][0]], dtype=torch.float),
                time_gyr=torch.tensor([snap_df["time_gyr"][0]], dtype=torch.float),
                scale_factor=torch.tensor(
                    [snap_df["scale_factor"][0]], dtype=torch.float
                ),
            )

            temporal_graphs.append(data)

        # Add temporal edges between consecutive snapshots if enabled
        if self.enable_temporal_edges and len(temporal_graphs) > 1:
            print(
                f"   Adding temporal edges between {len(temporal_graphs)} snapshots..."
            )

            for i in range(len(temporal_graphs) - 1):
                current_graph = temporal_graphs[i]
                next_graph = temporal_graphs[i + 1]

                # Create temporal edges: connect each node to itself in the next snapshot
                # This is a simple approach; more sophisticated methods could use particle tracking
                current_nodes = current_graph.num_nodes
                next_nodes = next_graph.num_nodes

                # Create temporal edge indices
                temporal_edge_index = torch.zeros(
                    (2, min(current_nodes, next_nodes)), dtype=torch.long
                )
                temporal_edge_index[0] = torch.arange(
                    min(current_nodes, next_nodes)
                )  # Current snapshot
                temporal_edge_index[1] = torch.arange(
                    min(current_nodes, next_nodes)
                )  # Next snapshot

                # Add temporal edge weights
                temporal_edge_attr = torch.full(
                    (min(current_nodes, next_nodes), 1),
                    self.temporal_edge_weight,
                    dtype=torch.float,
                )

                # Store temporal edges in the current graph
                current_graph.temporal_edge_index = temporal_edge_index
                current_graph.temporal_edge_attr = temporal_edge_attr
                current_graph.next_snapshot_id = torch.tensor([i + 1], dtype=torch.long)

        # Ensure all graphs have the same attributes for collation
        for i, graph in enumerate(temporal_graphs):
            if not hasattr(graph, "temporal_edge_index"):
                # Add empty temporal edges for graphs without them
                graph.temporal_edge_index = torch.zeros((2, 0), dtype=torch.long)
                graph.temporal_edge_attr = torch.zeros((0, 1), dtype=torch.float)
                graph.next_snapshot_id = torch.tensor(
                    [-1], dtype=torch.long
                )  # -1 indicates no next snapshot

        print(
            f"   Created {len(temporal_graphs)} temporal graphs with temporal evolution"
        )

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            temporal_graphs = [self.pre_transform(graph) for graph in temporal_graphs]

        # Save processed data
        self.save(temporal_graphs, self.processed_paths[0])


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
            Root directory for dataset files
        cutout_size : int, default 128
            Size of image cutouts
        pixel_scale : float, default 0.262
            Pixel scale in arcsec/pixel
        magnitude_range : tuple, default (10.0, 18.0)
            Magnitude range for filtering
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.catalog_path = Path(catalog_path)
        self.cutout_size = cutout_size
        self.pixel_scale = pixel_scale
        self.magnitude_range = magnitude_range
        self.k_neighbors = k_neighbors

        # Set default root if not provided
        if root is None:
            root = "data/processed/astrophot_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return [self.catalog_path.name]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        catalog_name = self.catalog_path.stem
        return [f"astrophot_graph_{catalog_name}_k{self.k_neighbors}.pt"]

    def download(self):
        """Download raw data if needed."""
        if not self.catalog_path.exists():
            print(f"⚠️  Catalog not found: {self.catalog_path}")
            print("   Please ensure catalog is available.")

    def process(self):
        """Process raw data into graph format."""
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")

        print(f"🔄 Processing AstroPhot catalog: {self.catalog_path.name}")

        # Load catalog
        df = pl.read_parquet(self.catalog_path)

        # Filter by magnitude
        if "phot_g_mean_mag" in df.columns:
            mag_col = "phot_g_mean_mag"
        elif "r_mag" in df.columns:
            mag_col = "r_mag"
        else:
            mag_col = df.columns[2]  # Assume third column is magnitude

        df = df.filter(
            (pl.col(mag_col) >= self.magnitude_range[0])
            & (pl.col(mag_col) <= self.magnitude_range[1])
        )

        # Extract coordinates and features
        coords = df.select(["ra", "dec"]).to_numpy()

        # Create feature matrix
        feature_cols = ["ra", "dec"]
        if mag_col in df.columns:
            feature_cols.append(mag_col)
        if "bp_rp" in df.columns:
            feature_cols.append("bp_rp")
        if "parallax" in df.columns:
            feature_cols.append("parallax")

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} galaxies...")

        # Create k-nearest neighbor graph
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.kneighbors(np.radians(coords))

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class NSAGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for NSA (NASA Sloan Atlas) galaxy data.

    Creates spatial graphs from galaxy catalogs with k-nearest neighbor
    connections based on 3D coordinates.
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
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.max_galaxies = max_galaxies
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold

        # Set default root if not provided
        if root is None:
            root = "data/processed/nsa_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["nsa_catalog.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"nsa_graph_k{self.k_neighbors}_n{self.max_galaxies}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/processed/nsa") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  NSA data not found: {raw_path}")
            print("   Please load NSA data using astro_lab.data.load_nsa_data()")

    def process(self):
        """Process raw data into graph format."""
        # Load NSA catalog
        raw_path = Path("data/processed/nsa") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"NSA data not found: {raw_path}")

        print(f"🔄 Processing NSA catalog: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Limit number of galaxies
        if len(df) > self.max_galaxies:
            df = df.sample(self.max_galaxies)

        # Extract coordinates and features
        if "ZDIST" in df.columns:
            # Use direct distance measurements
            ra = df["RA"].to_numpy()
            dec = df["DEC"].to_numpy()
            distances = df["ZDIST"].to_numpy()
        else:
            # Fallback to redshift
            ra = df["RA"].to_numpy()
            dec = df["DEC"].to_numpy()
            z = (
                df.get_column("Z").to_numpy()
                if "Z" in df.columns
                else np.ones(len(df)) * 0.1
            )
            # Convert redshift to distance (simplified)
            distances = z * 3000  # Rough Mpc conversion

        # Convert to Cartesian coordinates
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = distances * np.cos(dec_rad) * np.cos(ra_rad)
        y = distances * np.cos(dec_rad) * np.sin(ra_rad)
        z = distances * np.sin(dec_rad)

        coords_3d = np.column_stack([x, y, z])

        # Create feature matrix
        feature_cols = ["RA", "DEC"]
        if "ZDIST" in df.columns:
            feature_cols.append("ZDIST")
        if "PETROMAG_R" in df.columns:
            feature_cols.append("PETROMAG_R")
        if "MASS" in df.columns:
            feature_cols.append("MASS")

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        # Add 3D coordinates to features
        features = np.column_stack([features, coords_3d])

        print(f"   Creating graph with {len(features):,} galaxies...")

        # Create k-nearest neighbor graph in 3D space
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="euclidean")
        nbrs.fit(coords_3d)

        distances_3d, indices = nbrs.kneighbors(coords_3d)

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances_3d, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                if dist <= self.distance_threshold:
                    edge_list.append([i, idx])
                    edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(coords_3d, dtype=torch.float),
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class ExoplanetGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for Exoplanet data.

    Creates spatial graphs from exoplanet systems with connections
    based on stellar distances and system properties.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        max_planets: int = 5000,
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
        max_planets : int, default 5000
            Maximum number of planets to include
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
        self.max_planets = max_planets
        self.k_neighbors = k_neighbors
        self.max_distance = max_distance

        # Set default root if not provided
        if root is None:
            root = "data/processed/exoplanet_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["confirmed_exoplanets.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"exoplanet_graph_k{self.k_neighbors}_n{self.max_planets}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/processed/exoplanets") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  Exoplanet data not found: {raw_path}")
            print("   Please download using scripts/download_all_exoplanets.py")

    def process(self):
        """Process raw data into graph format."""
        # Load exoplanet catalog
        raw_path = Path("data/processed/exoplanets") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Exoplanet data not found: {raw_path}")

        print(f"🔄 Processing Exoplanet catalog: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Filter out planets without distance measurements
        if "sy_dist" in df.columns:
            df = df.filter(pl.col("sy_dist").is_not_null())
            df = df.filter(pl.col("sy_dist") > 0)

        # Limit number of planets
        if len(df) > self.max_planets:
            df = df.sample(self.max_planets)

        # Extract coordinates and features
        ra = df["ra"].to_numpy()
        dec = df["dec"].to_numpy()
        distances = (
            df["sy_dist"].to_numpy()
            if "sy_dist" in df.columns
            else np.ones(len(df)) * 10
        )

        # Convert to Cartesian coordinates
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)

        x = distances * np.cos(dec_rad) * np.cos(ra_rad)
        y = distances * np.cos(dec_rad) * np.sin(ra_rad)
        z = distances * np.sin(dec_rad)

        coords_3d = np.column_stack([x, y, z])

        # Create feature matrix
        feature_cols = ["ra", "dec"]
        if "sy_dist" in df.columns:
            feature_cols.append("sy_dist")
        if "pl_rade" in df.columns:
            feature_cols.append("pl_rade")
        if "pl_masse" in df.columns:
            feature_cols.append("pl_masse")
        if "disc_year" in df.columns:
            feature_cols.append("disc_year")

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        # Add 3D coordinates to features
        features = np.column_stack([features, coords_3d])

        print(f"   Creating graph with {len(features):,} exoplanets...")

        # Create k-nearest neighbor graph in 3D space
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="euclidean")
        nbrs.fit(coords_3d)

        distances_3d, indices = nbrs.kneighbors(coords_3d)

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances_3d, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                if dist <= self.max_distance:
                    edge_list.append([i, idx])
                    edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(coords_3d, dtype=torch.float),
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class SDSSSpectralDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for SDSS spectral data.

    Creates graph datasets from SDSS spectroscopic observations
    with connections based on spectral similarity and sky position.
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
            Threshold for spectral similarity connections
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.max_spectra = max_spectra
        self.k_neighbors = k_neighbors
        self.spectral_similarity_threshold = spectral_similarity_threshold

        # Set default root if not provided
        if root is None:
            root = "data/processed/sdss_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["sdss_spectra.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"sdss_spectral_graph_k{self.k_neighbors}_n{self.max_spectra}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  SDSS spectral data not found: {raw_path}")
            print("   Please ensure AstroML datasets are available.")

    def process(self):
        """Process raw data into graph format."""
        # Load SDSS spectral catalog
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"SDSS spectral data not found: {raw_path}")

        print(f"🔄 Processing SDSS spectral data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Limit number of spectra
        if len(df) > self.max_spectra:
            df = df.sample(self.max_spectra)

        # Extract coordinates and spectral features
        coords = df.select(["ra", "dec"]).to_numpy()

        # Create feature matrix with available spectral properties
        feature_cols = ["ra", "dec"]
        if "z" in df.columns:
            feature_cols.append("z")  # redshift
        if "class" in df.columns:
            # Convert class to numeric (0=galaxy, 1=star, 2=quasar)
            class_map = {"GALAXY": 0, "STAR": 1, "QSO": 2}
            df = df.with_columns(
                pl.col("class")
                .map_elements(lambda x: class_map.get(x, 0), return_dtype=pl.Int32)
                .alias("class_num")
            )
            feature_cols.append("class_num")
        if "subClass" in df.columns:
            feature_cols.append("subClass")

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} spectra...")

        # Create k-nearest neighbor graph based on sky coordinates
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.kneighbors(np.radians(coords))

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class LINEARLightcurveDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for LINEAR lightcurve data.

    Creates temporal graph datasets from LINEAR asteroid lightcurves
    with connections based on orbital similarity and temporal patterns.
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
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.max_objects = max_objects
        self.k_neighbors = k_neighbors
        self.min_observations = min_observations

        # Set default root if not provided
        if root is None:
            root = "data/processed/linear_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["linear_lightcurves.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"linear_lightcurve_graph_k{self.k_neighbors}_n{self.max_objects}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  LINEAR lightcurve data not found: {raw_path}")
            print("   Please ensure AstroML datasets are available.")

    def process(self):
        """Process raw data into graph format."""
        # Load LINEAR lightcurve catalog
        raw_path = Path("data/datasets/astroml") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"LINEAR lightcurve data not found: {raw_path}")

        print(f"🔄 Processing LINEAR lightcurve data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Filter objects with sufficient observations
        object_counts = df.group_by("objectid").agg(pl.count().alias("n_obs"))
        valid_objects = object_counts.filter(pl.col("n_obs") >= self.min_observations)[
            "objectid"
        ]

        df = df.filter(pl.col("objectid").is_in(valid_objects))

        # Limit number of objects
        if len(valid_objects) > self.max_objects:
            selected_objects = valid_objects.sample(self.max_objects)
            df = df.filter(pl.col("objectid").is_in(selected_objects))

        # Group by object and create features
        object_features = df.group_by("objectid").agg(
            [
                pl.col("ra").mean().alias("ra_mean"),
                pl.col("dec").mean().alias("dec_mean"),
                pl.col("mag").std().alias("mag_std"),
                pl.col("mag").mean().alias("mag_mean"),
                pl.col("mag").max().alias("mag_max"),
                pl.col("mag").min().alias("mag_min"),
                pl.count().alias("n_obs"),
            ]
        )

        coords = object_features.select(["ra_mean", "dec_mean"]).to_numpy()
        features = object_features.select(
            [
                "ra_mean",
                "dec_mean",
                "mag_mean",
                "mag_std",
                "mag_max",
                "mag_min",
                "n_obs",
            ]
        ).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} lightcurve objects...")

        # Create k-nearest neighbor graph based on sky coordinates
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.kneighbors(np.radians(coords))

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class RRLyraeDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for RR Lyrae variable star data.

    Creates temporal graph datasets from RR Lyrae observations
    with connections based on period similarity and sky position.
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
            Maximum number of RR Lyrae stars to include
        k_neighbors : int, default 5
            Number of nearest neighbors for graph construction
        period_similarity_threshold : float, default 0.1
            Threshold for period similarity connections (days)
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.max_stars = max_stars
        self.k_neighbors = k_neighbors
        self.period_similarity_threshold = period_similarity_threshold

        # Set default root if not provided
        if root is None:
            root = "data/processed/rrlyrae_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["rrlyrae_real_data_cleaned.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"rrlyrae_graph_k{self.k_neighbors}_n{self.max_stars}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/processed") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  RR Lyrae data not found: {raw_path}")
            print("   Please ensure RR Lyrae datasets are processed.")

    def process(self):
        """Process raw data into graph format."""
        # Load RR Lyrae catalog
        raw_path = Path("data/processed") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"RR Lyrae data not found: {raw_path}")

        print(f"🔄 Processing RR Lyrae data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Limit number of stars
        if len(df) > self.max_stars:
            df = df.sample(self.max_stars)

        # Extract coordinates and variable star features
        coords = df.select(["ra", "dec"]).to_numpy()

        # Create feature matrix with RR Lyrae properties
        feature_cols = ["ra", "dec"]
        if "period" in df.columns:
            feature_cols.append("period")
        if "amplitude" in df.columns:
            feature_cols.append("amplitude")
        if "mean_mag" in df.columns:
            feature_cols.append("mean_mag")
        if "metallicity" in df.columns:
            feature_cols.append("metallicity")
        if "distance" in df.columns:
            feature_cols.append("distance")

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        print(f"   Creating graph with {len(features):,} RR Lyrae stars...")

        # Create k-nearest neighbor graph based on sky coordinates
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="haversine")
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.kneighbors(np.radians(coords))

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                edge_list.append([i, idx])
                edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])


class SatelliteOrbitDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for satellite orbit data.

    Creates orbital graph datasets from satellite trajectory data
    with connections based on orbital similarity and proximity.
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
            Threshold for orbital similarity connections (km)
        transform : callable, optional
            Transform to apply to each graph
        pre_transform : callable, optional
            Transform to apply before saving
        pre_filter : callable, optional
            Filter to apply before saving
        """
        self.max_satellites = max_satellites
        self.k_neighbors = k_neighbors
        self.orbital_similarity_threshold = orbital_similarity_threshold

        # Set default root if not provided
        if root is None:
            root = "data/processed/satellite_graphs"

        super().__init__(root, transform, pre_transform, pre_filter)

        # Load processed data
        if len(self.processed_file_names) > 0:
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files to download."""
        return ["satellite_orbits.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files."""
        return [f"satellite_orbit_graph_k{self.k_neighbors}_n{self.max_satellites}.pt"]

    def download(self):
        """Download raw data if needed."""
        raw_path = Path("data/processed") / self.raw_file_names[0]
        if not raw_path.exists():
            print(f"⚠️  Satellite orbit data not found: {raw_path}")
            print("   Please ensure satellite datasets are processed.")

    def process(self):
        """Process raw data into graph format."""
        # Load satellite orbit catalog
        raw_path = Path("data/processed") / self.raw_file_names[0]

        if not raw_path.exists():
            raise FileNotFoundError(f"Satellite orbit data not found: {raw_path}")

        print(f"🔄 Processing Satellite orbit data: {raw_path.name}")

        # Load catalog with Polars
        df = pl.read_parquet(raw_path)

        # Limit number of satellites
        if len(df) > self.max_satellites:
            df = df.sample(self.max_satellites)

        # Extract orbital elements and position features
        feature_cols = []
        if "semi_major_axis" in df.columns:
            feature_cols.append("semi_major_axis")
        if "eccentricity" in df.columns:
            feature_cols.append("eccentricity")
        if "inclination" in df.columns:
            feature_cols.append("inclination")
        if "longitude_of_ascending_node" in df.columns:
            feature_cols.append("longitude_of_ascending_node")
        if "argument_of_perigee" in df.columns:
            feature_cols.append("argument_of_perigee")
        if "mean_anomaly" in df.columns:
            feature_cols.append("mean_anomaly")
        if "x" in df.columns and "y" in df.columns and "z" in df.columns:
            feature_cols.extend(["x", "y", "z"])

        # Fallback to available columns
        if not feature_cols:
            feature_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ][:6]

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)

        # Use first 3 columns as position coordinates for graph construction
        if features.shape[1] >= 3:
            coords_3d = features[:, :3]
        else:
            # Generate synthetic 3D coordinates if not available
            coords_3d = np.random.randn(len(features), 3) * 1000

        print(f"   Creating graph with {len(features):,} satellites...")

        # Create k-nearest neighbor graph based on orbital/spatial coordinates
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric="euclidean")
        nbrs.fit(coords_3d)

        distances_3d, indices = nbrs.kneighbors(coords_3d)

        # Convert to edge list (exclude self-connections)
        edge_list = []
        edge_weights = []

        for i, (dist_row, idx_row) in enumerate(zip(distances_3d, indices)):
            for j, (dist, idx) in enumerate(
                zip(dist_row[1:], idx_row[1:])
            ):  # Skip self
                if dist <= self.orbital_similarity_threshold:
                    edge_list.append([i, idx])
                    edge_weights.append(float(dist))

        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        node_features = torch.tensor(features, dtype=torch.float)

        # Create single large graph
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(coords_3d, dtype=torch.float),
            num_nodes=len(features),
        )

        print(f"   Graph created: {data.num_nodes:,} nodes, {data.num_edges:,} edges")

        # Apply pre-transform if provided
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save processed data
        self.save([data], self.processed_paths[0])
