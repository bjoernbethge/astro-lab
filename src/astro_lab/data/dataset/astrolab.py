"""Base dataset classes for astronomical data with TensorDict and PyG integration.

Modern implementation following PyG best practices with clear TensorDict ↔ PyG Data conversion.
Uses central configuration for all paths and settings.

# NOTE: This module ensures that every graph has a valid x attribute (feature matrix). If no features are found, a default tensor of ones is used. DataLoader uses num_workers=8 by default for better performance and to avoid Lightning warnings.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from torch_geometric.data import Data, HeteroData, InMemoryDataset
from torch_geometric.loader import DataLoader

# Import central configuration
from astro_lab.config import get_data_paths
from astro_lab.data.samplers.base import SpatialSamplerMixin

# Import existing TensorDict infrastructure
from astro_lab.data.transforms import AstronomicalFeatures, Compose


class AstroLabInMemoryDataset(InMemoryDataset):
    """
    Universal in-memory dataset for astronomical data using SurveyTensorDict and PyG integration.
    Handles spatial, photometric, and temporal data in a unified way.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        survey_name: str = "gaia",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False,
        sampling_strategy: Optional[str] = None,
        sampler_kwargs: Optional[dict] = None,
        task: str = "node_classification",  # Add task parameter
    ):
        self.survey_name = survey_name
        self.metadata: Dict[str, Any] = {}
        self.sampling_strategy = sampling_strategy
        self.sampler_kwargs = sampler_kwargs or {}
        self.task = task  # Store task type
        self._tensordict_list: List[TensorDict] = []
        self._pyg_data_cache: Dict[int, Union[Data, HeteroData]] = {}

        data_paths = get_data_paths()
        if root is None:
            root = str(Path(data_paths["processed_dir"]) / survey_name)
        else:
            root = str(root)

        if transform is None:
            transform = None

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload,
        )
        self._data, self._slices = self._load_data()

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names to check/download."""
        return [f"{self.survey_name}_raw.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed file names to save/load."""
        return [
            f"{self.survey_name}.pt",
            f"{self.survey_name}_metadata.json",
        ]

    @property
    def raw_dir(self) -> str:
        data_paths = get_data_paths()
        raw_dir = Path(data_paths["raw_dir"]) / self.survey_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        return str(raw_dir)

    @property
    def processed_dir(self) -> str:
        data_paths = get_data_paths()
        processed_dir = Path(data_paths["processed_dir"]) / self.survey_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        return str(processed_dir)

    def download(self):
        """Download raw data files. Override for survey-specific logic if needed."""
        pass

    def _create_tensordict(self, raw_data: Any) -> TensorDict:
        """Create a TensorDict from raw data (dict or DataFrame row)."""
        return TensorDict(raw_data)

    def _tensordict_to_pyg(
        self, tensordict: TensorDict, device: Optional[str] = None
    ) -> Union[Data, HeteroData]:
        """Convert SurveyTensorDict to PyG Data."""
        try:
            data_dict = {
                k: v.to(device) if device and hasattr(v, "to") else v
                for k, v in tensordict.items()
            }

            # Ensure x is always 2D if present
            if "x" in data_dict and isinstance(data_dict["x"], torch.Tensor):
                x = data_dict["x"]
                if x.dim() == 1:
                    data_dict["x"] = x.unsqueeze(1)

            data = Data(**data_dict)

            # Set masks as attributes
            for mask_name in ["train_mask", "val_mask", "test_mask"]:
                if mask_name in data_dict:
                    setattr(data, mask_name, data_dict[mask_name])

            # Explicitly set num_nodes to suppress PyG warning
            if hasattr(data, "x") and data.x is not None:
                data.num_nodes = data.x.shape[0]
            elif hasattr(data, "pos") and data.pos is not None:
                data.num_nodes = data.pos.shape[0]
            else:
                # Fallback: try to infer from any tensor attribute
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:
                        data.num_nodes = value.shape[0]
                        break
                else:
                    # Last resort: set to 1
                    data.num_nodes = 1

            # Keep all tensors on CPU for pin_memory to work
            # DataLoader will handle GPU transfer automatically

            return data
        except Exception as e:
            print(f"[WARNING] Error in _tensordict_to_pyg: {e}")
            # Return fallback data
            num_nodes = 1
            x = torch.ones((num_nodes, 10), device="cpu")
            edge_index = torch.tensor([[0, 0]], dtype=torch.long, device="cpu").t()
            return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    def _pyg_to_tensordict(self, pyg_data: Union[Data, HeteroData]) -> TensorDict:
        """Convert PyG Data back to TensorDict."""
        return TensorDict({k: v for k, v in pyg_data.items()})

    def _process(self):
        """Override to prevent creation of unnecessary cache files."""
        # Skip PyG's automatic cache file creation for None pre_transform/pre_filter
        if self.pre_transform is None and self.pre_filter is None:
            # Just load the data without creating cache files
            self._data, self._slices = self._load_data()
            # Ensure data is properly set for HeteroData
            if self._data is not None:
                from torch_geometric.data import HeteroData

                if isinstance(self._data, HeteroData):
                    # For HeteroData, we need to set the internal storage correctly
                    self._data = self._data
        else:
            # Use PyG's default behavior if we actually have transforms/filters
            super()._process()

    def process(
        self,
        batch_size: int = 10000,
        device: Optional[str] = None,
        chunk_size: int = 10000,
    ):
        """Process harmonized parquet data into ML-ready PT format with proper labeling."""
        from pathlib import Path

        import polars as pl
        from tensordict import TensorDict

        data_paths = get_data_paths()
        processed_dir = Path(data_paths["processed_dir"]) / self.survey_name
        processed_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = processed_dir / f"{self.survey_name}.parquet"

        # Step 1: Load harmonized parquet data
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Harmonized parquet file not found: {parquet_path}\n"
                f"Please run 'astro-lab preprocess {self.survey_name}' first."
            )

        df_harmonized = pl.read_parquet(parquet_path)
        n_rows = len(df_harmonized)

        # Step 2: Create graphs based on task type
        from astro_lab.data.samplers import get_sampler

        sampler = (
            get_sampler(self.sampling_strategy, config=self.sampler_kwargs)
            if self.sampling_strategy
            else None
        )

        tensordict_list = []
        data_list = []

        if self.task == "graph_classification":
            # For graph classification, create multiple smaller graphs
            # Each graph gets its own label
            graphs_per_chunk = max(1, chunk_size // 100)  # ~100 nodes per graph

            for graph_idx in range(0, n_rows, n_rows // graphs_per_chunk):
                end_idx = min(graph_idx + n_rows // graphs_per_chunk, n_rows)
                chunk = df_harmonized.slice(graph_idx, end_idx - graph_idx)

                if len(chunk) < 10:  # Skip very small graphs
                    continue

                torch_data = chunk.to_torch(return_type="dict")

                # Extract coordinates and features
                coordinates = self._extract_coordinates(torch_data)
                features = self._extract_features(torch_data, coordinates)

                # Create graph-level label based on some property
                graph_label = self._create_graph_label(torch_data, graph_idx)

                if sampler:
                    # Remove y from torch_data to avoid duplicate parameter
                    torch_data_clean = {k: v for k, v in torch_data.items() if k != "y"}
                    graph_data = sampler.create_graph(
                        coordinates,
                        features,
                        survey_name=self.survey_name,
                        **torch_data_clean,
                    )
                else:
                    # Remove y from torch_data to avoid duplicate parameter
                    torch_data_clean = {k: v for k, v in torch_data.items() if k != "y"}
                    graph_data = SpatialSamplerMixin.default_graph(
                        coordinates,
                        features,
                        y=graph_label,
                        survey_name=self.survey_name,
                        **torch_data_clean,
                    )

                # CRITICAL: Set graph-level label
                graph_data.y = graph_label

                # Ensure features is not None and has correct shape
                if (
                    features is None
                    or not isinstance(features, torch.Tensor)
                    or features.shape[0] != coordinates.shape[0]
                ):
                    features = torch.ones(
                        (coordinates.shape[0], 1), device=coordinates.device
                    )

                if hasattr(graph_data, "x"):
                    if graph_data.x is None or (
                        isinstance(graph_data.x, torch.Tensor)
                        and graph_data.x.numel() == 0
                    ):
                        graph_data.x = torch.ones(
                            (coordinates.shape[0], 1), device=coordinates.device
                        )
                else:
                    graph_data.x = torch.ones(
                        (coordinates.shape[0], 1), device=coordinates.device
                    )

                # Explicitly set num_nodes to suppress PyG warning
                graph_data.num_nodes = coordinates.shape[0]

                data_list.append(graph_data)
                tensordict_list.append(self._pyg_to_tensordict(graph_data))

        else:
            # For node classification, process as before
            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                chunk = df_harmonized.slice(start, end - start)
                torch_data = chunk.to_torch(return_type="dict")

                coordinates = self._extract_coordinates(torch_data)
                features = self._extract_features(torch_data, coordinates)

                # Create node labels
                node_labels = self._create_node_labels(torch_data, len(coordinates))

                if sampler:
                    # Remove y from torch_data to avoid duplicate parameter
                    torch_data_clean = {k: v for k, v in torch_data.items() if k != "y"}
                    graph_data = sampler.create_graph(
                        coordinates,
                        features,
                        node_type_labels=node_labels,
                        survey_name=self.survey_name,
                        **torch_data_clean,
                    )
                else:
                    # Remove y from torch_data to avoid duplicate parameter
                    torch_data_clean = {k: v for k, v in torch_data.items() if k != "y"}
                    graph_data = SpatialSamplerMixin.default_graph(
                        coordinates,
                        features,
                        y=node_labels,
                        survey_name=self.survey_name,
                        **torch_data_clean,
                    )

                # Ensure features is not None and has correct shape
                if (
                    features is None
                    or not isinstance(features, torch.Tensor)
                    or features.shape[0] != coordinates.shape[0]
                ):
                    features = torch.ones(
                        (coordinates.shape[0], 1), device=coordinates.device
                    )

                if hasattr(graph_data, "x"):
                    if graph_data.x is None or (
                        isinstance(graph_data.x, torch.Tensor)
                        and graph_data.x.numel() == 0
                    ):
                        graph_data.x = torch.ones(
                            (coordinates.shape[0], 1), device=coordinates.device
                        )
                else:
                    graph_data.x = torch.ones(
                        (coordinates.shape[0], 1), device=coordinates.device
                    )

                # Explicitly set num_nodes to suppress PyG warning
                graph_data.num_nodes = coordinates.shape[0]

                data_list.append(graph_data)
                tensordict_list.append(self._pyg_to_tensordict(graph_data))

        # Step 3: Save processed data
        self._tensordict_list = tensordict_list
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.metadata.update(
            {
                "survey_name": self.survey_name,
                "task": self.task,
                "num_samples": len(data_list),
                "batch_size": batch_size,
                "device": device or "cpu",
                "sampling_strategy": self.sampling_strategy,
                "sampler_kwargs": self.sampler_kwargs,
            }
        )

        with open(self.processed_paths[1], "w") as f:
            import json

            json.dump(self.metadata, f, indent=2)

    def _extract_coordinates(self, torch_data: dict) -> torch.Tensor:
        """Extract spatial coordinates from data."""
        coord_cols = [
            col
            for col in torch_data.keys()
            if any(
                coord in col.lower() for coord in ["ra", "dec", "x", "y", "z", "pos"]
            )
        ]

        if "pos" in torch_data:
            return torch_data["pos"]
        elif "coordinates" in torch_data:
            return torch_data["coordinates"]
        elif len(coord_cols) >= 3:
            return torch.stack([torch_data[col] for col in coord_cols[:3]], dim=1)
        else:
            # Generate random coordinates if none found
            n_points = len(next(iter(torch_data.values())))
            return torch.randn(n_points, 3)

    def _extract_features(
        self, torch_data: dict, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Extract feature vectors from data."""
        feature_cols = [
            col
            for col in torch_data.keys()
            if col not in ["pos", "coordinates", "ra", "dec", "x", "y", "z"]
        ]

        if feature_cols:
            features = []
            for col in feature_cols[
                :10
            ]:  # Limit features to prevent too high dimensionality
                feat = torch_data[col]
                if isinstance(feat, torch.Tensor):
                    if feat.dim() == 0:
                        feat = feat.unsqueeze(0)
                    features.append(feat)
                else:
                    features.append(torch.tensor(feat, device="cpu"))
            features = (
                torch.stack(features, dim=1)
                if features[0].dim() == 1
                else torch.cat(features, dim=1)
            )

            # Ensure exactly 10 features
            if features.shape[1] < 10:
                # Pad with zeros
                padding = torch.zeros(
                    (features.shape[0], 10 - features.shape[1]), device="cpu"
                )
                features = torch.cat([features, padding], dim=1)
            elif features.shape[1] > 10:
                # Truncate to 10 features
                features = features[:, :10]

            return features
        else:
            # Fallback: exactly 10 features per node
            num_nodes = coordinates.shape[0]
            return torch.ones((num_nodes, 10), device="cpu")

    def _create_graph_label(self, torch_data: dict, graph_idx: int) -> torch.Tensor:
        """Create graph-level label for graph classification."""
        # Simple strategy: use graph index modulo num_classes
        # In practice, this should be based on actual graph properties

        # Example strategies based on survey:
        if self.survey_name == "gaia":
            # Classify stellar clusters by density
            if "parallax" in torch_data:
                mean_parallax = torch_data["parallax"].mean().item()
                # Near vs far clusters
                label = 0 if mean_parallax > 2.0 else 1
            else:
                label = graph_idx % 2

        elif self.survey_name in ["nsa", "sdss"]:
            # Classify galaxy groups by mass/luminosity
            if "stellar_mass" in torch_data:
                mean_mass = torch_data["stellar_mass"].mean().item()
                # Low vs high mass groups
                label = 0 if mean_mass < 1e10 else 1
            else:
                label = graph_idx % 2

        else:
            # Default: alternating labels
            label = graph_idx % 2

        return torch.tensor(label, dtype=torch.long)

    def _create_node_labels(self, torch_data: dict, num_nodes: int) -> torch.Tensor:
        """Create node-level labels for node classification using official stellar classification."""
        # Default: all nodes are class 0
        node_labels = torch.zeros(num_nodes, dtype=torch.long)

        # Survey-specific node labeling
        if self.survey_name == "gaia":
            # Official stellar classification using Gaia features
            if "bp_rp" in torch_data and "mg_abs" in torch_data:
                bp_rp = torch_data["bp_rp"]
                mg_abs = torch_data["mg_abs"]

                # Spectral classification based on bp_rp (color index)
                # O-B stars: bp_rp < 0.0 (very blue)
                # A-F stars: 0.0 <= bp_rp < 0.3 (white-yellow)
                # G-K stars: 0.3 <= bp_rp < 0.8 (yellow-orange)
                # M stars: bp_rp >= 0.8 (red)

                # Luminosity classification based on mg_abs (absolute magnitude)
                # Supergiants: mg_abs < -5.0
                # Giants: -5.0 <= mg_abs < 0.0
                # Main sequence: 0.0 <= mg_abs < 10.0
                # White dwarfs: mg_abs >= 10.0

                # Combined classification (0-7):
                # 0: O-B main sequence (hot blue stars)
                # 1: A-F main sequence (white-yellow stars)
                # 2: G-K main sequence (yellow-orange stars, like Sun)
                # 3: M main sequence (red dwarfs)
                # 4: Giants (red giants)
                # 5: Supergiants (very bright giants)
                # 6: White dwarfs (compact remnants)
                # 7: Other/unknown

                # Initialize all as "other"
                node_labels = torch.full((num_nodes,), 7, dtype=torch.long)

                # O-B main sequence
                mask = (bp_rp < 0.0) & (mg_abs >= 0.0) & (mg_abs < 10.0)
                node_labels[mask] = 0

                # A-F main sequence
                mask = (
                    (bp_rp >= 0.0) & (bp_rp < 0.3) & (mg_abs >= 0.0) & (mg_abs < 10.0)
                )
                node_labels[mask] = 1

                # G-K main sequence (most common)
                mask = (
                    (bp_rp >= 0.3) & (bp_rp < 0.8) & (mg_abs >= 0.0) & (mg_abs < 10.0)
                )
                node_labels[mask] = 2

                # M main sequence (red dwarfs)
                mask = (bp_rp >= 0.8) & (mg_abs >= 0.0) & (mg_abs < 10.0)
                node_labels[mask] = 3

                # Giants (red giants)
                mask = (mg_abs >= -5.0) & (mg_abs < 0.0)
                node_labels[mask] = 4

                # Supergiants
                mask = mg_abs < -5.0
                node_labels[mask] = 5

                # White dwarfs
                mask = mg_abs >= 10.0
                node_labels[mask] = 6

            elif "bp_rp" in torch_data:
                # Fallback: spectral classification only
                bp_rp = torch_data["bp_rp"]
                node_labels = torch.where(
                    bp_rp < 0.0,
                    0,  # O-B stars
                    torch.where(
                        bp_rp < 0.3,
                        1,  # A-F stars
                        torch.where(
                            bp_rp < 0.8,
                            2,  # G-K stars
                            3,
                        ),
                    ),
                )  # M stars

            elif "phot_g_mean_mag" in torch_data:
                # Fallback: magnitude-based classification
                mag = torch_data["phot_g_mean_mag"]
                node_labels = torch.where(
                    mag < 10.0,
                    0,  # Bright stars
                    torch.where(
                        mag < 15.0,
                        1,  # Medium stars
                        2,
                    ),
                )  # Faint stars

            elif "parallax" in torch_data:
                # Fallback: distance-based classification
                parallax = torch_data["parallax"]
                node_labels = torch.where(
                    parallax > 5.0,
                    0,  # Near stars
                    torch.where(
                        parallax > 1.0,
                        1,  # Medium distance
                        2,
                    ),
                )  # Far stars
            else:
                # Last resort: alternate labels
                node_labels = torch.arange(num_nodes, dtype=torch.long) % 8

        elif self.survey_name == "sdss":
            if "z" in torch_data:
                z = torch_data["z"]
                # Classify by redshift
                node_labels = torch.where(z > 0.1, 1, 0)

        return node_labels

    def _load_data(self) -> Tuple[Any, Any]:
        """Load processed data."""
        from pathlib import Path

        import torch
        from torch_geometric.data import HeteroData

        try:
            data, slices = torch.load(self.processed_paths[0], weights_only=False)
        except FileNotFoundError:
            # Data not processed yet - return None silently
            return None, None

        with open(self.processed_paths[1], "r") as f:
            import json

            self.metadata = json.load(f)

        # Reconstruct tensordict_list from PyG data if needed
        self._tensordict_list = []
        if slices is not None:
            # Handle both Data and HeteroData
            if isinstance(data, HeteroData):
                # For HeteroData, create a single tensordict from the entire graph
                tensordict = self._pyg_to_tensordict(data)
                self._tensordict_list = [tensordict]
            else:
                # For Data, reconstruct individual samples from slices
                # Don't call self.get() to avoid recursion
                if "x" in slices:
                    num_samples = len(slices["x"]) - 1
                    for i in range(num_samples):
                        # Extract individual sample from slices
                        sample_data = {}
                        for key, slice_list in slices.items():
                            if key in data and slice_list[i] != slice_list[i + 1]:
                                sample_data[key] = data[key][
                                    slice_list[i] : slice_list[i + 1]
                                ]

                        # Create Data object for this sample
                        sample_pyg = Data(**sample_data)

                        # Convert to tensordict
                        tensordict = self._pyg_to_tensordict(sample_pyg)
                        self._tensordict_list.append(tensordict)
                else:
                    # Fallback: create single tensordict from entire data
                    tensordict = self._pyg_to_tensordict(data)
                    self._tensordict_list = [tensordict]
        return data, slices

    def _load_raw_data(self) -> List[Any]:
        from pathlib import Path

        import pandas as pd

        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        if raw_path.exists():
            df = pd.read_parquet(raw_path)
            return df.to_dict("records")
        else:
            return []

    def len(self) -> int:
        # For HeteroData, return 1 (single graph)
        # For Data, return number of tensordicts
        if hasattr(self, "_data") and self._data is not None:
            from torch_geometric.data import HeteroData

            if isinstance(self._data, HeteroData):
                return 1
        return len(self._tensordict_list)

    def get(self, idx: int) -> Union[Data, HeteroData]:
        # Safety check for empty dataset
        if len(self._tensordict_list) == 0:
            # Create default data if dataset is empty
            num_nodes = 1
            x = torch.ones((num_nodes, 10), device="cpu")  # 10 features as default
            edge_index = torch.tensor([[0, 0]], dtype=torch.long, device="cpu").t()
            return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        # Bounds checking
        if idx >= len(self._tensordict_list):
            raise IndexError(
                f"Index {idx} out of bounds for dataset with {len(self._tensordict_list)} samples"
            )

        if idx in self._pyg_data_cache:
            data = self._pyg_data_cache[idx]
        else:
            try:
                tensordict = self._tensordict_list[idx]
                # Keep data on CPU - let DataLoader handle GPU transfer
                data = self._tensordict_to_pyg(tensordict, device="cpu")
            except Exception as e:
                print(f"[WARNING] Error converting tensordict {idx} to PyG data: {e}")
                # Create fallback data
                num_nodes = 1
                x = torch.ones((num_nodes, 10), device="cpu")  # 10 features as default
                edge_index = torch.tensor([[0, 0]], dtype=torch.long, device="cpu").t()
                data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

            # Ensure x is set - if tensordict is empty, create default data
            if not hasattr(data, "x") or data.x is None:
                # Create default data with at least one node on CPU
                num_nodes = 1
                x = torch.ones((num_nodes, 10), device="cpu")  # 10 features as default
                edge_index = torch.tensor([[0, 0]], dtype=torch.long, device="cpu").t()
                data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
            else:
                # Ensure x has the correct number of features (10)
                if data.x.shape[1] != 10:
                    # Pad or truncate to 10 features
                    current_features = data.x.shape[1]
                    if current_features < 10:
                        # Pad with zeros
                        padding = torch.zeros(
                            (data.x.shape[0], 10 - current_features), device="cpu"
                        )
                        data.x = torch.cat([data.x, padding], dim=1)
                    else:
                        # Truncate to 10 features
                        data.x = data.x[:, :10]

            if len(self._pyg_data_cache) < 1000:
                self._pyg_data_cache[idx] = data
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_tensordict(self, idx: int) -> TensorDict:
        tensordict = self._tensordict_list[idx]
        if self.transform is not None and hasattr(self.transform, "forward_tensordict"):
            tensordict = self.transform.forward_tensordict(tensordict)
        return tensordict

    def get_info(self) -> Dict[str, Any]:
        info = {
            "survey_name": self.survey_name,
            "task": self.task,
            "num_samples": len(self),
            "metadata": self.metadata,
            "has_pyg_cache": len(self._pyg_data_cache) > 0,
            "cache_size": len(self._pyg_data_cache),
            "raw_dir": self.raw_dir,
            "processed_dir": self.processed_dir,
        }
        if len(self) > 0:
            sample_td = self._tensordict_list[0]
            info["tensordict_keys"] = [str(k) for k in sample_td.keys()]
            info["tensordict_shape"] = sample_td.shape
            # Use self._data directly for HeteroData, otherwise use get(0)
            if hasattr(self, "_data") and self._data is not None:
                from torch_geometric.data import HeteroData

                if isinstance(self._data, HeteroData):
                    sample_pyg = self._data
                else:
                    sample_pyg = self.get(0)
            else:
                sample_pyg = self.get(0)
            if isinstance(sample_pyg, Data):
                info["pyg_type"] = "Data"
                info["num_nodes"] = sample_pyg.num_nodes
                info["num_edges"] = (
                    sample_pyg.num_edges if hasattr(sample_pyg, "edge_index") else 0
                )
                # Extract feature dimensions
                if hasattr(sample_pyg, "x") and sample_pyg.x is not None:
                    if isinstance(sample_pyg.x, torch.Tensor):
                        if sample_pyg.x.dim() == 1:
                            info["num_features"] = 1
                        else:
                            info["num_features"] = sample_pyg.x.shape[1]
                    else:
                        info["num_features"] = 1
                else:
                    # Fallback: count non-coordinate features in tensordict
                    coord_keys = {"ra", "dec", "x", "y", "z", "pos", "coordinates"}
                    feature_keys = [
                        k for k in sample_td.keys() if str(k) not in coord_keys
                    ]
                    if feature_keys:
                        info["num_features"] = len(feature_keys)
                    else:
                        info["num_features"] = 3  # Default fallback

                # Extract number of classes for classification tasks
                if hasattr(sample_pyg, "y") and sample_pyg.y is not None:
                    if isinstance(sample_pyg.y, torch.Tensor):
                        if sample_pyg.y.dim() == 0:  # Single graph label
                            info["num_classes"] = 2  # Binary classification default
                        elif sample_pyg.y.dim() == 1:
                            info["num_classes"] = int(sample_pyg.y.max().item()) + 1
                        else:
                            info["num_classes"] = sample_pyg.y.shape[1]
                    else:
                        info["num_classes"] = 2
                else:
                    # Try to find target column in tensordict
                    target_keys = ["target", "label", "class", "y"]
                    for key in target_keys:
                        if key in sample_td:
                            target = sample_td[key]
                            if isinstance(target, torch.Tensor) and hasattr(
                                target, "max"
                            ):
                                info["num_classes"] = int(target.max().item()) + 1
                                break
                    else:
                        info["num_classes"] = 2  # Default binary classification
            elif isinstance(sample_pyg, HeteroData):
                info["pyg_type"] = "HeteroData"
                # HeteroData.num_nodes and num_edges are already totals across all types
                info["num_nodes"] = sample_pyg.num_nodes
                info["num_edges"] = sample_pyg.num_edges
                # Extract features from first node type
                if sample_pyg.node_types:
                    first_node_type = sample_pyg.node_types[0]
                    if (
                        hasattr(sample_pyg[first_node_type], "x")
                        and sample_pyg[first_node_type].x is not None
                    ):
                        x_val = sample_pyg[first_node_type].x
                        if isinstance(x_val, torch.Tensor):
                            info["num_features"] = (
                                x_val.shape[1] if x_val.dim() > 1 else 1
                            )
                        else:
                            info["num_features"] = 3
                    else:
                        info["num_features"] = 3
                else:
                    info["num_features"] = 3
            else:
                info["pyg_type"] = "Unknown"
                info["num_nodes"] = None
                info["num_edges"] = 0
        else:
            # Fallback values for empty dataset
            info["num_features"] = 3
            info["num_classes"] = 2

        # Ensure num_classes is always set
        if "num_classes" not in info:
            info["num_classes"] = 2

        # Ensure num_classes >= 2 for node_classification
        if self.task == "node_classification" and info["num_classes"] < 2:
            # Use actual number of classes present in the data
            info["num_classes"] = max(2, info["num_classes"])
        return info

    def get_loader(self, batch_size=32, shuffle=True, **kwargs):
        """Return the appropriate loader based on data type.
        Uses HeteroDataLoader for HeteroData and DataLoader for regular Data.
        """
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = 8
        if "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = True
        if "pin_memory" not in kwargs:
            kwargs["pin_memory"] = True  # Enable pin_memory for better GPU performance

        if self.sampling_strategy:
            from astro_lab.data.samplers import get_sampler

            sampler = get_sampler(self.sampling_strategy, config=self.sampler_kwargs)
            return sampler.create_dataloader(
                self, batch_size=batch_size, shuffle=shuffle, **kwargs
            )
        else:
            # Check if we have HeteroData
            if hasattr(self, "_data") and self._data is not None:
                from torch_geometric.data import HeteroData

                if isinstance(self._data, HeteroData):
                    # Use HeteroDataLoader for heterogeneous graphs
                    from torch_geometric.loader import HeteroDataLoader

                    return HeteroDataLoader(
                        self, batch_size=batch_size, shuffle=shuffle, **kwargs
                    )

            # Use regular DataLoader for homogeneous graphs
            return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)


def create_dataset(*args, **kwargs):
    """Factory function for CLI and scripts to create a dataset instance."""
    # Only pass allowed arguments to AstroLabInMemoryDataset
    allowed = {
        "root",
        "survey_name",
        "transform",
        "pre_transform",
        "pre_filter",
        "force_reload",
        "sampling_strategy",
        "sampler_kwargs",
        "task",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return AstroLabInMemoryDataset(*args, **filtered_kwargs)
