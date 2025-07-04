"""
Data Adapters for UI Components
==============================

Adapters to convert between TensorDicts (used in models/training) and
simple data structures (used in UI for performance).
"""

from typing import Any, Dict

import polars as pl
import torch


def tensordict_to_ui_data(tensordict: Any) -> Dict[str, Any]:
    """Convert TensorDict to simple dict for UI visualization.

    Args:
        tensordict: Any TensorDict subclass

    Returns:
        Dict with numpy arrays and metadata
    """
    ui_data = {"coordinates": None, "properties": {}, "metadata": {}}

    # Extract coordinates
    if hasattr(tensordict, "coordinates"):
        ui_data["coordinates"] = tensordict.coordinates.cpu().numpy()
    elif hasattr(tensordict, "__getitem__") and "coordinates" in tensordict:
        ui_data["coordinates"] = tensordict["coordinates"].cpu().numpy()
    elif hasattr(tensordict, "pos"):
        ui_data["coordinates"] = tensordict.pos.cpu().numpy()

    # Extract metadata
    if hasattr(tensordict, "meta"):
        ui_data["metadata"] = tensordict.meta
    elif hasattr(tensordict, "metadata"):
        ui_data["metadata"] = tensordict.metadata

    # Extract other tensor data as properties
    if hasattr(tensordict, "keys"):
        for key in tensordict.keys():
            if key not in ["coordinates", "pos", "meta", "metadata"]:
                value = tensordict[key]
                if torch.is_tensor(value):
                    ui_data["properties"][key] = value.cpu().numpy()
                else:
                    ui_data["properties"][key] = value

    return ui_data


def dataframe_to_ui_data(df: pl.DataFrame) -> Dict[str, Any]:
    """Convert Polars DataFrame to UI data format.

    Args:
        df: Polars DataFrame

    Returns:
        Dict with coordinates and properties
    """
    ui_data = {
        "coordinates": None,
        "properties": {},
        "metadata": {"source": "dataframe", "n_objects": len(df)},
    }

    # Extract coordinates
    if all(col in df.columns for col in ["x", "y", "z"]):
        ui_data["coordinates"] = df.select(["x", "y", "z"]).to_numpy()
    elif all(col in df.columns for col in ["ra", "dec"]):
        # Will be converted to Cartesian in visualization components
        ui_data["properties"]["ra"] = df["ra"].to_numpy()
        ui_data["properties"]["dec"] = df["dec"].to_numpy()
        if "distance_pc" in df.columns:
            ui_data["properties"]["distance_pc"] = df["distance_pc"].to_numpy()
        elif "parallax" in df.columns:
            ui_data["properties"]["parallax"] = df["parallax"].to_numpy()

    # Extract other columns as properties
    for col in df.columns:
        if col not in ["x", "y", "z", "ra", "dec", "distance_pc", "parallax"]:
            ui_data["properties"][col] = df[col].to_numpy()

    return ui_data


def pyg_data_to_ui_data(data: Any) -> Dict[str, Any]:
    """Convert PyG Data object to UI data format.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dict with coordinates and properties
    """
    ui_data = {
        "coordinates": None,
        "properties": {},
        "metadata": {"source": "pyg", "has_edges": hasattr(data, "edge_index")},
    }

    # Extract node positions
    if hasattr(data, "pos"):
        ui_data["coordinates"] = data.pos.cpu().numpy()
    elif hasattr(data, "x") and data.x.shape[1] >= 3:
        ui_data["coordinates"] = data.x[:, :3].cpu().numpy()

    # Extract node features
    if hasattr(data, "x"):
        ui_data["properties"]["features"] = data.x.cpu().numpy()

    # Extract edge information
    if hasattr(data, "edge_index"):
        ui_data["properties"]["edge_index"] = data.edge_index.cpu().numpy()

    # Extract other attributes
    for key in data.keys:
        if key not in ["pos", "x", "edge_index", "batch"]:
            value = getattr(data, key)
            if torch.is_tensor(value):
                ui_data["properties"][key] = value.cpu().numpy()
            else:
                ui_data["properties"][key] = value

    return ui_data


def adapt_to_ui_data(data: Any) -> Dict[str, Any]:
    """Adapt any supported data format to UI data format.

    Args:
        data: TensorDict, DataFrame, PyG Data, or dict

    Returns:
        Dict with standardized UI data format
    """
    # Already in UI format
    if isinstance(data, dict) and "coordinates" in data:
        return data

    # Polars DataFrame
    if isinstance(data, pl.DataFrame):
        return dataframe_to_ui_data(data)

    # PyG Data
    if hasattr(data, "pos") and hasattr(data, "edge_index"):
        return pyg_data_to_ui_data(data)

    # TensorDict or similar
    if hasattr(data, "keys") or hasattr(data, "__getitem__"):
        return tensordict_to_ui_data(data)

    # Fallback
    return {
        "coordinates": None,
        "properties": {"data": data},
        "metadata": {"source": "unknown"},
    }
