"""
Graph Data Utilities für PyTorch Geometric

Diese Utilities gehören ins data Modul:
- Graph construction
- Data type handling
- Data validation
- Format conversion
"""

from typing import Optional, Union

import torch
from torch_geometric.data import Batch, Data, HeteroData


def extract_graph_data(data: Union[Data, Batch, HeteroData]) -> dict:
    """
    Extract standard graph data components.

    Args:
        data: PyG Data object

    Returns:
        Dictionary with extracted components
    """
    if hasattr(data, "x"):
        # Standard PyG Data/Batch object
        return {
            "x": getattr(data, "x"),
            "edge_index": getattr(data, "edge_index"),
            "batch": getattr(data, "batch", None),
            "edge_attr": getattr(data, "edge_attr", None),
        }
    else:
        # Handle tensor input as fallback
        if isinstance(data, torch.Tensor):
            return {
                "x": data,
                "edge_index": torch.empty((2, 0), dtype=torch.long, device=data.device),
                "batch": None,
                "edge_attr": None,
            }
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


def create_graph_batch(
    node_features: torch.Tensor,
    edge_indices: list,
    edge_attributes: Optional[list] = None,
) -> Batch:
    """
    Create PyG Batch from components.

    Args:
        node_features: Node feature tensor
        edge_indices: List of edge index tensors
        edge_attributes: Optional edge attributes

    Returns:
        PyG Batch object
    """
    data_list = []

    for i, edge_index in enumerate(edge_indices):
        # Create individual Data object
        data_obj = Data(
            x=node_features[i] if node_features.dim() == 3 else node_features,
            edge_index=edge_index,
        )

        if edge_attributes is not None and i < len(edge_attributes):
            data_obj.edge_attr = edge_attributes[i]

        data_list.append(data_obj)

    return Batch.from_data_list(data_list)


def validate_graph_data(data: Union[Data, Batch]) -> bool:
    """
    Validate PyG graph data structure.

    Args:
        data: PyG Data object

    Returns:
        True if valid
    """
    try:
        # Check required attributes
        if not hasattr(data, "x") or not hasattr(data, "edge_index"):
            return False

        # Check tensor shapes
        x = getattr(data, "x")
        edge_index = getattr(data, "edge_index")

        if x.dim() != 2:
            return False

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            return False

        # Check edge index validity
        if edge_index.max() >= x.size(0):
            return False

        return True

    except Exception:
        return False


def convert_hetero_to_homo(hetero_data: HeteroData) -> Data:
    """
    Convert HeteroData to homogeneous Data.

    Args:
        hetero_data: HeteroData object

    Returns:
        Homogeneous Data object
    """
    # This is a simplified conversion - real implementation would be more complex
    node_features = []
    edge_indices = []

    # Concatenate all node types
    for node_type, features in hetero_data.x_dict.items():
        node_features.append(features)

    # Concatenate all edge types (with proper offset)
    node_offset = 0
    for edge_type, edge_index in hetero_data.edge_index_dict.items():
        adjusted_edge_index = edge_index + node_offset
        edge_indices.append(adjusted_edge_index)
        # Update offset based on source node type size
        src_type = edge_type[0]
        if src_type in hetero_data.x_dict:
            node_offset += hetero_data.x_dict[src_type].size(0)

    # Combine all
    combined_x = torch.cat(node_features, dim=0)
    combined_edge_index = torch.cat(edge_indices, dim=1)

    return Data(x=combined_x, edge_index=combined_edge_index)


__all__ = [
    "extract_graph_data",
    "create_graph_batch",
    "validate_graph_data",
    "convert_hetero_to_homo",
]
