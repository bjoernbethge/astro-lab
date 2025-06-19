# Graph_Utils Module

Auto-generated documentation for `astro_lab.data.graph_utils`

## Functions

### convert_hetero_to_homo(hetero_data: torch_geometric.data.hetero_data.HeteroData) -> torch_geometric.data.data.Data

Convert HeteroData to homogeneous Data.

Args:
    hetero_data: HeteroData object

Returns:
    Homogeneous Data object

### create_graph_batch(node_features: torch.Tensor, edge_indices: list, edge_attributes: Optional[list] = None) -> torch_geometric.data.batch.Batch

Create PyG Batch from components.

Args:
    node_features: Node feature tensor
    edge_indices: List of edge index tensors
    edge_attributes: Optional edge attributes

Returns:
    PyG Batch object

### extract_graph_data(data: Union[torch_geometric.data.data.Data, torch_geometric.data.batch.Batch, torch_geometric.data.hetero_data.HeteroData]) -> dict

Extract standard graph data components.

Args:
    data: PyG Data object

Returns:
    Dictionary with extracted components

### validate_graph_data(data: Union[torch_geometric.data.data.Data, torch_geometric.data.batch.Batch]) -> bool

Validate PyG graph data structure.

Args:
    data: PyG Data object

Returns:
    True if valid
