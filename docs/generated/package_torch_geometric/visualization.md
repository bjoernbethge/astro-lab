# visualization Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.visualization`

## Description

Visualization package.

## Functions (2)

### `influence(model: torch.nn.modules.module.Module, src: torch.Tensor, *args: Any) -> torch.Tensor`

### `visualize_graph(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, path: Optional[str] = None, backend: Optional[str] = None, node_labels: Optional[List[str]] = None) -> Any`

Visualizes the graph given via :obj:`edge_index` and (optional)
:obj:`edge_weight`.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_weight (torch.Tensor, optional): The edge weights.
    path (str, optional): The path to where the plot is saved.
        If set to :obj:`None`, will visualize the plot on-the-fly.
        (default: :obj:`None`)
    backend (str, optional): The graph drawing backend to use for
        visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
        If set to :obj:`None`, will use the most appropriate
        visualization backend based on available system packages.
        (default: :obj:`None`)
    node_labels (List[str], optional): The labels/IDs of nodes.
        (default: :obj:`None`)

## Nested Submodules (1)

Each nested submodule is documented in a separate file:

### [graph](./visualization/graph.md)
Module: `torch_geometric.visualization.graph`

*Contains: 2 functions, 2 classes*
