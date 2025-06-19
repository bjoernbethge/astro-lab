# graph

Part of `torch_geometric.visualization`
Module: `torch_geometric.visualization.graph`

## Functions (2)

### `has_graphviz() -> bool`

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

## Classes (2)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
