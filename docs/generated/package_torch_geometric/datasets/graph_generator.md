# graph_generator

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.graph_generator`

## Classes (5)

### `BAGraph`

Generates random Barabasi-Albert (BA) graphs.
See :meth:`~torch_geometric.utils.barabasi_albert_graph` for more
information.

Args:
    num_nodes (int): The number of nodes.
    num_edges (int): The number of edges from a new node to existing nodes.

### `ERGraph`

Generates random Erdos-Renyi (ER) graphs.
See :meth:`~torch_geometric.utils.erdos_renyi_graph` for more information.

Args:
    num_nodes (int): The number of nodes.
    edge_prob (float): Probability of an edge.

### `GraphGenerator`

An abstract base class for generating synthetic graphs.

#### Methods

- **`resolve(query: Any, *args: Any, **kwargs: Any) -> 'GraphGenerator'`**

### `GridGraph`

Generates two-dimensional grid graphs.
See :meth:`~torch_geometric.utils.grid` for more information.

Args:
    height (int): The height of the grid.
    width (int): The width of the grid.
    dtype (:obj:`torch.dtype`, optional): The desired data type of the
        returned position tensor. (default: :obj:`None`)

### `TreeGraph`

Generates tree graphs.

Args:
    depth (int): The depth of the tree.
    branch (int, optional): The branch size of the tree.
        (default: :obj:`2`)
    undirected (bool, optional): If set to :obj:`True`, the tree graph will
        be undirected. (default: :obj:`False`)
