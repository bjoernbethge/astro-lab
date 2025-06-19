# half_hop

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.half_hop`

## Functions (1)

### `functional_transform(name: str) -> Callable`

## Classes (3)

### `BaseTransform`

An abstract base class for writing transforms.

Transforms are a general way to modify and customize
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects, either by implicitly
passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or
by applying them explicitly to individual
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    dataset = TUDataset(path, name='MUTAG', transform=transform)
    data = dataset[0]  # Implicitly transform data on every access.

    data = TUDataset(path, name='MUTAG')[0]
    data = transform(data)  # Explicitly transform data.

#### Methods

- **`forward(self, data: Any) -> Any`**

### `Data`

A data object describing a homogeneous graph.
The data object can hold node-level, link-level and graph-level attributes.
In general, :class:`~torch_geometric.data.Data` tries to mimic the
behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/get_started/
introduction.html#data-handling-of-graphs>`__ for the accompanying
tutorial.

.. code-block:: python

    from torch_geometric.data import Data

    data = Data(x=x, edge_index=edge_index, ...)

    # Add additional arguments to `data`:
    data.train_idx = torch.tensor([...], dtype=torch.long)
    data.test_mask = torch.tensor([...], dtype=torch.bool)

    # Analyzing the graph structure:
    data.num_nodes
    >>> 23

    data.is_directed()
    >>> False

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Graph connectivity in COO format
        with shape :obj:`[2, num_edges]`. (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_edges]` or :obj:`[num_nodes]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

### `HalfHop`

The graph upsampling augmentation from the
`"Half-Hop: A Graph Upsampling Approach for Slowing Down Message Passing"
<https://openreview.net/forum?id=lXczFIwQkv>`_ paper.
The graph is augmented by adding artificial slow nodes between neighbors
to slow down message propagation. (functional name: :obj:`half_hop`).

.. note::
    :class:`HalfHop` augmentation is not supported if :obj:`data` has
    :attr:`edge_weight` or :attr:`edge_attr`.

Args:
    alpha (float, optional): The interpolation factor
        used to compute slow node features
        :math:`x = \alpha*x_src + (1-\alpha)*x_dst` (default: :obj:`0.5`)
    p (float, optional): The probability of half-hopping
        an edge. (default: :obj:`1.0`)

.. code-block:: python

    import torch_geometric.transforms as T

    transform = T.HalfHop(alpha=0.5)
    data = transform(data)  # Apply transformation.
    out = model(data.x, data.edge_index)  # Feed-forward.
    out = out[~data.slow_node_mask]  # Get rid of slow nodes.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**
