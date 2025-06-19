# one_hot_degree

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.one_hot_degree`

## Functions (3)

### `degree(index: torch.Tensor, num_nodes: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor`

Computes the (unweighted) degree of a given one-dimensional index
tensor.

Args:
    index (LongTensor): Index tensor.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dtype (:obj:`torch.dtype`, optional): The desired data type of the
        returned tensor.

:rtype: :class:`Tensor`

Example:
    >>> row = torch.tensor([0, 1, 0, 2, 0])
    >>> degree(row, dtype=torch.long)
    tensor([3, 1, 1])

### `functional_transform(name: str) -> Callable`

### `one_hot(index: torch.Tensor, num_classes: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor`

Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
encoded representation of it with shape :obj:`[*, num_classes]` that has
zeros everywhere except where the index of last dimension matches the
corresponding value of the input tensor, in which case it will be :obj:`1`.

.. note::
    This is a more memory-efficient version of
    :meth:`torch.nn.functional.one_hot` as you can customize the output
    :obj:`dtype`.

Args:
    index (torch.Tensor): The one-dimensional input tensor.
    num_classes (int, optional): The total number of classes. If set to
        :obj:`None`, the number of classes will be inferred as one greater
        than the largest class value in the input tensor.
        (default: :obj:`None`)
    dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.

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

### `OneHotDegree`

Adds the node degree as one hot encodings to the node features
(functional name: :obj:`one_hot_degree`).

Args:
    max_degree (int): Maximum degree.
    in_degree (bool, optional): If set to :obj:`True`, will compute the
        in-degree of nodes instead of the out-degree.
        (default: :obj:`False`)
    cat (bool, optional): Concat node degrees to node features instead
        of replacing them. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**
