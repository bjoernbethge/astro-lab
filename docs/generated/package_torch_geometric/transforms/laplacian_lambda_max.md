# laplacian_lambda_max

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.laplacian_lambda_max`

## Functions (3)

### `functional_transform(name: str) -> Callable`

### `get_laplacian(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, normalization: Optional[str] = None, dtype: Optional[torch.dtype] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Computes the graph Laplacian of the graph given by :obj:`edge_index`
and optional :obj:`edge_weight`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_weight (Tensor, optional): One-dimensional edge weights.
        (default: :obj:`None`)
    normalization (str, optional): The normalization scheme for the graph
        Laplacian (default: :obj:`None`):

        1. :obj:`None`: No normalization
        :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

        2. :obj:`"sym"`: Symmetric normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}`

        3. :obj:`"rw"`: Random-walk normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
    dtype (torch.dtype, optional): The desired data type of returned tensor
        in case :obj:`edge_weight=None`. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1, 2],
    ...                            [1, 0, 2, 1]])
    >>> edge_weight = torch.tensor([1., 2., 2., 4.])

    >>> # No normalization
    >>> lap = get_laplacian(edge_index, edge_weight)

    >>> # Symmetric normalization
    >>> lap_sym = get_laplacian(edge_index, edge_weight,
                                normalization='sym')

    >>> # Random-walk normalization
    >>> lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')

### `to_scipy_sparse_matrix(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> Any`

Converts a graph given by edge indices and edge attributes to a scipy
sparse matrix.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> to_scipy_sparse_matrix(edge_index)
    <4x4 sparse matrix of type '<class 'numpy.float32'>'
        with 6 stored elements in COOrdinate format>

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

### `LaplacianLambdaMax`

Computes the highest eigenvalue of the graph Laplacian given by
:meth:`torch_geometric.utils.get_laplacian`
(functional name: :obj:`laplacian_lambda_max`).

Args:
    normalization (str, optional): The normalization scheme for the graph
        Laplacian (default: :obj:`None`):

        1. :obj:`None`: No normalization
        :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

        2. :obj:`"sym"`: Symmetric normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}`

        3. :obj:`"rw"`: Random-walk normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
    is_undirected (bool, optional): If set to :obj:`True`, this transform
        expects undirected graphs as input, and can hence speed up the
        computation of the largest eigenvalue. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**
