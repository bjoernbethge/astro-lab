# add_positional_encoding

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.add_positional_encoding`

## Functions (10)

### `add_node_attr(data: torch_geometric.data.data.Data, value: Any, attr_name: Optional[str] = None) -> torch_geometric.data.data.Data`

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

### `get_self_loop_attr(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> torch.Tensor`

Returns the edge features or weights of self-loops
:math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
graph given by :attr:`edge_index`. Edge features of missing self-loops not
present in :attr:`edge_index` will be filled with zeros. If
:attr:`edge_attr` is not given, it will be the vector of ones.

.. note::
    This operation is analogous to getting the diagonal elements of the
    dense adjacency matrix.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: :class:`Tensor`

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
    >>> get_self_loop_attr(edge_index, edge_weight)
    tensor([0.5000, 0.0000])

    >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
    tensor([0.5000, 0.0000, 0.0000, 0.0000])

### `is_torch_sparse_tensor(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is a
:class:`torch.sparse.Tensor` (in any sparse layout).

Args:
    src (Any): The input object to be checked.

### `scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None, reduce: str = 'sum') -> torch.Tensor`

Reduces all values from the :obj:`src` tensor at the indices
specified in the :obj:`index` tensor along a given dimension
:obj:`dim`. See the `documentation
<https://pytorch-scatter.readthedocs.io/en/latest/functions/
scatter.html>`__ of the :obj:`torch_scatter` package for more
information.

Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The index tensor.
    dim (int, optional): The dimension along which to index.
        (default: :obj:`0`)
    dim_size (int, optional): The size of the output tensor at
        dimension :obj:`dim`. If set to :obj:`None`, will create a
        minimal-sized output tensor according to
        :obj:`index.max() + 1`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation (:obj:`"sum"`,
        :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
        :obj:`"any"`). (default: :obj:`"sum"`)

### `to_edge_index(adj: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> Tuple[torch.Tensor, torch.Tensor]`

Converts a :class:`torch.sparse.Tensor` or a
:class:`torch_sparse.SparseTensor` to edge indices and edge attributes.

Args:
    adj (torch.sparse.Tensor or SparseTensor): The adjacency matrix.

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> adj = to_torch_coo_tensor(edge_index)
    >>> to_edge_index(adj)
    (tensor([[0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]]),
    tensor([1., 1., 1., 1., 1., 1.]))

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

### `to_torch_coo_tensor(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, size: Union[int, Tuple[Optional[int], Optional[int]], NoneType] = None, is_coalesced: bool = False) -> torch.Tensor`

Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a :class:`torch.sparse.Tensor` with layout
`torch.sparse_coo`.
See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): The edge attributes.
        (default: :obj:`None`)
    size (int or (int, int), optional): The size of the sparse matrix.
        If given as an integer, will create a quadratic sparse matrix.
        If set to :obj:`None`, will infer a quadratic sparse matrix based
        on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
    is_coalesced (bool): If set to :obj:`True`, will assume that
        :obj:`edge_index` is already coalesced and thus avoids expensive
        computation. (default: :obj:`False`)

:rtype: :class:`torch.sparse.Tensor`

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> to_torch_coo_tensor(edge_index)
    tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]]),
           values=tensor([1., 1., 1., 1., 1., 1.]),
           size=(4, 4), nnz=6, layout=torch.sparse_coo)

### `to_torch_csr_tensor(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, size: Union[int, Tuple[Optional[int], Optional[int]], NoneType] = None, is_coalesced: bool = False) -> torch.Tensor`

Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a :class:`torch.sparse.Tensor` with layout
`torch.sparse_csr`.
See :meth:`~torch_geometric.utils.to_edge_index` for the reverse operation.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): The edge attributes.
        (default: :obj:`None`)
    size (int or (int, int), optional): The size of the sparse matrix.
        If given as an integer, will create a quadratic sparse matrix.
        If set to :obj:`None`, will infer a quadratic sparse matrix based
        on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
    is_coalesced (bool): If set to :obj:`True`, will assume that
        :obj:`edge_index` is already coalesced and thus avoids expensive
        computation. (default: :obj:`False`)

:rtype: :class:`torch.sparse.Tensor`

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> to_torch_csr_tensor(edge_index)
    tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
           col_indices=tensor([1, 0, 2, 1, 3, 2]),
           values=tensor([1., 1., 1., 1., 1., 1.]),
           size=(4, 4), nnz=6, layout=torch.sparse_csr)

## Classes (6)

### `AddLaplacianEigenvectorPE`

Adds the Laplacian eigenvector positional encoding from the
`"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
paper to the given graph
(functional name: :obj:`add_laplacian_eigenvector_pe`).

Args:
    k (int): The number of non-trivial eigenvectors to consider.
    attr_name (str, optional): The attribute name of the data object to add
        positional encodings to. If set to :obj:`None`, will be
        concatenated to :obj:`data.x`.
        (default: :obj:`"laplacian_eigenvector_pe"`)
    is_undirected (bool, optional): If set to :obj:`True`, this transform
        expects undirected graphs as input, and can hence speed up the
        computation of eigenvectors. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
        :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
        :attr:`is_undirected` is :obj:`True`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `AddRandomWalkPE`

Adds the random walk positional encoding from the `"Graph Neural
Networks with Learnable Structural and Positional Representations"
<https://arxiv.org/abs/2110.07875>`_ paper to the given graph
(functional name: :obj:`add_random_walk_pe`).

Args:
    walk_length (int): The number of random walk steps.
    attr_name (str, optional): The attribute name of the data object to add
        positional encodings to. If set to :obj:`None`, will be
        concatenated to :obj:`data.x`.
        (default: :obj:`"random_walk_pe"`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
