# to_sparse_tensor

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.to_sparse_tensor`

## Functions (4)

### `functional_transform(name: str) -> Callable`

### `sort_edge_index(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, sort_by_row: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Row-wise sorts :obj:`edge_index`.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
        or multi-dimensional edge features.
        If given as a list, will re-shuffle and remove duplicates for all
        its entries. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    sort_by_row (bool, optional): If set to :obj:`False`, will sort
        :obj:`edge_index` column-wise/by destination node.
        (default: :obj:`True`)

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Examples:
    >>> edge_index = torch.tensor([[2, 1, 1, 0],
                            [1, 2, 0, 1]])
    >>> edge_attr = torch.tensor([[1], [2], [3], [4]])
    >>> sort_edge_index(edge_index)
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]])

    >>> sort_edge_index(edge_index, edge_attr)
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([[4],
            [3],
            [2],
            [1]]))

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

### `HeteroData`

A data object describing a heterogeneous graph, holding multiple node
and/or edge types in disjunct storage objects.
Storage objects can hold either node-level, link-level or graph-level
attributes.
In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
behavior of a regular **nested** :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block::

    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Create two node types "paper" and "author" holding a feature matrix:
    data['paper'].x = torch.randn(num_papers, num_paper_features)
    data['author'].x = torch.randn(num_authors, num_authors_features)

    # Create an edge type "(author, writes, paper)" and building the
    # graph connectivity:
    data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

    data['paper'].num_nodes
    >>> 23

    data['author', 'writes', 'paper'].num_edges
    >>> 52

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Note that there exists multiple ways to create a heterogeneous graph data,
*e.g.*:

* To initialize a node of type :obj:`"paper"` holding a node feature
  matrix :obj:`x_paper` named :obj:`x`:

  .. code-block:: python

    from torch_geometric.data import HeteroData

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['paper'].x = x_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(paper={ 'x': x_paper })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({'paper': { 'x': x_paper }})

* To initialize an edge from source node type :obj:`"author"` to
  destination node type :obj:`"paper"` with relation type :obj:`"writes"`
  holding a graph connectivity matrix :obj:`edge_index_author_paper` named
  :obj:`edge_index`:

  .. code-block:: python

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(author__writes__paper={
        'edge_index': edge_index_author_paper
    })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({
        ('author', 'writes', 'paper'):
        { 'edge_index': edge_index_author_paper }
    })

#### Methods

- **`stores_as(self, data: Self)`**

- **`node_items(self) -> List[Tuple[str, torch_geometric.data.storage.NodeStorage]]`**
  Returns a list of node type and node storage pairs.

- **`edge_items(self) -> List[Tuple[Tuple[str, str, str], torch_geometric.data.storage.EdgeStorage]]`**
  Returns a list of edge type and edge storage pairs.

### `SparseTensor`

#### Methods

- **`size(self, dim: int) -> int`**

- **`nnz(self) -> int`**

- **`is_cuda(self) -> bool`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `ToSparseTensor`

Converts the :obj:`edge_index` attributes of a homogeneous or
heterogeneous data object into a **transposed**
:class:`torch_sparse.SparseTensor` or :pytorch:`PyTorch`
:class:`torch.sparse.Tensor` object with key :obj:`adj_t`
(functional name: :obj:`to_sparse_tensor`).

.. note::

    In case of composing multiple transforms, it is best to convert the
    :obj:`data` object via :class:`ToSparseTensor` as late as possible,
    since there exist some transforms that are only able to operate on
    :obj:`data.edge_index` for now.

Args:
    attr (str, optional): The name of the attribute to add as a value to
        the :class:`~torch_sparse.SparseTensor` or
        :class:`torch.sparse.Tensor` object (if present).
        (default: :obj:`edge_weight`)
    remove_edge_index (bool, optional): If set to :obj:`False`, the
        :obj:`edge_index` tensor will not be removed.
        (default: :obj:`True`)
    fill_cache (bool, optional): If set to :obj:`True`, will fill the
        underlying :class:`torch_sparse.SparseTensor` cache (if used).
        (default: :obj:`True`)
    layout (torch.layout, optional): Specifies the layout of the returned
        sparse tensor (:obj:`None`, :obj:`torch.sparse_coo` or
        :obj:`torch.sparse_csr`).
        If set to :obj:`None` and the :obj:`torch_sparse` dependency is
        installed, will convert :obj:`edge_index` into a
        :class:`torch_sparse.SparseTensor` object.
        If set to :obj:`None` and the :obj:`torch_sparse` dependency is
        not installed, will convert :obj:`edge_index` into a
        :class:`torch.sparse.Tensor` object with layout
        :obj:`torch.sparse_csr`. (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**
