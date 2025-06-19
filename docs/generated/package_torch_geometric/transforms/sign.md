# sign

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.sign`

## Functions (2)

### `functional_transform(name: str) -> Callable`

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

## Classes (4)

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

### `EdgeIndex`

A COO :obj:`edge_index` tensor with additional (meta)data attached.

:class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
Edges are given as pairwise source and destination node indices in sparse
COO format.

While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`sparse_size`: The underlying sparse matrix size
* :obj:`sort_order`: The sort order (if present), either by row or column.
* :obj:`is_undirected`: Whether edges are bidirectional.

Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
in case its representation is sorted, such as its :obj:`rowptr` or
:obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
versa.
Caches are filled based on demand (*e.g.*, when calling
:meth:`EdgeIndex.sort_by`), or when explicitly requested via
:meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import EdgeIndex

    edge_index = EdgeIndex(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]]
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
        device='cpu',
    )
    >>> EdgeIndex([[0, 1, 1, 2],
    ...            [1, 0, 2, 1]])
    assert edge_index.is_sorted_by_row
    assert edge_index.is_undirected

    # Flipping order:
    edge_index = edge_index.flip(0)
    >>> EdgeIndex([[1, 0, 2, 1],
    ...            [0, 1, 1, 2]])
    assert edge_index.is_sorted_by_col
    assert edge_index.is_undirected

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    edge_index = edge_index[:, mask]
    >>> EdgeIndex([[1, 0, 2],
    ...            [0, 1, 1]])
    assert edge_index.is_sorted_by_col
    assert not edge_index.is_undirected

    # Sparse-Dense Matrix Multiplication:
    out = edge_index.flip(0) @ torch.randn(3, 16)
    assert out.size() == (3, 16)

#### Methods

- **`validate(self) -> 'EdgeIndex'`**
  Validates the :class:`EdgeIndex` representation.

- **`sparse_size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**
  The size of the underlying sparse matrix.

- **`get_sparse_size(self, dim: Optional[int] = None) -> Union[torch.Size, int]`**
  The size of the underlying sparse matrix.

### `SIGN`

The Scalable Inception Graph Neural Network module (SIGN) from the
`"SIGN: Scalable Inception Graph Neural Networks"
<https://arxiv.org/abs/2004.11198>`_ paper (functional name: :obj:`sign`),
which precomputes the fixed representations.

.. math::
    \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

for :math:`i \in \{ 1, \ldots, K \}` and saves them in
:obj:`data.x1`, :obj:`data.x2`, ...

.. note::

    Since intermediate node representations are pre-computed, this operator
    is able to scale well to large graphs via classic mini-batching.
    For an example of using SIGN, see `examples/sign.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    sign.py>`_.

Args:
    K (int): The number of hops/layer.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**
