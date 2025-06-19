# loop

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.loop`

## Functions (14)

### `add_remaining_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, fill_value: Union[float, torch.Tensor, str, NoneType] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Adds remaining self-loop :math:`(i,i) \in \mathcal{E}` to every node
:math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(:obj:`edge_attr != None`), edge features of non-existing self-loops will
be added according to :obj:`fill_value`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features. (default: :obj:`None`)
    fill_value (float or Tensor or str, optional): The way to generate
        edge features of self-loops (in case :obj:`edge_attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Example:
    >>> edge_index = torch.tensor([[0, 1],
    ...                            [1, 0]])
    >>> edge_weight = torch.tensor([0.5, 0.5])
    >>> add_remaining_self_loops(edge_index, edge_weight)
    (tensor([[0, 1, 0, 1],
            [1, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 1.0000, 1.0000]))

### `add_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, fill_value: Union[float, torch.Tensor, str, NoneType] = None, num_nodes: Union[int, Tuple[int, int], NoneType] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
:math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(:obj:`edge_attr != None`), edge features of self-loops will be added
according to :obj:`fill_value`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features. (default: :obj:`None`)
    fill_value (float or Tensor or str, optional): The way to generate
        edge features of self-loops (in case :obj:`edge_attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    num_nodes (int or Tuple[int, int], optional): The number of nodes,
        *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
        (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> edge_weight = torch.tensor([0.5, 0.5, 0.5])
    >>> add_self_loops(edge_index)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    None)

    >>> add_self_loops(edge_index, edge_weight)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 1.0000, 1.0000]))

    >>> # edge features of self-loops are filled by constant `2.0`
    >>> add_self_loops(edge_index, edge_weight,
    ...                fill_value=2.)
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 2.0000, 2.0000]))

    >>> # Use 'add' operation to merge edge features for self-loops
    >>> add_self_loops(edge_index, edge_weight,
    ...                fill_value='add')
    (tensor([[0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1]]),
    tensor([0.5000, 0.5000, 0.5000, 1.0000, 0.5000]))

### `compute_loop_attr(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int, is_sparse: bool, fill_value: Union[float, torch.Tensor, str, NoneType] = None) -> torch.Tensor`

### `contains_self_loops(edge_index: torch.Tensor) -> bool`

Returns :obj:`True` if the graph given by :attr:`edge_index` contains
self-loops.

Args:
    edge_index (LongTensor): The edge indices.

:rtype: bool

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> contains_self_loops(edge_index)
    True

    >>> edge_index = torch.tensor([[0, 1, 1],
    ...                            [1, 0, 2]])
    >>> contains_self_loops(edge_index)
    False

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

### `maybe_num_nodes(edge_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch_geometric.typing.SparseTensor], num_nodes: Optional[int] = None) -> int`

### `overload(func)`

### `remove_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Removes every self-loop in the graph given by :attr:`edge_index`, so
that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Example:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> edge_attr = [[1, 2], [3, 4], [5, 6]]
    >>> edge_attr = torch.tensor(edge_attr)
    >>> remove_self_loops(edge_index, edge_attr)
    (tensor([[0, 1],
            [1, 0]]),
    tensor([[1, 2],
            [3, 4]]))

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

### `segregate_self_loops(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]`

Segregates self-loops from the graph.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`, :class:`LongTensor`,
    :class:`Tensor`)

Example:
    >>> edge_index = torch.tensor([[0, 0, 1],
    ...                            [0, 1, 0]])
    >>> (edge_index, edge_attr,
    ...  loop_edge_index,
    ...  loop_edge_attr) = segregate_self_loops(edge_index)
    >>>  loop_edge_index
    tensor([[0],
            [0]])

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

## Classes (2)

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
