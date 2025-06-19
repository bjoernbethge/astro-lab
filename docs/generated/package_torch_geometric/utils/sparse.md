# sparse

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.sparse`

## Functions (18)

### `cat(tensors: List[torch.Tensor], dim: Union[int, Tuple[int, int]]) -> torch.Tensor`

### `cat_coo(tensors: List[torch.Tensor], dim: Union[int, Tuple[int, int]]) -> torch.Tensor`

### `cat_csc(tensors: List[torch.Tensor], dim: Union[int, Tuple[int, int]]) -> torch.Tensor`

### `cat_csr(tensors: List[torch.Tensor], dim: Union[int, Tuple[int, int]]) -> torch.Tensor`

### `coalesce(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'sum', is_sorted: bool = False, sort_by_row: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
Duplicate entries in :obj:`edge_attr` are merged by scattering them
together according to the given :obj:`reduce` option.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
        or multi-dimensional edge features.
        If given as a list, will re-shuffle and remove duplicates for all
        its entries. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
    is_sorted (bool, optional): If set to :obj:`True`, will expect
        :obj:`edge_index` to be already sorted row-wise.
    sort_by_row (bool, optional): If set to :obj:`False`, will sort
        :obj:`edge_index` column-wise.

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Example:
    >>> edge_index = torch.tensor([[1, 1, 2, 3],
    ...                            [3, 3, 1, 2]])
    >>> edge_attr = torch.tensor([1., 1., 1., 1.])
    >>> coalesce(edge_index)
    tensor([[1, 2, 3],
            [3, 1, 2]])

    >>> # Sort `edge_index` column-wise
    >>> coalesce(edge_index, sort_by_row=False)
    tensor([[2, 3, 1],
            [1, 2, 3]])

    >>> coalesce(edge_index, edge_attr)
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([2., 1., 1.]))

    >>> # Use 'mean' operation to merge edge features
    >>> coalesce(edge_index, edge_attr, reduce='mean')
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([1., 1., 1.]))

### `cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor`

Returns the cumulative sum of elements of :obj:`x`.
In contrast to :meth:`torch.cumsum`, prepends the output with zero.

Args:
    x (torch.Tensor): The input tensor.
    dim (int, optional): The dimension to do the operation over.
        (default: :obj:`0`)

Example:
    >>> x = torch.tensor([2, 4, 1])
    >>> cumsum(x)
    tensor([0, 2, 6, 7])

### `dense_to_sparse(adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Converts a dense adjacency matrix to a sparse adjacency matrix defined
by edge indices and edge attributes.

Args:
    adj (torch.Tensor): The dense adjacency matrix of shape
        :obj:`[num_nodes, num_nodes]` or
        :obj:`[batch_size, num_nodes, num_nodes]`.
    mask (torch.Tensor, optional): A boolean tensor of shape
        :obj:`[batch_size, num_nodes]` holding information about which
        nodes are in each example are valid. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> # For a single adjacency matrix:
    >>> adj = torch.tensor([[3, 1],
    ...                     [2, 0]])
    >>> dense_to_sparse(adj)
    (tensor([[0, 0, 1],
            [0, 1, 0]]),
    tensor([3, 1, 2]))

    >>> # For two adjacency matrixes:
    >>> adj = torch.tensor([[[3, 1],
    ...                      [2, 0]],
    ...                     [[0, 1],
    ...                      [0, 2]]])
    >>> dense_to_sparse(adj)
    (tensor([[0, 0, 1, 2, 3],
            [0, 1, 0, 3, 3]]),
    tensor([3, 1, 2, 1, 2]))

    >>> # First graph with two nodes, second with three:
    >>> adj = torch.tensor([[
    ...         [3, 1, 0],
    ...         [2, 0, 0],
    ...         [0, 0, 0]
    ...     ], [
    ...         [0, 1, 0],
    ...         [0, 2, 3],
    ...         [0, 5, 0]
    ...     ]])
    >>> mask = torch.tensor([
    ...         [True, True, False],
    ...         [True, True, True]
    ...     ])
    >>> dense_to_sparse(adj, mask)
    (tensor([[0, 0, 1, 2, 3, 3, 4],
            [0, 1, 0, 3, 3, 4, 3]]),
    tensor([3, 1, 2, 1, 2, 3, 5]))

### `get_sparse_diag(size: int, fill_value: float = 1.0, layout: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

### `index2ptr(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

### `is_sparse(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is of type
:class:`torch.sparse.Tensor` (in any sparse layout) or of type
:class:`torch_sparse.SparseTensor`.

Args:
    src (Any): The input object to be checked.

### `is_torch_sparse_tensor(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is a
:class:`torch.sparse.Tensor` (in any sparse layout).

Args:
    src (Any): The input object to be checked.

### `ptr2index(ptr: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor`

### `set_sparse_value(adj: torch.Tensor, value: torch.Tensor) -> torch.Tensor`

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

### `to_torch_csc_tensor(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, size: Union[int, Tuple[Optional[int], Optional[int]], NoneType] = None, is_coalesced: bool = False) -> torch.Tensor`

Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a :class:`torch.sparse.Tensor` with layout
`torch.sparse_csc`.
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
    >>> to_torch_csc_tensor(edge_index)
    tensor(ccol_indices=tensor([0, 1, 3, 5, 6]),
           row_indices=tensor([1, 0, 2, 1, 3, 2]),
           values=tensor([1., 1., 1., 1., 1., 1.]),
           size=(4, 4), nnz=6, layout=torch.sparse_csc)

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

### `to_torch_sparse_tensor(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, size: Union[int, Tuple[Optional[int], Optional[int]], NoneType] = None, is_coalesced: bool = False, layout: torch.layout = torch.sparse_coo) -> torch.Tensor`

Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a :class:`torch.sparse.Tensor` with custom :obj:`layout`.
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
    layout (torch.layout, optional): The layout of the output sparse tensor
        (:obj:`torch.sparse_coo`, :obj:`torch.sparse_csr`,
        :obj:`torch.sparse_csc`). (default: :obj:`torch.sparse_coo`)

:rtype: :class:`torch.sparse.Tensor`

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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
