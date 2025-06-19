# dropout

Part of `torch_geometric.experimental`
Module: `torch_geometric.utils.dropout`

## Functions (12)

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

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

### `dropout_adj(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, p: float = 0.5, force_undirected: bool = False, num_nodes: Optional[int] = None, training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Randomly drops edges from the adjacency matrix
:obj:`(edge_index, edge_attr)` with probability :obj:`p` using samples from
a Bernoulli distribution.

.. warning::

    :class:`~torch_geometric.utils.dropout_adj` is deprecated and will
    be removed in a future release.
    Use :class:`torch_geometric.utils.dropout_edge` instead.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    p (float, optional): Dropout probability. (default: :obj:`0.5`)
    force_undirected (bool, optional): If set to :obj:`True`, will either
        drop or keep both edges of an undirected edge.
        (default: :obj:`False`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
    >>> dropout_adj(edge_index, edge_attr)
    (tensor([[0, 1, 2, 3],
            [1, 2, 3, 2]]),
    tensor([1, 3, 5, 6]))

    >>> # The returned graph is kept undirected
    >>> dropout_adj(edge_index, edge_attr, force_undirected=True)
    (tensor([[0, 1, 2, 1, 2, 3],
            [1, 2, 3, 0, 1, 2]]),
    tensor([1, 3, 5, 1, 3, 5]))

### `dropout_edge(edge_index: torch.Tensor, p: float = 0.5, force_undirected: bool = False, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Randomly drops edges from the adjacency matrix
:obj:`edge_index` with probability :obj:`p` using samples from
a Bernoulli distribution.

The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
or index indicating which edges were retained, depending on the argument
:obj:`force_undirected`.

Args:
    edge_index (LongTensor): The edge indices.
    p (float, optional): Dropout probability. (default: :obj:`0.5`)
    force_undirected (bool, optional): If set to :obj:`True`, will either
        drop or keep both edges of an undirected edge.
        (default: :obj:`False`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

:rtype: (:class:`LongTensor`, :class:`BoolTensor` or :class:`LongTensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> edge_index, edge_mask = dropout_edge(edge_index)
    >>> edge_index
    tensor([[0, 1, 2, 2],
            [1, 2, 1, 3]])
    >>> edge_mask # masks indicating which edges are retained
    tensor([ True, False,  True,  True,  True, False])

    >>> edge_index, edge_id = dropout_edge(edge_index,
    ...                                    force_undirected=True)
    >>> edge_index
    tensor([[0, 1, 2, 1, 2, 3],
            [1, 2, 3, 0, 1, 2]])
    >>> edge_id # indices indicating which edges are retained
    tensor([0, 2, 4, 0, 2, 4])

### `dropout_node(edge_index: torch.Tensor, p: float = 0.5, num_nodes: Optional[int] = None, training: bool = True, relabel_nodes: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Randomly drops nodes from the adjacency matrix
:obj:`edge_index` with probability :obj:`p` using samples from
a Bernoulli distribution.

The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
indicating which edges were retained. (3) the node mask indicating
which nodes were retained.

Args:
    edge_index (LongTensor): The edge indices.
    p (float, optional): Dropout probability. (default: :obj:`0.5`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)
    relabel_nodes (bool, optional): If set to `True`, the resulting
        `edge_index` will be relabeled to hold consecutive indices
        starting from zero.

:rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
    >>> edge_index
    tensor([[0, 1],
            [1, 0]])
    >>> edge_mask
    tensor([ True,  True, False, False, False, False])
    >>> node_mask
    tensor([ True,  True, False, False])

### `dropout_path(edge_index: torch.Tensor, p: float = 0.2, walks_per_node: int = 1, walk_length: int = 3, num_nodes: Optional[int] = None, is_sorted: bool = False, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Drops edges from the adjacency matrix :obj:`edge_index`
based on random walks. The source nodes to start random walks from are
sampled from :obj:`edge_index` with probability :obj:`p`, following
a Bernoulli distribution.

The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
indicating which edges were retained.

Args:
    edge_index (LongTensor): The edge indices.
    p (float, optional): Sample probability. (default: :obj:`0.2`)
    walks_per_node (int, optional): The number of walks per node, same as
        :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`1`)
    walk_length (int, optional): The walk length, same as
        :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`3`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    is_sorted (bool, optional): If set to :obj:`True`, will expect
        :obj:`edge_index` to be already sorted row-wise.
        (default: :obj:`False`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

:rtype: (:class:`LongTensor`, :class:`BoolTensor`)

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> edge_index, edge_mask = dropout_path(edge_index)
    >>> edge_index
    tensor([[1, 2],
            [2, 3]])
    >>> edge_mask # masks indicating which edges are retained
    tensor([False, False,  True, False,  True, False])

### `filter_adj(row: torch.Tensor, col: torch.Tensor, edge_attr: Optional[torch.Tensor], mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`

### `is_compiling() -> bool`

Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
:meth:`torch.compile`.

### `maybe_num_nodes(edge_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch_geometric.typing.SparseTensor], num_nodes: Optional[int] = None) -> int`

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

### `subgraph(subset: Union[torch.Tensor, List[int]], edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, relabel_nodes: bool = False, num_nodes: Optional[int] = None, *, return_edge_mask: bool = False) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]`

Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
containing the nodes in :obj:`subset`.

Args:
    subset (LongTensor, BoolTensor or [int]): The nodes to keep.
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
        :obj:`edge_index` will be relabeled to hold consecutive indices
        starting from zero. (default: :obj:`False`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index) + 1`. (default: :obj:`None`)
    return_edge_mask (bool, optional): If set to :obj:`True`, will return
        the edge mask to filter out additional edge features.
        (default: :obj:`False`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
    ...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
    >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> subset = torch.tensor([3, 4, 5])
    >>> subgraph(subset, edge_index, edge_attr)
    (tensor([[3, 4, 4, 5],
            [4, 3, 5, 4]]),
    tensor([ 7.,  8.,  9., 10.]))

    >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
    (tensor([[3, 4, 4, 5],
            [4, 3, 5, 4]]),
    tensor([ 7.,  8.,  9., 10.]),
    tensor([False, False, False, False, False, False,  True,
            True,  True,  True,  False, False]))

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
