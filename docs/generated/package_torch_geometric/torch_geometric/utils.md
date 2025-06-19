# utils

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.utils`

## Description

Utility package.

## Functions (89)

### `add_random_edge(edge_index: torch.Tensor, p: float = 0.5, force_undirected: bool = False, num_nodes: Union[int, Tuple[int, int], NoneType] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Randomly adds edges to :obj:`edge_index`.

The method returns (1) the retained :obj:`edge_index`, (2) the added
edge indices.

Args:
    edge_index (LongTensor): The edge indices.
    p (float): Ratio of added edges to the existing edges.
        (default: :obj:`0.5`)
    force_undirected (bool, optional): If set to :obj:`True`,
        added edges will be undirected.
        (default: :obj:`False`)
    num_nodes (int, Tuple[int], optional): The overall number of nodes,
        *i.e.* :obj:`max_val + 1`, or the number of source and
        destination nodes, *i.e.* :obj:`(max_src_val + 1, max_dst_val + 1)`
        of :attr:`edge_index`. (default: :obj:`None`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

:rtype: (:class:`LongTensor`, :class:`LongTensor`)

Examples:
    >>> # Standard case
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
    ...                            [1, 0, 2, 1, 3, 2]])
    >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
    >>> edge_index
    tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
            [1, 0, 2, 1, 3, 2, 0, 2, 1]])
    >>> added_edges
    tensor([[2, 1, 3],
            [0, 2, 1]])

    >>> # The returned graph is kept undirected
    >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
    ...                                           force_undirected=True)
    >>> edge_index
    tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
            [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
    >>> added_edges
    tensor([[2, 1, 3, 0, 2, 1],
            [0, 2, 1, 2, 1, 3]])

    >>> # For bipartite graphs
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    ...                            [2, 3, 1, 4, 2, 1]])
    >>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
    ...                                           num_nodes=(6, 5))
    >>> edge_index
    tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
            [2, 3, 1, 4, 2, 1, 1, 3, 2]])
    >>> added_edges
    tensor([[3, 4, 1],
            [1, 3, 2]])

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

### `assortativity(edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> float`

The degree assortativity coefficient from the
`"Mixing patterns in networks"
<https://arxiv.org/abs/cond-mat/0209450>`_ paper.
Assortativity in a network refers to the tendency of nodes to
connect with other similar nodes over dissimilar nodes.
It is computed from Pearson correlation coefficient of the node degrees.

Args:
    edge_index (Tensor or SparseTensor): The graph connectivity.

Returns:
    The value of the degree assortativity coefficient for the input
    graph :math:`\in [-1, 1]`

Example:
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 2],
    ...                            [1, 2, 0, 1, 3]])
    >>> assortativity(edge_index)
    -0.666667640209198

### `barabasi_albert_graph(num_nodes: int, num_edges: int) -> torch.Tensor`

Returns the :obj:`edge_index` of a Barabasi-Albert preferential
attachment model, where a graph of :obj:`num_nodes` nodes grows by
attaching new nodes with :obj:`num_edges` edges that are preferentially
attached to existing nodes with high degree.

Args:
    num_nodes (int): The number of nodes.
    num_edges (int): The number of edges from a new node to existing nodes.

Example:
    >>> barabasi_albert_graph(num_nodes=4, num_edges=3)
    tensor([[0, 0, 0, 1, 1, 2, 2, 3],
            [1, 2, 3, 0, 2, 0, 1, 0]])

### `batched_negative_sampling(edge_index: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], num_neg_samples: Optional[int] = None, method: str = 'sparse', force_undirected: bool = False) -> torch.Tensor`

Samples random negative edges of multiple graphs given by
:attr:`edge_index` and :attr:`batch`.

Args:
    edge_index (LongTensor): The edge indices.
    batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph connecting two different node types.
    num_neg_samples (int, optional): The number of negative samples to
        return. If set to :obj:`None`, will try to return a negative edge
        for every positive edge. (default: :obj:`None`)
    method (str, optional): The method to use for negative sampling,
        *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
        This is a memory/runtime trade-off.
        :obj:`"sparse"` will work on any graph of any size, while
        :obj:`"dense"` can perform faster true-negative checks.
        (default: :obj:`"sparse"`)
    force_undirected (bool, optional): If set to :obj:`True`, sampled
        negative edges will be undirected. (default: :obj:`False`)

:rtype: LongTensor

Examples:
    >>> # Standard usage
    >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
    >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
    >>> edge_index
    tensor([[0, 0, 1, 2, 4, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6, 7]])
    >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    >>> batched_negative_sampling(edge_index, batch)
    tensor([[3, 1, 3, 2, 7, 7, 6, 5],
            [2, 0, 1, 1, 5, 6, 4, 4]])

    >>> # For bipartite graph
    >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
    >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
    >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
    >>> edge_index = torch.cat([edge_index1, edge_index2,
    ...                         edge_index3], dim=1)
    >>> edge_index
    tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
    >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
    >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    >>> batched_negative_sampling(edge_index,
    ...                           (src_batch, dst_batch))
    tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
            [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])

### `bipartite_subgraph(subset: Union[Tuple[torch.Tensor, torch.Tensor], Tuple[List[int], List[int]]], edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, relabel_nodes: bool = False, size: Optional[Tuple[int, int]] = None, return_edge_mask: bool = False) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]`

Returns the induced subgraph of the bipartite graph
:obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

Args:
    subset (Tuple[Tensor, Tensor] or tuple([int],[int])): The nodes
        to keep.
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
        :obj:`edge_index` will be relabeled to hold consecutive indices
        starting from zero. (default: :obj:`False`)
    size (tuple, optional): The number of nodes.
        (default: :obj:`None`)
    return_edge_mask (bool, optional): If set to :obj:`True`, will return
        the edge mask to filter out additional edge features.
        (default: :obj:`False`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
    ...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
    >>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    >>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
    >>> bipartite_subgraph(subset, edge_index, edge_attr)
    (tensor([[2, 3, 5, 5],
            [3, 2, 2, 3]]),
    tensor([ 3,  4,  9, 10]))

    >>> bipartite_subgraph(subset, edge_index, edge_attr,
    ...                    return_edge_mask=True)
    (tensor([[2, 3, 5, 5],
            [3, 2, 2, 3]]),
    tensor([ 3,  4,  9, 10]),
    tensor([False, False,  True,  True, False, False, False, False,
            True,  True,  False]))

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

### `contains_isolated_nodes(edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> bool`

Returns :obj:`True` if the graph given by :attr:`edge_index` contains
isolated nodes.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: bool

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> contains_isolated_nodes(edge_index)
    False

    >>> contains_isolated_nodes(edge_index, num_nodes=3)
    True

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

### `erdos_renyi_graph(num_nodes: int, edge_prob: float, directed: bool = False) -> torch.Tensor`

Returns the :obj:`edge_index` of a random Erdos-Renyi graph.

Args:
    num_nodes (int): The number of nodes.
    edge_prob (float): Probability of an edge.
    directed (bool, optional): If set to :obj:`True`, will return a
        directed graph. (default: :obj:`False`)

Examples:
    >>> erdos_renyi_graph(5, 0.2, directed=False)
    tensor([[0, 1, 1, 4],
            [1, 0, 4, 1]])

    >>> erdos_renyi_graph(5, 0.2, directed=True)
    tensor([[0, 1, 3, 3, 4, 4],
            [4, 3, 1, 2, 1, 3]])

### `from_cugraph(g: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Converts a :obj:`cugraph` graph object into :obj:`edge_index` and
optional :obj:`edge_weight` tensors.

Args:
    g (cugraph.Graph): A :obj:`cugraph` graph object.

### `from_dgl(g: Any) -> Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')]`

Converts a :obj:`dgl` graph object to a
:class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData` instance.

Args:
    g (dgl.DGLGraph): The :obj:`dgl` graph object.

Example:
    >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
    >>> g.ndata['x'] = torch.randn(g.num_nodes(), 3)
    >>> g.edata['edge_attr'] = torch.randn(g.num_edges(), 2)
    >>> data = from_dgl(g)
    >>> data
    Data(x=[6, 3], edge_attr=[4, 2], edge_index=[2, 4])

    >>> g = dgl.heterograph({
    >>> g = dgl.heterograph({
    ...     ('author', 'writes', 'paper'): ([0, 1, 1, 2, 3, 3, 4],
    ...                                     [0, 0, 1, 1, 1, 2, 2])})
    >>> g.nodes['author'].data['x'] = torch.randn(5, 3)
    >>> g.nodes['paper'].data['x'] = torch.randn(5, 3)
    >>> data = from_dgl(g)
    >>> data
    HeteroData(
    author={ x=[5, 3] },
    paper={ x=[3, 3] },
    (author, writes, paper)={ edge_index=[2, 7] }
    )

### `from_nested_tensor(x: torch.Tensor, return_batch: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Given a `nested PyTorch tensor
<https://pytorch.org/docs/stable/nested.html>`__, creates a contiguous
batch of tensors
:math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`, and
optionally a batch vector which assigns each element to a specific example.
Reverse operation of :meth:`to_nested_tensor`.

Args:
    x (torch.Tensor): The nested input tensor. The size of nested tensors
        need to match except for the first dimension.
    return_batch (bool, optional): If set to :obj:`True`, will also return
        the batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`.
        (default: :obj:`False`)

### `from_networkit(g: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Converts a :class:`networkit.Graph` to a
:obj:`(edge_index, edge_weight)` tuple.
If the :class:`networkit.Graph` is not weighted, the returned
:obj:`edge_weight` will be :obj:`None`.

Args:
    g (networkkit.graph.Graph): A :obj:`networkit` graph object.

### `from_networkx(G: Any, group_node_attrs: Union[List[str], Literal['all'], NoneType] = None, group_edge_attrs: Union[List[str], Literal['all'], NoneType] = None) -> 'torch_geometric.data.Data'`

Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
:class:`torch_geometric.data.Data` instance.

Args:
    G (networkx.Graph or networkx.DiGraph): A networkx graph.
    group_node_attrs (List[str] or "all", optional): The node attributes to
        be concatenated and added to :obj:`data.x`. (default: :obj:`None`)
    group_edge_attrs (List[str] or "all", optional): The edge attributes to
        be concatenated and added to :obj:`data.edge_attr`.
        (default: :obj:`None`)

.. note::

    All :attr:`group_node_attrs` and :attr:`group_edge_attrs` values must
    be numeric.

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>> g = to_networkx(data)
    >>> # A `Data` object is returned
    >>> from_networkx(g)
    Data(edge_index=[2, 6], num_nodes=4)

### `from_rdmol(mol: Any) -> 'torch_geometric.data.Data'`

Converts a :class:`rdkit.Chem.Mol` instance to a
:class:`torch_geometric.data.Data` instance.

Args:
    mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.

### `from_scipy_sparse_matrix(A: Any) -> Tuple[torch.Tensor, torch.Tensor]`

Converts a scipy sparse matrix to edge indices and edge attributes.

Args:
    A (scipy.sparse): A sparse matrix.

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> adj = to_scipy_sparse_matrix(edge_index)
    >>> # `edge_index` and `edge_weight` are both returned
    >>> from_scipy_sparse_matrix(adj)
    (tensor([[0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]]),
    tensor([1., 1., 1., 1., 1., 1.]))

### `from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data'`

Converts a SMILES string to a :class:`torch_geometric.data.Data`
instance.

Args:
    smiles (str): The SMILES string.
    with_hydrogen (bool, optional): If set to :obj:`True`, will store
        hydrogens in the molecule graph. (default: :obj:`False`)
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

### `from_trimesh(mesh: Any) -> 'torch_geometric.data.Data'`

Converts a :obj:`trimesh.Trimesh` to a
:class:`torch_geometric.data.Data` instance.

Args:
    mesh (trimesh.Trimesh): A :obj:`trimesh` mesh.

Example:
    >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    ...                    dtype=torch.float)
    >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    >>> data = Data(pos=pos, face=face)
    >>> mesh = to_trimesh(data)
    >>> from_trimesh(mesh)
    Data(pos=[4, 3], face=[3, 2])

### `geodesic_distance(pos: torch.Tensor, face: torch.Tensor, src: Optional[torch.Tensor] = None, dst: Optional[torch.Tensor] = None, norm: bool = True, max_distance: Optional[float] = None, num_workers: int = 0, **kwargs: Optional[torch.Tensor]) -> torch.Tensor`

Computes (normalized) geodesic distances of a mesh given by :obj:`pos`
and :obj:`face`. If :obj:`src` and :obj:`dst` are given, this method only
computes the geodesic distances for the respective source and target
node-pairs.

.. note::

    This function requires the :obj:`gdist` package.
    To install, run :obj:`pip install cython && pip install gdist`.

Args:
    pos (torch.Tensor): The node positions.
    face (torch.Tensor): The face indices.
    src (torch.Tensor, optional): If given, only compute geodesic distances
        for the specified source indices. (default: :obj:`None`)
    dst (torch.Tensor, optional): If given, only compute geodesic distances
        for the specified target indices. (default: :obj:`None`)
    norm (bool, optional): Normalizes geodesic distances by
        :math:`\sqrt{\textrm{area}(\mathcal{M})}`. (default: :obj:`True`)
    max_distance (float, optional): If given, only yields results for
        geodesic distances less than :obj:`max_distance`. This will speed
        up runtime dramatically. (default: :obj:`None`)
    num_workers (int, optional): How many subprocesses to use for
        calculating geodesic distances.
        :obj:`0` means that computation takes place in the main process.
        :obj:`-1` means that the available amount of CPU cores is used.
        (default: :obj:`0`)

:rtype: :class:`Tensor`

Example:
    >>> pos = torch.tensor([[0.0, 0.0, 0.0],
    ...                     [2.0, 0.0, 0.0],
    ...                     [0.0, 2.0, 0.0],
    ...                     [2.0, 2.0, 0.0]])
    >>> face = torch.tensor([[0, 0],
    ...                      [1, 2],
    ...                      [3, 3]])
    >>> geodesic_distance(pos, face)
    [[0, 1, 1, 1.4142135623730951],
    [1, 0, 1.4142135623730951, 1],
    [1, 1.4142135623730951, 0, 1],
    [1.4142135623730951, 1, 1, 0]]

### `get_embeddings(model: torch.nn.modules.module.Module, *args: Any, **kwargs: Any) -> List[torch.Tensor]`

Returns the output embeddings of all
:class:`~torch_geometric.nn.conv.MessagePassing` layers in
:obj:`model`.

Internally, this method registers forward hooks on all
:class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
and runs the forward pass of the :obj:`model` by calling
:obj:`model(*args, **kwargs)`.

Args:
    model (torch.nn.Module): The message passing model.
    *args: Arguments passed to the model.
    **kwargs (optional): Additional keyword arguments passed to the model.

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

### `get_mesh_laplacian(pos: torch.Tensor, face: torch.Tensor, normalization: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Computes the mesh Laplacian of a mesh given by :obj:`pos` and
:obj:`face`.

Computation is based on the cotangent matrix defined as

.. math::
    \mathbf{C}_{ij} = \begin{cases}
        \frac{\cot \angle_{ikj}~+\cot \angle_{ilj}}{2} &
        \text{if } i, j \text{ is an edge} \\
        -\sum_{j \in N(i)}{C_{ij}} &
        \text{if } i \text{ is in the diagonal} \\
        0 & \text{otherwise}
  \end{cases}

Normalization depends on the mass matrix defined as

.. math::
    \mathbf{M}_{ij} = \begin{cases}
        a(i) & \text{if } i \text{ is in the diagonal} \\
        0 & \text{otherwise}
  \end{cases}

where :math:`a(i)` is obtained by joining the barycenters of the
triangles around vertex :math:`i`.

Args:
    pos (Tensor): The node positions.
    face (LongTensor): The face indices.
    normalization (str, optional): The normalization scheme for the mesh
        Laplacian (default: :obj:`None`):

        1. :obj:`None`: No normalization
        :math:`\mathbf{L} = \mathbf{C}`

        2. :obj:`"sym"`: Symmetric normalization
        :math:`\mathbf{L} = \mathbf{M}^{-1/2} \mathbf{C}\mathbf{M}^{-1/2}`

        3. :obj:`"rw"`: Row-wise normalization
        :math:`\mathbf{L} = \mathbf{M}^{-1} \mathbf{C}`

### `get_num_hops(model: torch.nn.modules.module.Module) -> int`

Returns the number of hops the model is aggregating information
from.

.. note::

    This function counts the number of message passing layers as an
    approximation of the total number of hops covered by the model.
    Its output may not necessarily be correct in case message passing
    layers perform multi-hop aggregation, *e.g.*, as in
    :class:`~torch_geometric.nn.conv.ChebConv`.

Example:
    >>> class GNN(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv1 = GCNConv(3, 16)
    ...         self.conv2 = GCNConv(16, 16)
    ...         self.lin = Linear(16, 2)
    ...
    ...     def forward(self, x, edge_index):
    ...         x = self.conv1(x, edge_index).relu()
    ...         x = self.conv2(x, edge_index).relu()
    ...         return self.lin(x)
    >>> get_num_hops(GNN())
    2

### `get_ppr(edge_index: torch.Tensor, alpha: float = 0.2, eps: float = 1e-05, target: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Calculates the personalized PageRank (PPR) vector for all or a subset
of nodes using a variant of the `Andersen algorithm
<https://mathweb.ucsd.edu/~fan/wp/localpartition.pdf>`_.

Args:
    edge_index (torch.Tensor): The indices of the graph.
    alpha (float, optional): The alpha value of the PageRank algorithm.
        (default: :obj:`0.2`)
    eps (float, optional): The threshold for stopping the PPR calculation
        (:obj:`edge_weight >= eps * out_degree`). (default: :obj:`1e-5`)
    target (torch.Tensor, optional): The target nodes to compute PPR for.
        If not given, calculates PPR vectors for all nodes.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes. (default: :obj:`None`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)

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

### `grid(height: int, width: int, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Returns the edge indices of a two-dimensional grid graph with height
:attr:`height` and width :attr:`width` and its node positions.

Args:
    height (int): The height of the grid.
    width (int): The width of the grid.
    dtype (torch.dtype, optional): The desired data type of the returned
        position tensor. (default: :obj:`None`)
    device (torch.device, optional): The desired device of the returned
        tensors. (default: :obj:`None`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Example:
    >>> (row, col), pos = grid(height=2, width=2)
    >>> row
    tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
    >>> col
    tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    >>> pos
    tensor([[0., 1.],
            [1., 1.],
            [0., 0.],
            [1., 0.]])

### `group_argsort(src: torch.Tensor, index: torch.Tensor, dim: int = 0, num_groups: Optional[int] = None, descending: bool = False, return_consecutive: bool = False, stable: bool = False) -> torch.Tensor`

Returns the indices that sort the tensor :obj:`src` along a given
dimension in ascending order by value.
In contrast to :meth:`torch.argsort`, sorting is performed in groups
according to the values in :obj:`index`.

Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The index tensor.
    dim (int, optional): The dimension along which to index.
        (default: :obj:`0`)
    num_groups (int, optional): The number of groups.
        (default: :obj:`None`)
    descending (bool, optional): Controls the sorting order (ascending or
        descending). (default: :obj:`False`)
    return_consecutive (bool, optional): If set to :obj:`True`, will not
        offset the output to start from :obj:`0` for each group.
        (default: :obj:`False`)
    stable (bool, optional): Controls the relative order of equivalent
        elements. (default: :obj:`False`)

Example:
    >>> src = torch.tensor([0, 1, 5, 4, 3, 2, 6, 7, 8])
    >>> index = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2])
    >>> group_argsort(src, index)
    tensor([0, 1, 3, 2, 1, 0, 0, 1, 2])

### `group_cat(tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], indices: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], dim: int = 0, return_index: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Concatenates the given sequence of tensors :obj:`tensors` in the given
dimension :obj:`dim`.
Different from :meth:`torch.cat`, values along the concatenating dimension
are grouped according to the indices defined in the :obj:`index` tensors.
All tensors must have the same shape (except in the concatenating
dimension).

Args:
    tensors ([Tensor]): Sequence of tensors.
    indices ([Tensor]): Sequence of index tensors.
    dim (int, optional): The dimension along which the tensors are
        concatenated. (default: :obj:`0`)
    return_index (bool, optional): If set to :obj:`True`, will return the
        new index tensor. (default: :obj:`False`)

Example:
    >>> x1 = torch.tensor([[0.2716, 0.4233],
    ...                    [0.3166, 0.0142],
    ...                    [0.2371, 0.3839],
    ...                    [0.4100, 0.0012]])
    >>> x2 = torch.tensor([[0.3752, 0.5782],
    ...                    [0.7757, 0.5999]])
    >>> index1 = torch.tensor([0, 0, 1, 2])
    >>> index2 = torch.tensor([0, 2])
    >>> scatter_concat([x1,x2], [index1, index2], dim=0)
    tensor([[0.2716, 0.4233],
            [0.3166, 0.0142],
            [0.3752, 0.5782],
            [0.2371, 0.3839],
            [0.4100, 0.0012],
            [0.7757, 0.5999]])

### `homophily(edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], y: torch.Tensor, batch: Optional[torch.Tensor] = None, method: str = 'edge') -> Union[float, torch.Tensor]`

The homophily of a graph characterizes how likely nodes with the same
label are near each other in a graph.

There are many measures of homophily that fits this definition.
In particular:

- In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
  and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
  homophily is the fraction of edges in a graph which connects nodes
  that have the same class label:

  .. math::
    \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
    {|\mathcal{E}|}

  That measure is called the *edge homophily ratio*.

- In the `"Geom-GCN: Geometric Graph Convolutional Networks"
  <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
  across neighborhoods:

  .. math::
    \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
    \in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }

  That measure is called the *node homophily ratio*.

- In the `"Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
  and Strong Simple Methods" <https://arxiv.org/abs/2110.14446>`_ paper,
  edge homophily is modified to be insensitive to the number of classes
  and size of each class:

  .. math::
    \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
    {|\mathcal{V}|} \right),

  where :math:`C` denotes the number of classes, :math:`|\mathcal{C}_k|`
  denotes the number of nodes of class :math:`k`, and :math:`h_k` denotes
  the edge homophily ratio of nodes of class :math:`k`.

  Thus, that measure is called the *class insensitive edge homophily
  ratio*.

Args:
    edge_index (Tensor or SparseTensor): The graph connectivity.
    y (Tensor): The labels.
    batch (LongTensor, optional): Batch vector\
        :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns
        each node to a specific example. (default: :obj:`None`)
    method (str, optional): The method used to calculate the homophily,
        either :obj:`"edge"` (first formula), :obj:`"node"` (second
        formula) or :obj:`"edge_insensitive"` (third formula).
        (default: :obj:`"edge"`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 2, 3],
    ...                            [1, 2, 0, 4]])
    >>> y = torch.tensor([0, 0, 0, 0, 1])
    >>> # Edge homophily ratio
    >>> homophily(edge_index, y, method='edge')
    0.75

    >>> # Node homophily ratio
    >>> homophily(edge_index, y, method='node')
    0.6000000238418579

    >>> # Class insensitive edge homophily ratio
    >>> homophily(edge_index, y, method='edge_insensitive')
    0.19999998807907104

### `index_sort(inputs: torch.Tensor, max_value: Optional[int] = None, stable: bool = False) -> Tuple[torch.Tensor, torch.Tensor]`

Sorts the elements of the :obj:`inputs` tensor in ascending order.
It is expected that :obj:`inputs` is one-dimensional and that it only
contains positive integer values. If :obj:`max_value` is given, it can
be used by the underlying algorithm for better performance.

Args:
    inputs (torch.Tensor): A vector with positive integer values.
    max_value (int, optional): The maximum value stored inside
        :obj:`inputs`. This value can be an estimation, but needs to be
        greater than or equal to the real maximum.
        (default: :obj:`None`)
    stable (bool, optional): Makes the sorting routine stable, which
        guarantees that the order of equivalent elements is preserved.
        (default: :obj:`False`)

### `index_to_mask(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

Converts indices to a mask representation.

Args:
    index (Tensor): The indices.
    size (int, optional): The size of the mask. If set to :obj:`None`, a
        minimal sized output mask is returned.

Example:
    >>> index = torch.tensor([1, 3, 5])
    >>> index_to_mask(index)
    tensor([False,  True, False,  True, False,  True])

    >>> index_to_mask(index, size=7)
    tensor([False,  True, False,  True, False,  True, False])

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

### `is_undirected(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor]] = None, num_nodes: Optional[int] = None) -> bool`

Returns :obj:`True` if the graph given by :attr:`edge_index` is
undirected.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
        dimensional edge features.
        If given as a list, will check for equivalence in all its entries.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index) + 1`. (default: :obj:`None`)

:rtype: bool

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                         [1, 0, 0]])
    >>> weight = torch.tensor([0, 0, 1])
    >>> is_undirected(edge_index, weight)
    True

    >>> weight = torch.tensor([0, 1, 1])
    >>> is_undirected(edge_index, weight)
    False

### `k_hop_subgraph(node_idx: Union[int, List[int], torch.Tensor], num_hops: int, edge_index: torch.Tensor, relabel_nodes: bool = False, num_nodes: Optional[int] = None, flow: str = 'source_to_target', directed: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

Computes the induced subgraph of :obj:`edge_index` around all nodes in
:attr:`node_idx` reachable within :math:`k` hops.

The :attr:`flow` argument denotes the direction of edges for finding
:math:`k`-hop neighbors. If set to :obj:`"source_to_target"`, then the
method will find all neighbors that point to the initial set of seed nodes
in :attr:`node_idx.`
This mimics the natural flow of message passing in Graph Neural Networks.

The method returns (1) the nodes involved in the subgraph, (2) the filtered
:obj:`edge_index` connectivity, (3) the mapping from node indices in
:obj:`node_idx` to their new location, and (4) the edge mask indicating
which edges were preserved.

Args:
    node_idx (int, list, tuple or :obj:`torch.Tensor`): The central seed
        node(s).
    num_hops (int): The number of hops :math:`k`.
    edge_index (LongTensor): The edge indices.
    relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
        :obj:`edge_index` will be relabeled to hold consecutive indices
        starting from zero. (default: :obj:`False`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    flow (str, optional): The flow direction of :math:`k`-hop aggregation
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)
    directed (bool, optional): If set to :obj:`True`, will only include
        directed edges to the seed nodes :obj:`node_idx`.
        (default: :obj:`False`)

:rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
         :class:`BoolTensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
    ...                            [2, 2, 4, 4, 6, 6]])

    >>> # Center node 6, 2-hops
    >>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
    ...     6, 2, edge_index, relabel_nodes=True)
    >>> subset
    tensor([2, 3, 4, 5, 6])
    >>> edge_index
    tensor([[0, 1, 2, 3],
            [2, 2, 4, 4]])
    >>> mapping
    tensor([4])
    >>> edge_mask
    tensor([False, False,  True,  True,  True,  True])
    >>> subset[mapping]
    tensor([6])

    >>> edge_index = torch.tensor([[1, 2, 4, 5],
    ...                            [0, 1, 5, 6]])
    >>> (subset, edge_index,
    ...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
    ...                                       edge_index,
    ...                                       relabel_nodes=True)
    >>> subset
    tensor([0, 1, 2, 4, 5, 6])
    >>> edge_index
    tensor([[1, 2, 3, 4],
            [0, 1, 4, 5]])
    >>> mapping
    tensor([0, 5])
    >>> edge_mask
    tensor([True, True, True, True])
    >>> subset[mapping]
    tensor([0, 6])

### `lexsort(keys: List[torch.Tensor], dim: int = -1, descending: bool = False) -> torch.Tensor`

Performs an indirect stable sort using a sequence of keys.

Given multiple sorting keys, returns an array of integer indices that
describe their sort order.
The last key in the sequence is used for the primary sort order, the
second-to-last key for the secondary sort order, and so on.

Args:
    keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
        The last key is the primary sort key.
    dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
    descending (bool, optional): Controls the sorting order (ascending or
        descending). (default: :obj:`False`)

### `mask_feature(x: torch.Tensor, p: float = 0.5, mode: str = 'col', fill_value: float = 0.0, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Randomly masks feature from the feature matrix
:obj:`x` with probability :obj:`p` using samples from
a Bernoulli distribution.

The method returns (1) the retained :obj:`x`, (2) the feature
mask broadcastable with :obj:`x` (:obj:`mode='row'` and :obj:`mode='col'`)
or with the same shape as :obj:`x` (:obj:`mode='all'`),
indicating where features are retained.

Args:
    x (FloatTensor): The feature matrix.
    p (float, optional): The masking ratio. (default: :obj:`0.5`)
    mode (str, optional): The masked scheme to use for feature masking.
        (:obj:`"row"`, :obj:`"col"` or :obj:`"all"`).
        If :obj:`mode='col'`, will mask entire features of all nodes
        from the feature matrix. If :obj:`mode='row'`, will mask entire
        nodes from the feature matrix. If :obj:`mode='all'`, will mask
        individual features across all nodes. (default: :obj:`'col'`)
    fill_value (float, optional): The value for masked features in the
        output tensor. (default: :obj:`0`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

:rtype: (:class:`FloatTensor`, :class:`BoolTensor`)

Examples:
    >>> # Masked features are column-wise sampled
    >>> x = torch.tensor([[1, 2, 3],
    ...                   [4, 5, 6],
    ...                   [7, 8, 9]], dtype=torch.float)
    >>> x, feat_mask = mask_feature(x)
    >>> x
    tensor([[1., 0., 3.],
            [4., 0., 6.],
            [7., 0., 9.]]),
    >>> feat_mask
    tensor([[True, False, True]])

    >>> # Masked features are row-wise sampled
    >>> x, feat_mask = mask_feature(x, mode='row')
    >>> x
    tensor([[1., 2., 3.],
            [0., 0., 0.],
            [7., 8., 9.]]),
    >>> feat_mask
    tensor([[True], [False], [True]])

    >>> # Masked features are uniformly sampled
    >>> x, feat_mask = mask_feature(x, mode='all')
    >>> x
    tensor([[0., 0., 0.],
            [4., 0., 6.],
            [0., 0., 9.]])
    >>> feat_mask
    tensor([[False, False, False],
            [True, False,  True],
            [False, False,  True]])

### `mask_select(src: torch.Tensor, dim: int, mask: torch.Tensor) -> torch.Tensor`

Returns a new tensor which masks the :obj:`src` tensor along the
dimension :obj:`dim` according to the boolean mask :obj:`mask`.

Args:
    src (torch.Tensor): The input tensor.
    dim (int): The dimension in which to mask.
    mask (torch.BoolTensor): The 1-D tensor containing the binary mask to
        index with.

### `mask_to_index(mask: torch.Tensor) -> torch.Tensor`

Converts a mask to an index representation.

Args:
    mask (Tensor): The mask.

Example:
    >>> mask = torch.tensor([False, True, False])
    >>> mask_to_index(mask)
    tensor([1])

### `narrow(src: Union[torch.Tensor, List[Any]], dim: int, start: int, length: int) -> Union[torch.Tensor, List[Any]]`

Narrows the input tensor or input list to the specified range.

Args:
    src (torch.Tensor or list): The input tensor or list.
    dim (int): The dimension along which to narrow.
    start (int): The starting dimension.
    length (int): The distance to the ending dimension.

### `negative_sampling(edge_index: torch.Tensor, num_nodes: Union[int, Tuple[int, int], NoneType] = None, num_neg_samples: Optional[int] = None, method: str = 'sparse', force_undirected: bool = False) -> torch.Tensor`

Samples random negative edges of a graph given by :attr:`edge_index`.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int or Tuple[int, int], optional): The number of nodes,
        *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
        (default: :obj:`None`)
    num_neg_samples (int, optional): The (approximate) number of negative
        samples to return.
        If set to :obj:`None`, will try to return a negative edge for every
        positive edge. (default: :obj:`None`)
    method (str, optional): The method to use for negative sampling,
        *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
        This is a memory/runtime trade-off.
        :obj:`"sparse"` will work on any graph of any size, while
        :obj:`"dense"` can perform faster true-negative checks.
        (default: :obj:`"sparse"`)
    force_undirected (bool, optional): If set to :obj:`True`, sampled
        negative edges will be undirected. (default: :obj:`False`)

:rtype: LongTensor

Examples:
    >>> # Standard usage
    >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
    ...                               [0, 1, 2, 3]])
    >>> negative_sampling(edge_index)
    tensor([[3, 0, 0, 3],
            [2, 3, 2, 1]])

    >>> # For bipartite graph
    >>> negative_sampling(edge_index, num_nodes=(3, 4))
    tensor([[0, 2, 2, 1],
            [2, 2, 1, 3]])

### `normalize_edge_index(edge_index: torch.Tensor, num_nodes: Optional[int] = None, add_self_loops: bool = True, symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Applies normalization to the edges of a graph.

This function can add self-loops to the graph and apply either symmetric or
asymmetric normalization based on the node degrees.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int, int], optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    symmetric (bool, optional):  If set to :obj:`True`, symmetric
        normalization (:math:`D^{-1/2} A D^{-1/2}`) is used, otherwise
        asymmetric normalization (:math:`D^{-1} A`).

### `normalized_cut(edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor`

Computes the normalized cut :math:`\mathbf{e}_{i,j} \cdot
\left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)` of a weighted graph
given by edge indices and edge attributes.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor): Edge weights or multi-dimensional edge features.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: :class:`Tensor`

Example:
    >>> edge_index = torch.tensor([[1, 1, 2, 3],
    ...                            [3, 3, 1, 2]])
    >>> edge_attr = torch.tensor([1., 1., 1., 1.])
    >>> normalized_cut(edge_index, edge_attr)
    tensor([1.5000, 1.5000, 2.0000, 1.5000])

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

### `remove_isolated_nodes(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]`

Removes the isolated nodes from the graph given by :attr:`edge_index`
with optional edge attributes :attr:`edge_attr`.
In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
out isolated node features later on.
Self-loops are preserved for non-isolated nodes.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor, optional): Edge weights or multi-dimensional
        edge features. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: (LongTensor, Tensor, BoolTensor)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 0],
    ...                            [1, 0, 0]])
    >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index)
    >>> mask # node mask (2 nodes)
    tensor([True, True])

    >>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index,
    ...                                                     num_nodes=3)
    >>> mask # node mask (3 nodes)
    tensor([True, True, False])

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

### `segment(src: torch.Tensor, ptr: torch.Tensor, reduce: str = 'sum') -> torch.Tensor`

Reduces all values in the first dimension of the :obj:`src` tensor
within the ranges specified in the :obj:`ptr`. See the `documentation
<https://pytorch-scatter.readthedocs.io/en/latest/functions/
segment_csr.html>`__ of the :obj:`torch_scatter` package for more
information.

Args:
    src (torch.Tensor): The source tensor.
    ptr (torch.Tensor): A monotonically increasing pointer tensor that
        refers to the boundaries of segments such that :obj:`ptr[0] = 0`
        and :obj:`ptr[-1] = src.size(0)`.
    reduce (str, optional): The reduce operation (:obj:`"sum"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`).
        (default: :obj:`"sum"`)

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

### `select(src: Union[torch.Tensor, List[Any], torch_geometric.typing.TensorFrame], index_or_mask: torch.Tensor, dim: int) -> Union[torch.Tensor, List[Any]]`

Selects the input tensor or input list according to a given index or
mask vector.

Args:
    src (torch.Tensor or list): The input tensor or list.
    index_or_mask (torch.Tensor): The index or mask vector.
    dim (int): The dimension along which to select.

### `shuffle_node(x: torch.Tensor, batch: Optional[torch.Tensor] = None, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]`

Randomly shuffle the feature matrix :obj:`x` along the
first dimension.

The method returns (1) the shuffled :obj:`x`, (2) the permutation
indicating the orders of original nodes after shuffling.

Args:
    x (FloatTensor): The feature matrix.
    batch (LongTensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. Must be ordered. (default: :obj:`None`)
    training (bool, optional): If set to :obj:`False`, this operation is a
        no-op. (default: :obj:`True`)

:rtype: (:class:`FloatTensor`, :class:`LongTensor`)

Example:
    >>> # Standard case
    >>> x = torch.tensor([[0, 1, 2],
    ...                   [3, 4, 5],
    ...                   [6, 7, 8],
    ...                   [9, 10, 11]], dtype=torch.float)
    >>> x, node_perm = shuffle_node(x)
    >>> x
    tensor([[ 3.,  4.,  5.],
            [ 9., 10., 11.],
            [ 0.,  1.,  2.],
            [ 6.,  7.,  8.]])
    >>> node_perm
    tensor([1, 3, 0, 2])

    >>> # For batched graphs as inputs
    >>> batch = torch.tensor([0, 0, 1, 1])
    >>> x, node_perm = shuffle_node(x, batch)
    >>> x
    tensor([[ 3.,  4.,  5.],
            [ 0.,  1.,  2.],
            [ 9., 10., 11.],
            [ 6.,  7.,  8.]])
    >>> node_perm
    tensor([1, 0, 3, 2])

### `softmax(src: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None, dim: int = 0) -> torch.Tensor`

Computes a sparsely evaluated softmax.
Given a value tensor :attr:`src`, this function first groups the values
along the first dimension based on the indices specified in :attr:`index`,
and then proceeds to compute the softmax individually for each group.

Args:
    src (Tensor): The source tensor.
    index (LongTensor, optional): The indices of elements for applying the
        softmax. (default: :obj:`None`)
    ptr (LongTensor, optional): If given, computes the softmax based on
        sorted inputs in CSR representation. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    dim (int, optional): The dimension in which to normalize.
        (default: :obj:`0`)

:rtype: :class:`Tensor`

Examples:
    >>> src = torch.tensor([1., 1., 1., 1.])
    >>> index = torch.tensor([0, 0, 1, 2])
    >>> ptr = torch.tensor([0, 2, 3, 4])
    >>> softmax(src, index)
    tensor([0.5000, 0.5000, 1.0000, 1.0000])

    >>> softmax(src, None, ptr)
    tensor([0.5000, 0.5000, 1.0000, 1.0000])

    >>> src = torch.randn(4, 4)
    >>> ptr = torch.tensor([0, 4])
    >>> softmax(src, index, dim=-1)
    tensor([[0.7404, 0.2596, 1.0000, 1.0000],
            [0.1702, 0.8298, 1.0000, 1.0000],
            [0.7607, 0.2393, 1.0000, 1.0000],
            [0.8062, 0.1938, 1.0000, 1.0000]])

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

### `spmm(src: Union[torch.Tensor, torch_geometric.typing.SparseTensor], other: torch.Tensor, reduce: str = 'sum') -> torch.Tensor`

Matrix product of sparse matrix with dense matrix.

Args:
    src (torch.Tensor or torch_sparse.SparseTensor or EdgeIndex):
        The input sparse matrix which can be a
        :pyg:`PyG` :class:`torch_sparse.SparseTensor`,
        a :pytorch:`PyTorch` :class:`torch.sparse.Tensor` or
        a :pyg:`PyG` :class:`EdgeIndex`.
    other (torch.Tensor): The input dense matrix.
    reduce (str, optional): The reduce operation to use
        (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
        (default: :obj:`"sum"`)

:rtype: :class:`Tensor`

### `stochastic_blockmodel_graph(block_sizes: Union[List[int], torch.Tensor], edge_probs: Union[List[List[float]], torch.Tensor], directed: bool = False) -> torch.Tensor`

Returns the :obj:`edge_index` of a stochastic blockmodel graph.

Args:
    block_sizes ([int] or LongTensor): The sizes of blocks.
    edge_probs ([[float]] or FloatTensor): The density of edges going
        from each block to each other block. Must be symmetric if the
        graph is undirected.
    directed (bool, optional): If set to :obj:`True`, will return a
        directed graph. (default: :obj:`False`)

Examples:
    >>> block_sizes = [2, 2, 4]
    >>> edge_probs = [[0.25, 0.05, 0.02],
    ...               [0.05, 0.35, 0.07],
    ...               [0.02, 0.07, 0.40]]
    >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
    ...                             directed=False)
    tensor([[2, 4, 4, 5, 5, 6, 7, 7],
            [5, 6, 7, 2, 7, 4, 4, 5]])

    >>> stochastic_blockmodel_graph(block_sizes, edge_probs,
    ...                             directed=True)
    tensor([[0, 2, 3, 4, 4, 5, 5],
            [3, 4, 1, 5, 6, 6, 7]])

### `structured_negative_sampling(edge_index: torch.Tensor, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`

Samples a negative edge :obj:`(i,k)` for every positive edge
:obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
tuple of the form :obj:`(i,j,k)`.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    contains_neg_self_loops (bool, optional): If set to
        :obj:`False`, sampled negative edges will not contain self loops.
        (default: :obj:`True`)

:rtype: (LongTensor, LongTensor, LongTensor)

Example:
    >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
    ...                               [0, 1, 2, 3]])
    >>> structured_negative_sampling(edge_index)
    (tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))

### `structured_negative_sampling_feasible(edge_index: torch.Tensor, num_nodes: Optional[int] = None, contains_neg_self_loops: bool = True) -> bool`

Returns :obj:`True` if
:meth:`~torch_geometric.utils.structured_negative_sampling` is feasible
on the graph given by :obj:`edge_index`.
:meth:`~torch_geometric.utils.structured_negative_sampling` is infeasible
if at least one node is connected to all other nodes.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    contains_neg_self_loops (bool, optional): If set to
        :obj:`False`, sampled negative edges will not contain self loops.
        (default: :obj:`True`)

:rtype: bool

Examples:
    >>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
    ...                                [1, 2, 0, 2, 0, 1, 1]])
    >>> structured_negative_sampling_feasible(edge_index, 3, False)
    False

    >>> structured_negative_sampling_feasible(edge_index, 3, True)
    True

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

### `to_cugraph(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, relabel_nodes: bool = True, directed: bool = True) -> Any`

Converts a graph given by :obj:`edge_index` and optional
:obj:`edge_weight` into a :obj:`cugraph` graph object.

Args:
    edge_index (torch.Tensor): The edge indices of the graph.
    edge_weight (torch.Tensor, optional): The edge weights of the graph.
        (default: :obj:`None`)
    relabel_nodes (bool, optional): If set to :obj:`True`,
        :obj:`cugraph` will remove any isolated nodes, leading to a
        relabeling of nodes. (default: :obj:`True`)
    directed (bool, optional): If set to :obj:`False`, the graph will be
        undirected. (default: :obj:`True`)

### `to_dense_adj(edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None, edge_attr: Optional[torch.Tensor] = None, max_num_nodes: Optional[int] = None, batch_size: Optional[int] = None) -> torch.Tensor`

Converts batched sparse adjacency matrices given by edge indices and
edge attributes to a single dense batched adjacency matrix.

Args:
    edge_index (LongTensor): The edge indices.
    batch (LongTensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
        features.
        If :obj:`edge_index` contains duplicated edges, the dense adjacency
        matrix output holds the summed up entries of :obj:`edge_attr` for
        duplicated edges. (default: :obj:`None`)
    max_num_nodes (int, optional): The size of the output node dimension.
        (default: :obj:`None`)
    batch_size (int, optional): The batch size. (default: :obj:`None`)

:rtype: :class:`Tensor`

Examples:
    >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
    ...                            [0, 1, 0, 3, 0]])
    >>> batch = torch.tensor([0, 0, 1, 1])
    >>> to_dense_adj(edge_index, batch)
    tensor([[[1., 1.],
            [1., 0.]],
            [[0., 1.],
            [1., 0.]]])

    >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
    tensor([[[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],
            [[0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]])

    >>> edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> to_dense_adj(edge_index, batch, edge_attr)
    tensor([[[1., 2.],
            [3., 0.]],
            [[0., 4.],
            [5., 0.]]])

### `to_dense_batch(x: torch.Tensor, batch: Optional[torch.Tensor] = None, fill_value: float = 0.0, max_num_nodes: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Given a sparse batch of node features
:math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
:math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
dense node feature tensor
:math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
:math:`N_{\max} = \max_i^B N_i`).
In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
N_{\max}}` is returned, holding information about the existence of
fake-nodes in the dense representation.

Args:
    x (Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (LongTensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. Must be ordered. (default: :obj:`None`)
    fill_value (float, optional): The value for invalid entries in the
        resulting dense output tensor. (default: :obj:`0`)
    max_num_nodes (int, optional): The size of the output node dimension.
        (default: :obj:`None`)
    batch_size (int, optional): The batch size. (default: :obj:`None`)

:rtype: (:class:`Tensor`, :class:`BoolTensor`)

Examples:
    >>> x = torch.arange(12).view(6, 2)
    >>> x
    tensor([[ 0,  1],
            [ 2,  3],
            [ 4,  5],
            [ 6,  7],
            [ 8,  9],
            [10, 11]])

    >>> out, mask = to_dense_batch(x)
    >>> mask
    tensor([[True, True, True, True, True, True]])

    >>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
    >>> out, mask = to_dense_batch(x, batch)
    >>> out
    tensor([[[ 0,  1],
            [ 2,  3],
            [ 0,  0]],
            [[ 4,  5],
            [ 0,  0],
            [ 0,  0]],
            [[ 6,  7],
            [ 8,  9],
            [10, 11]]])
    >>> mask
    tensor([[ True,  True, False],
            [ True, False, False],
            [ True,  True,  True]])

    >>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
    >>> out
    tensor([[[ 0,  1],
            [ 2,  3],
            [ 0,  0],
            [ 0,  0]],
            [[ 4,  5],
            [ 0,  0],
            [ 0,  0],
            [ 0,  0]],
            [[ 6,  7],
            [ 8,  9],
            [10, 11],
            [ 0,  0]]])

    >>> mask
    tensor([[ True,  True, False, False],
            [ True, False, False, False],
            [ True,  True,  True, False]])

### `to_dgl(data: Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')]) -> Any`

Converts a :class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData` instance to a :obj:`dgl` graph
object.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
        The data object.

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
    >>> x = torch.randn(5, 3)
    >>> edge_attr = torch.randn(6, 2)
    >>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
    >>> g = to_dgl(data)
    >>> g
    Graph(num_nodes=5, num_edges=6,
        ndata_schemes={'x': Scheme(shape=(3,))}
        edata_schemes={'edge_attr': Scheme(shape=(2, ))})

    >>> data = HeteroData()
    >>> data['paper'].x = torch.randn(5, 3)
    >>> data['author'].x = torch.ones(5, 3)
    >>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    >>> data['author', 'cites', 'paper'].edge_index = edge_index
    >>> g = to_dgl(data)
    >>> g
    Graph(num_nodes={'author': 5, 'paper': 5},
        num_edges={('author', 'cites', 'paper'): 5},
        metagraph=[('author', 'paper', 'cites')])

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

### `to_nested_tensor(x: torch.Tensor, batch: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`

Given a contiguous batch of tensors
:math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`
(with :math:`N_i` indicating the number of elements in example :math:`i`),
creates a `nested PyTorch tensor
<https://pytorch.org/docs/stable/nested.html>`__.
Reverse operation of :meth:`from_nested_tensor`.

Args:
    x (torch.Tensor): The input tensor
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        element to a specific example. Must be ordered.
        (default: :obj:`None`)
    ptr (torch.Tensor, optional): Alternative representation of
        :obj:`batch` in compressed format. (default: :obj:`None`)
    batch_size (int, optional): The batch size :math:`B`.
        (default: :obj:`None`)

### `to_networkit(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None, directed: bool = True) -> Any`

Converts a :obj:`(edge_index, edge_weight)` tuple to a
:class:`networkit.Graph`.

Args:
    edge_index (torch.Tensor): The edge indices of the graph.
    edge_weight (torch.Tensor, optional): The edge weights of the graph.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes in the graph.
        (default: :obj:`None`)
    directed (bool, optional): If set to :obj:`False`, the graph will be
        undirected. (default: :obj:`True`)

### `to_networkx(data: Union[ForwardRef('torch_geometric.data.Data'), ForwardRef('torch_geometric.data.HeteroData')], node_attrs: Optional[Iterable[str]] = None, edge_attrs: Optional[Iterable[str]] = None, graph_attrs: Optional[Iterable[str]] = None, to_undirected: Union[bool, str, NoneType] = False, to_multi: bool = False, remove_self_loops: bool = False) -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
a directed :obj:`networkx.DiGraph` otherwise.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData): A
        homogeneous or heterogeneous data object.
    node_attrs (iterable of str, optional): The node attributes to be
        copied. (default: :obj:`None`)
    edge_attrs (iterable of str, optional): The edge attributes to be
        copied. (default: :obj:`None`)
    graph_attrs (iterable of str, optional): The graph attributes to be
        copied. (default: :obj:`None`)
    to_undirected (bool or str, optional): If set to :obj:`True`, will
        return a :class:`networkx.Graph` instead of a
        :class:`networkx.DiGraph`.
        By default, will include all edges and make them undirected.
        If set to :obj:`"upper"`, the undirected graph will only correspond
        to the upper triangle of the input adjacency matrix.
        If set to :obj:`"lower"`, the undirected graph will only correspond
        to the lower triangle of the input adjacency matrix.
        Only applicable in case the :obj:`data` object holds a homogeneous
        graph. (default: :obj:`False`)
    to_multi (bool, optional): if set to :obj:`True`, will return a
        :class:`networkx.MultiGraph` or a :class:`networkx:MultiDiGraph`
        (depending on the :obj:`to_undirected` option), which will not drop
        duplicated edges that may exist in :obj:`data`.
        (default: :obj:`False`)
    remove_self_loops (bool, optional): If set to :obj:`True`, will not
        include self-loops in the resulting graph. (default: :obj:`False`)

Examples:
    >>> edge_index = torch.tensor([
    ...     [0, 1, 1, 2, 2, 3],
    ...     [1, 0, 2, 1, 3, 2],
    ... ])
    >>> data = Data(edge_index=edge_index, num_nodes=4)
    >>> to_networkx(data)
    <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

### `to_rdmol(data: 'torch_geometric.data.Data', kekulize: bool = False) -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:class:`rdkit.Chem.Mol` instance.

Args:
    data (torch_geometric.data.Data): The molecular graph data.
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

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

### `to_smiles(data: 'torch_geometric.data.Data', kekulize: bool = False) -> str`

Converts a :class:`torch_geometric.data.Data` instance to a SMILES
string.

Args:
    data (torch_geometric.data.Data): The molecular graph.
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

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

### `to_trimesh(data: 'torch_geometric.data.Data') -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:obj:`trimesh.Trimesh`.

Args:
    data (torch_geometric.data.Data): The data object.

Example:
    >>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    ...                    dtype=torch.float)
    >>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()

    >>> data = Data(pos=pos, face=face)
    >>> to_trimesh(data)
    <trimesh.Trimesh(vertices.shape=(4, 3), faces.shape=(2, 3))>

### `to_undirected(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'add') -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Converts the graph given by :attr:`edge_index` to an undirected graph
such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
\mathcal{E}`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
        dimensional edge features.
        If given as a list, will remove duplicates for all its entries.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index) + 1`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`). (default: :obj:`"add"`)

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1],
    ...                            [1, 0, 2]])
    >>> to_undirected(edge_index)
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]])

    >>> edge_index = torch.tensor([[0, 1, 1],
    ...                            [1, 0, 2]])
    >>> edge_weight = torch.tensor([1., 1., 1.])
    >>> to_undirected(edge_index, edge_weight)
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([2., 2., 1., 1.]))

    >>> # Use 'mean' operation to merge edge features
    >>>  to_undirected(edge_index, edge_weight, reduce='mean')
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([1., 1., 1., 1.]))

### `train_test_split_edges(data: 'torch_geometric.data.Data', val_ratio: float = 0.05, test_ratio: float = 0.1) -> 'torch_geometric.data.Data'`

Splits the edges of a :class:`torch_geometric.data.Data` object
into positive and negative train/val/test edges.
As such, it will replace the :obj:`edge_index` attribute with
:obj:`train_pos_edge_index`, :obj:`train_pos_neg_adj_mask`,
:obj:`val_pos_edge_index`, :obj:`val_neg_edge_index` and
:obj:`test_pos_edge_index` attributes.
If :obj:`data` has edge features named :obj:`edge_attr`, then
:obj:`train_pos_edge_attr`, :obj:`val_pos_edge_attr` and
:obj:`test_pos_edge_attr` will be added as well.

.. warning::

    :meth:`~torch_geometric.utils.train_test_split_edges` is deprecated and
    will be removed in a future release.
    Use :class:`torch_geometric.transforms.RandomLinkSplit` instead.

Args:
    data (Data): The data object.
    val_ratio (float, optional): The ratio of positive validation edges.
        (default: :obj:`0.05`)
    test_ratio (float, optional): The ratio of positive test edges.
        (default: :obj:`0.1`)

:rtype: :class:`torch_geometric.data.Data`

### `tree_decomposition(mol: Any, return_vocab: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]`

The tree decomposition algorithm of molecules from the
`"Junction Tree Variational Autoencoder for Molecular Graph Generation"
<https://arxiv.org/abs/1802.04364>`_ paper.
Returns the graph connectivity of the junction tree, the assignment
mapping of each atom to the clique in the junction tree, and the number
of cliques.

Args:
    mol (rdkit.Chem.Mol): An :obj:`rdkit` molecule.
    return_vocab (bool, optional): If set to :obj:`True`, will return an
        identifier for each clique (ring, bond, bridged compounds, single).
        (default: :obj:`False`)

:rtype: :obj:`(LongTensor, LongTensor, int)` if :obj:`return_vocab` is
    :obj:`False`, else :obj:`(LongTensor, LongTensor, int, LongTensor)`

### `trim_to_layer(layer: int, num_sampled_nodes_per_hop: Union[List[int], Dict[str, List[int]]], num_sampled_edges_per_hop: Union[List[int], Dict[Tuple[str, str, str], List[int]]], x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], edge_attr: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor], NoneType] = None) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], Union[torch.Tensor, Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]]], Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor], NoneType]]`

Trims the :obj:`edge_index` representation, node features :obj:`x` and
edge features :obj:`edge_attr` to a minimal-sized representation for the
current GNN layer :obj:`layer` in directed
:class:`~torch_geometric.loader.NeighborLoader` scenarios.

This ensures that no computation is performed for nodes and edges that are
not included in the current GNN layer, thus avoiding unnecessary
computation within the GNN when performing neighborhood sampling.

Args:
    layer (int): The current GNN layer.
    num_sampled_nodes_per_hop (List[int] or Dict[NodeType, List[int]]): The
        number of sampled nodes per hop.
    num_sampled_edges_per_hop (List[int] or Dict[EdgeType, List[int]]): The
        number of sampled edges per hop.
    x (torch.Tensor or Dict[NodeType, torch.Tensor]): The homogeneous or
        heterogeneous (hidden) node features.
    edge_index (torch.Tensor or Dict[EdgeType, torch.Tensor]): The
        homogeneous or heterogeneous edge indices.
    edge_attr (torch.Tensor or Dict[EdgeType, torch.Tensor], optional): The
        homogeneous or heterogeneous (hidden) edge features.

### `unbatch(src: torch.Tensor, batch: torch.Tensor, dim: int = 0, batch_size: Optional[int] = None) -> List[torch.Tensor]`

Splits :obj:`src` according to a :obj:`batch` vector along dimension
:obj:`dim`.

Args:
    src (Tensor): The source tensor.
    batch (LongTensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        entry in :obj:`src` to a specific example. Must be ordered.
    dim (int, optional): The dimension along which to split the :obj:`src`
        tensor. (default: :obj:`0`)
    batch_size (int, optional): The batch size. (default: :obj:`None`)

:rtype: :class:`List[Tensor]`

Example:
    >>> src = torch.arange(7)
    >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    >>> unbatch(src, batch)
    (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))

### `unbatch_edge_index(edge_index: torch.Tensor, batch: torch.Tensor, batch_size: Optional[int] = None) -> List[torch.Tensor]`

Splits the :obj:`edge_index` according to a :obj:`batch` vector.

Args:
    edge_index (Tensor): The edge_index tensor. Must be ordered.
    batch (LongTensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. Must be ordered.
    batch_size (int, optional): The batch size. (default: :obj:`None`)

:rtype: :class:`List[Tensor]`

Example:
    >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
    ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
    >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    >>> unbatch_edge_index(edge_index, batch)
    (tensor([[0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]]),
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]))
