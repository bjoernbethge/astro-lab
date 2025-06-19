# random

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.random`

## Functions (5)

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
