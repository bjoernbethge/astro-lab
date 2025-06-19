# augmentation

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.augmentation`

## Functions (6)

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

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
