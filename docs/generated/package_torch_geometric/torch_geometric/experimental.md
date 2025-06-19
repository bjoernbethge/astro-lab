# experimental

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.experimental`

## Functions (43)

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

### `disable_dynamic_shapes(required_args: List[str]) -> Callable`

A decorator that disables the usage of dynamic shapes for the given
arguments, i.e., it will raise an error in case :obj:`required_args` are
not passed and needs to be automatically inferred.

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

### `get_options(options: Union[str, List[str], NoneType]) -> List[str]`

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

### `is_experimental_mode_enabled(options: Union[str, List[str], NoneType] = None) -> bool`

Returns :obj:`True` if the experimental mode is enabled. See
:class:`torch_geometric.experimental_mode` for a list of (optional)
options.

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

### `set_experimental_mode_enabled(mode: bool, options: Union[str, List[str], NoneType] = None) -> None`

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

## Classes (3)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `experimental_mode`

Context-manager that enables the experimental mode to test new but
potentially unstable features.

.. code-block:: python

    with torch_geometric.experimental_mode():
        out = model(data.x, data.edge_index)

Args:
    options (str or list, optional): Currently there are no experimental
        features.

### `set_experimental_mode`

Context-manager that sets the experimental mode on or off.

:class:`set_experimental_mode` will enable or disable the experimental mode
based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`experimental_mode` above for more details.
