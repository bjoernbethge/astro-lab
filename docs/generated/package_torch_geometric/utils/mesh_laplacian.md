# mesh_laplacian

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.mesh_laplacian`

## Functions (4)

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

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
