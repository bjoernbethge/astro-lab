# laplacian

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.laplacian`

## Functions (5)

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

### `maybe_num_nodes(edge_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch_geometric.typing.SparseTensor], num_nodes: Optional[int] = None) -> int`

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

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
