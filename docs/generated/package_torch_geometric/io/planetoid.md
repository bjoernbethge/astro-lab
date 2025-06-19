# planetoid

Part of `torch_geometric.io`
Module: `torch_geometric.io.planetoid`

## Functions (8)

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

### `edge_index_from_dict(graph_dict: Dict[int, List[int]], num_nodes: Optional[int] = None) -> torch.Tensor`

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

### `read_file(folder: str, prefix: str, name: str) -> torch.Tensor`

### `read_planetoid_data(folder: str, prefix: str) -> torch_geometric.data.data.Data`

### `read_txt_array(path: str, sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

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

## Classes (3)

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `repeat`

repeat(object [,times]) -> create an iterator which returns the object
for the specified number of times.  If not specified, returns the object
endlessly.
