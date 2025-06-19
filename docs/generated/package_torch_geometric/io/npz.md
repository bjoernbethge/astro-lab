# npz

Part of `torch_geometric.io`
Module: `torch_geometric.io.npz`

## Functions (4)

### `parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> torch_geometric.data.data.Data`

### `read_npz(path: str, to_undirected: bool = True) -> torch_geometric.data.data.Data`

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

### `to_undirected_fn(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'add') -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

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

## Classes (2)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

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
