# isolated

Part of `torch_geometric.experimental`
Module: `torch_geometric.utils.isolated`

## Functions (5)

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

### `maybe_num_nodes(edge_index: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch_geometric.typing.SparseTensor], num_nodes: Optional[int] = None) -> int`

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

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
