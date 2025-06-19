# hetero

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.hetero`

## Functions (7)

### `check_add_self_loops(module: torch.nn.modules.module.Module, edge_types: List[Tuple[str, str, str]]) -> None`

### `construct_bipartite_edge_index(edge_index_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]], src_offset_dict: Dict[Tuple[str, str, str], int], dst_offset_dict: Dict[str, int], edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None, num_nodes: Optional[int] = None) -> Tuple[Union[torch.Tensor, torch_geometric.typing.SparseTensor], Optional[torch.Tensor]]`

Constructs a tensor of edge indices by concatenating edge indices
for each edge type. The edge indices are increased by the offset of the
source and destination nodes.

Args:
    edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
        dictionary holding graph connectivity information for each
        individual edge type, either as a :class:`torch.Tensor` of
        shape :obj:`[2, num_edges]` or a
        :class:`torch_sparse.SparseTensor`.
    src_offset_dict (Dict[Tuple[str, str, str], int]): A dictionary of
        offsets to apply to the source node type for each edge type.
    dst_offset_dict (Dict[str, int]): A dictionary of offsets to apply for
        destination node types.
    edge_attr_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
        dictionary holding edge features for each individual edge type.
        (default: :obj:`None`)
    num_nodes (int, optional): The final number of nodes in the bipartite
        adjacency matrix. (default: :obj:`None`)

### `get_unused_node_types(node_types: List[str], edge_types: List[Tuple[str, str, str]]) -> Set[str]`

### `group_hetero_graph(edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], num_nodes_dict: Optional[Dict[str, int]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[Union[str, int], torch.Tensor], Dict[Union[str, Tuple[str, str, str]], int]]`

### `is_sparse(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is of type
:class:`torch.sparse.Tensor` (in any sparse layout) or of type
:class:`torch_sparse.SparseTensor`.

Args:
    src (Any): The input object to be checked.

### `maybe_num_nodes_dict(edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor], num_nodes_dict: Optional[Dict[str, int]] = None) -> Dict[str, int]`

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

## Classes (4)

### `NodeType`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

#### Methods

- **`encode(self, /, encoding='utf-8', errors='strict')`**
  Encode the string using the codec registered for encoding.

- **`replace(self, old, new, count=-1, /)`**
  Return a copy with all occurrences of substring old replaced by new.

- **`split(self, /, sep=None, maxsplit=-1)`**
  Return a list of the substrings in the string, using sep as the separator string.

### `ParameterDict`

Holds parameters in a dictionary.

ParameterDict can be indexed like a regular Python dictionary, but Parameters it
contains are properly registered, and will be visible by all Module methods.
Other objects are treated as would be done by a regular Python dictionary

:class:`~torch.nn.ParameterDict` is an **ordered** dictionary.
:meth:`~torch.nn.ParameterDict.update` with other unordered mapping
types (e.g., Python's plain ``dict``) does not preserve the order of the
merged mapping. On the other hand, ``OrderedDict`` or another :class:`~torch.nn.ParameterDict`
will preserve their ordering.

Note that the constructor, assigning an element of the dictionary and the
:meth:`~torch.nn.ParameterDict.update` method will convert any :class:`~torch.Tensor` into
:class:`~torch.nn.Parameter`.

Args:
    values (iterable, optional): a mapping (dictionary) of
        (string : Any) or an iterable of key-value pairs
        of type (string, Any)

Example::

    class MyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.params = nn.ParameterDict({
                    'left': nn.Parameter(torch.randn(5, 10)),
                    'right': nn.Parameter(torch.randn(5, 10))
            })

        def forward(self, x, choice):
            x = self.params[choice].mm(x)
            return x

#### Methods

- **`copy(self) -> 'ParameterDict'`**
  Return a copy of this :class:`~torch.nn.ParameterDict` instance.

- **`setdefault(self, key: str, default: Optional[Any] = None) -> Any`**
  Set the default for a key in the Parameterdict.

- **`clear(self) -> None`**
  Remove all items from the ParameterDict.

### `SparseTensor`

#### Methods

- **`size(self, dim: int) -> int`**

- **`nnz(self) -> int`**

- **`is_cuda(self) -> bool`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
