# data

Part of `torch_geometric.data`
Module: `torch_geometric.data.data`

## Functions (9)

### `NamedTuple(typename, fields=None, /, **kwargs)`

Typed version of namedtuple.

Usage::

    class Employee(NamedTuple):
        name: str
        id: int

This is equivalent to::

    Employee = collections.namedtuple('Employee', ['name', 'id'])

The resulting class has an extra __annotations__ attribute, giving a
dict that maps field names to types.  (The field names are also in
the _fields attribute, which is part of the namedtuple API.)
An alternative equivalent functional syntax is also accepted::

    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

### `is_sparse(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is of type
:class:`torch.sparse.Tensor` (in any sparse layout) or of type
:class:`torch_sparse.SparseTensor`.

Args:
    src (Any): The input object to be checked.

### `overload(func)`

Decorator for overloaded functions/methods.

In a stub file, place two or more stub definitions for the same
function in a row, each decorated with @overload.

For example::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...

In a non-stub file (i.e. a regular .py file), do the same but
follow it with an implementation.  The implementation should *not*
be decorated with @overload::

    @overload
    def utf8(value: None) -> None: ...
    @overload
    def utf8(value: bytes) -> bytes: ...
    @overload
    def utf8(value: str) -> bytes: ...
    def utf8(value):
        ...  # implementation goes here

The overloads for a function can be retrieved at runtime using the
get_overloads() function.

### `select(src: Union[torch.Tensor, List[Any], torch_geometric.typing.TensorFrame], index_or_mask: torch.Tensor, dim: int) -> Union[torch.Tensor, List[Any]]`

Selects the input tensor or input list according to a given index or
mask vector.

Args:
    src (torch.Tensor or list): The input tensor or list.
    index_or_mask (torch.Tensor): The index or mask vector.
    dim (int): The dimension along which to select.

### `size_repr(key: Any, value: Any, indent: int = 0) -> str`

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

### `warn_or_raise(msg: str, raise_on_error: bool = True)`

## Classes (22)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `BaseData`

#### Methods

- **`stores_as(self, data: Self)`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

- **`to_namedtuple(self) -> <function NamedTuple at 0x000001FE17E66F20>`**
  Returns a :obj:`NamedTuple` of stored key/value pairs.

### `BaseStorage`

A MutableMapping is a generic container for associating
key/value pairs.

This class provides concrete generic implementations of all
methods except for __getitem__, __setitem__, __delitem__,
__iter__, and __len__.

#### Methods

- **`keys(self, *args: str) -> torch_geometric.data.view.KeysView`**
  D.keys() -> a set-like object providing a view on D's keys

- **`values(self, *args: str) -> torch_geometric.data.view.ValuesView`**
  D.values() -> an object providing a view on D's values

- **`items(self, *args: str) -> torch_geometric.data.view.ItemsView`**
  D.items() -> a set-like object providing a view on D's items

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

### `DataEdgeAttr`

Edge attribute class for `Data` without edge type.

### `DataTensorAttr`

Tensor attribute for `Data` without group name.

### `EdgeAttr`

Defines the attributes of a :obj:`GraphStore` edge.
It holds all the parameters necessary to uniquely identify an edge from
the :class:`GraphStore`.

Note that the order of the attributes is important; this is the order in
which attributes must be provided for indexing calls. :class:`GraphStore`
implementations can define a different ordering by overriding
:meth:`EdgeAttr.__init__`.

### `EdgeLayout`

Create a collection of name/value pairs.

Example enumeration:

>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access::

>>> Color.RED
<Color.RED: 1>

- value lookup:

>>> Color(1)
<Color.RED: 1>

- name lookup:

>>> Color['RED']
<Color.RED: 1>

Enumerations can be iterated over, and know how many members they have:

>>> len(Color)
3

>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.

### `EdgeStorage`

A storage for edge-level information.

We support multiple ways to store edge connectivity in a
:class:`EdgeStorage` object:

* :obj:`edge_index`: A :class:`torch.LongTensor` holding edge indices in
  COO format with shape :obj:`[2, num_edges]` (the default format)

* :obj:`adj`: A :class:`torch_sparse.SparseTensor` holding edge indices in
  a sparse format, supporting both COO and CSR format.

* :obj:`adj_t`: A **transposed** :class:`torch_sparse.SparseTensor` holding
  edge indices in a sparse format, supporting both COO and CSR format.
  This is the most efficient one for graph-based deep learning models as
  indices are sorted based on target nodes.

#### Methods

- **`size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

### `FeatureStore`

An abstract base class to access features from a remote feature store.

Args:
    tensor_attr_cls (TensorAttr, optional): A user-defined
        :class:`TensorAttr` class to customize the required attributes and
        their ordering to unique identify tensor values.
        (default: :obj:`None`)

#### Methods

- **`put_tensor(self, tensor: Union[torch.Tensor, numpy.ndarray], *args, **kwargs) -> bool`**
  Synchronously adds a :obj:`tensor` to the :class:`FeatureStore`.

- **`get_tensor(self, *args, convert_type: bool = False, **kwargs) -> Union[torch.Tensor, numpy.ndarray]`**
  Synchronously obtains a :class:`tensor` from the

- **`multi_get_tensor(self, attrs: List[torch_geometric.data.feature_store.TensorAttr], convert_type: bool = False) -> List[Union[torch.Tensor, numpy.ndarray]]`**
  Synchronously obtains a list of tensors from the

### `GlobalStorage`

A storage for both node-level and edge-level information.

#### Methods

- **`size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

### `GraphStore`

An abstract base class to access edges from a remote graph store.

Args:
    edge_attr_cls (EdgeAttr, optional): A user-defined
        :class:`EdgeAttr` class to customize the required attributes and
        their ordering to uniquely identify edges. (default: :obj:`None`)

#### Methods

- **`put_edge_index(self, edge_index: Tuple[torch.Tensor, torch.Tensor], *args, **kwargs) -> bool`**
  Synchronously adds an :obj:`edge_index` tuple to the

- **`get_edge_index(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]`**
  Synchronously obtains an :obj:`edge_index` tuple from the

- **`remove_edge_index(self, *args, **kwargs) -> bool`**
  Synchronously deletes an :obj:`edge_index` tuple from the

### `Index`

A one-dimensional :obj:`index` tensor with additional (meta)data
attached.

:class:`Index` is a :pytorch:`null` :class:`torch.Tensor` that holds
indices of shape :obj:`[num_indices]`.

While :class:`Index` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`dim_size`: The size of the underlying sparse vector size, *i.e.*,
  the size of a dimension that can be indexed via :obj:`index`.
  By default, it is inferred as :obj:`dim_size=index.max() + 1`.
* :obj:`is_sorted`: Whether indices are sorted in ascending order.

Additionally, :class:`Index` caches data via :obj:`indptr` for fast CSR
conversion in case its representation is sorted.
Caches are filled based on demand (*e.g.*, when calling
:meth:`Index.get_indptr`), or when explicitly requested via
:meth:`Index.fill_cache_`, and are maintaned and adjusted over its
lifespan.

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import Index

    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    >>> Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    assert index.dim_size == 3
    assert index.is_sorted

    # Flipping order:
    edge_index.flip(0)
    >>> Index([[2, 1, 1, 0], dim_size=3)
    assert not index.is_sorted

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    index[:, mask]
    >>> Index([[0, 1, 1], dim_size=3, is_sorted=True)
    assert index.is_sorted

#### Methods

- **`validate(self) -> 'Index'`**
  Validates the :class:`Index` representation.

- **`get_dim_size(self) -> int`**
  The size of the underlying sparse vector.

- **`dim_resize_(self, dim_size: Optional[int]) -> 'Index'`**
  Assigns or re-assigns the size of the underlying sparse vector.

### `Mapping`

A Mapping is a generic container for associating key/value
pairs.

This class provides concrete generic implementations of all
methods except for __getitem__, __iter__, and __len__.

#### Methods

- **`get(self, key, default=None)`**
  D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.

- **`keys(self)`**
  D.keys() -> a set-like object providing a view on D's keys

- **`items(self)`**
  D.items() -> a set-like object providing a view on D's items

### `NodeStorage`

A storage for node-level information.

#### Methods

- **`is_node_attr(self, key: str) -> bool`**

- **`is_edge_attr(self, key: str) -> bool`**

- **`node_attrs(self) -> List[str]`**

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

### `Sequence`

All the operations on a read-only sequence.

Concrete subclasses must override __new__ or __init__,
__getitem__, and __len__.

#### Methods

- **`index(self, value, start=0, stop=None)`**
  S.index(value, [start, [stop]]) -> integer -- return first index of value.

- **`count(self, value)`**
  S.count(value) -> integer -- return number of occurrences of value

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

### `TensorAttr`

Defines the attributes of a :class:`FeatureStore` tensor.
It holds all the parameters necessary to uniquely identify a tensor from
the :class:`FeatureStore`.

Note that the order of the attributes is important; this is the order in
which attributes must be provided for indexing calls. :class:`FeatureStore`
implementations can define a different ordering by overriding
:meth:`TensorAttr.__init__`.

#### Methods

- **`is_set(self, key: str) -> bool`**
  Whether an attribute is set in :obj:`TensorAttr`.

- **`is_fully_specified(self) -> bool`**
  Whether the :obj:`TensorAttr` has no unset fields.

- **`fully_specify(self) -> 'TensorAttr'`**
  Sets all :obj:`UNSET` fields to :obj:`None`.

### `TensorFrame`

### `chain`

chain(*iterables) --> chain object

Return a chain object whose .__next__() method returns elements from the
first iterable until it is exhausted, then elements from the next
iterable, until all of the iterables are exhausted.

#### Methods

- **`from_iterable(type, iterable, /)`**
  Alternative chain() constructor taking a single iterable argument that evaluates lazily.
