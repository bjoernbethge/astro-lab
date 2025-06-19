# hetero_data

Part of `torch_geometric.data`
Module: `torch_geometric.data.hetero_data`

## Functions (12)

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

### `get_node_slices(num_nodes: Dict[str, int]) -> Dict[str, Tuple[int, int]]`

Returns the boundaries of each node type in a graph.

### `is_sparse(src: Any) -> bool`

Returns :obj:`True` if the input :obj:`src` is of type
:class:`torch.sparse.Tensor` (in any sparse layout) or of type
:class:`torch_sparse.SparseTensor`.

Args:
    src (Any): The input object to be checked.

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

### `mask_select(src: torch.Tensor, dim: int, mask: torch.Tensor) -> torch.Tensor`

Returns a new tensor which masks the :obj:`src` tensor along the
dimension :obj:`dim` according to the boolean mask :obj:`mask`.

Args:
    src (torch.Tensor): The input tensor.
    dim (int): The dimension in which to mask.
    mask (torch.BoolTensor): The 1-D tensor containing the binary mask to
        index with.

### `namedtuple(typename, field_names, *, rename=False, defaults=None, module=None)`

Returns a new subclass of tuple with named fields.

>>> Point = namedtuple('Point', ['x', 'y'])
>>> Point.__doc__                   # docstring for the new class
'Point(x, y)'
>>> p = Point(11, y=22)             # instantiate with positional args or keywords
>>> p[0] + p[1]                     # indexable like a plain tuple
33
>>> x, y = p                        # unpack like a regular tuple
>>> x, y
(11, 22)
>>> p.x + p.y                       # fields also accessible by name
33
>>> d = p._asdict()                 # convert to a dictionary
>>> d['x']
11
>>> Point(**d)                      # convert from a dictionary
Point(x=11, y=22)
>>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
Point(x=100, y=22)

### `offset_edge_index(node_slices: Dict[str, Tuple[int, int]], edge_type: Tuple[str, str, str], edge_index: torch.Tensor) -> torch.Tensor`

Increases the edge indices by the offsets of source and destination
node types.

### `size_repr(key: Any, value: Any, indent: int = 0) -> str`

### `to_homogeneous_edge_index(data: torch_geometric.data.hetero_data.HeteroData) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Dict[Tuple[str, str, str], Any]]`

Converts a heterogeneous graph into a homogeneous typed graph.

### `warn_or_raise(msg: str, raise_on_error: bool = True)`

## Classes (21)

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

### `HeteroData`

A data object describing a heterogeneous graph, holding multiple node
and/or edge types in disjunct storage objects.
Storage objects can hold either node-level, link-level or graph-level
attributes.
In general, :class:`~torch_geometric.data.HeteroData` tries to mimic the
behavior of a regular **nested** :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block::

    from torch_geometric.data import HeteroData

    data = HeteroData()

    # Create two node types "paper" and "author" holding a feature matrix:
    data['paper'].x = torch.randn(num_papers, num_paper_features)
    data['author'].x = torch.randn(num_authors, num_authors_features)

    # Create an edge type "(author, writes, paper)" and building the
    # graph connectivity:
    data['author', 'writes', 'paper'].edge_index = ...  # [2, num_edges]

    data['paper'].num_nodes
    >>> 23

    data['author', 'writes', 'paper'].num_edges
    >>> 52

    # PyTorch tensor functionality:
    data = data.pin_memory()
    data = data.to('cuda:0', non_blocking=True)

Note that there exists multiple ways to create a heterogeneous graph data,
*e.g.*:

* To initialize a node of type :obj:`"paper"` holding a node feature
  matrix :obj:`x_paper` named :obj:`x`:

  .. code-block:: python

    from torch_geometric.data import HeteroData

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['paper'].x = x_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(paper={ 'x': x_paper })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({'paper': { 'x': x_paper }})

* To initialize an edge from source node type :obj:`"author"` to
  destination node type :obj:`"paper"` with relation type :obj:`"writes"`
  holding a graph connectivity matrix :obj:`edge_index_author_paper` named
  :obj:`edge_index`:

  .. code-block:: python

    # (1) Assign attributes after initialization,
    data = HeteroData()
    data['author', 'writes', 'paper'].edge_index = edge_index_author_paper

    # or (2) pass them as keyword arguments during initialization,
    data = HeteroData(author__writes__paper={
        'edge_index': edge_index_author_paper
    })

    # or (3) pass them as dictionaries during initialization,
    data = HeteroData({
        ('author', 'writes', 'paper'):
        { 'edge_index': edge_index_author_paper }
    })

#### Methods

- **`stores_as(self, data: Self)`**

- **`node_items(self) -> List[Tuple[str, torch_geometric.data.storage.NodeStorage]]`**
  Returns a list of node type and node storage pairs.

- **`edge_items(self) -> List[Tuple[Tuple[str, str, str], torch_geometric.data.storage.EdgeStorage]]`**
  Returns a list of edge type and edge storage pairs.

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

### `defaultdict`

defaultdict(default_factory=None, /, [...]) --> dict with default factory

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.

#### Methods

- **`copy(...)`**
  D.copy() -> a shallow copy of D.

### `torch_frame`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.
