# pad

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.pad`

## Functions (3)

### `abstractmethod(funcobj)`

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, arg1, arg2, argN):
            ...

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

### `functional_transform(name: str) -> Callable`

## Classes (15)

### `ABC`

Helper class that provides a standard way to create an ABC using
inheritance.

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `AttrNamePadding`

Padding dependent on attribute names.

Args:
    values (dict): The mapping from attribute names to padding values.
    default (int or float, optional): The padding value to use for
        attribute names not specified in :obj:`values`.
        (default: :obj:`0.0`)

#### Methods

- **`validate_key_value(self, key: Any, value: Any) -> None`**

- **`get_value(self, store_type: Union[str, Tuple[str, str, str], NoneType] = None, attr_name: Optional[str] = None) -> Union[int, float]`**

### `BaseTransform`

An abstract base class for writing transforms.

Transforms are a general way to modify and customize
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects, either by implicitly
passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or
by applying them explicitly to individual
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    dataset = TUDataset(path, name='MUTAG', transform=transform)
    data = dataset[0]  # Implicitly transform data on every access.

    data = TUDataset(path, name='MUTAG')[0]
    data = transform(data)  # Explicitly transform data.

#### Methods

- **`forward(self, data: Any) -> Any`**

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

### `EdgeTypePadding`

Padding dependent on node types.

Args:
    values (dict): The mapping from edge types to padding values.
    default (int or float, optional): The padding value to use for edge
        types not specified in :obj:`values`. (default: :obj:`0.0`)

#### Methods

- **`validate_key_value(self, key: Any, value: Any) -> None`**

- **`get_value(self, store_type: Union[str, Tuple[str, str, str], NoneType] = None, attr_name: Optional[str] = None) -> Union[int, float]`**

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

### `MappingPadding`

An abstract class for specifying different padding values.

#### Methods

- **`validate_key_value(self, key: Any, value: Any) -> None`**

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

### `NodeTypePadding`

Padding dependent on node types.

Args:
    values (dict): The mapping from node types to padding values.
    default (int or float, optional): The padding value to use for node
        types not specified in :obj:`values`. (default: :obj:`0.0`)

#### Methods

- **`validate_key_value(self, key: Any, value: Any) -> None`**

- **`get_value(self, store_type: Union[str, Tuple[str, str, str], NoneType] = None, attr_name: Optional[str] = None) -> Union[int, float]`**

### `Pad`

Applies padding to enforce consistent tensor shapes
(functional name: :obj:`pad`).

This transform will pad node and edge features up to a maximum allowed size
in the node or edge feature dimension. By default :obj:`0.0` is used as the
padding value and can be configured by setting :obj:`node_pad_value` and
:obj:`edge_pad_value`.

In case of applying :class:`Pad` to a :class:`~torch_geometric.data.Data`
object, the :obj:`node_pad_value` value (or :obj:`edge_pad_value`) can be
either:

* an int, float or object of :class:`UniformPadding` class for cases when
  all attributes are going to be padded with the same value;
* an object of :class:`AttrNamePadding` class for cases when padding is
  going to differ based on attribute names.

In case of applying :class:`Pad` to a
:class:`~torch_geometric.data.HeteroData` object, the :obj:`node_pad_value`
value (or :obj:`edge_pad_value`) can be either:

* an int, float or object of :class:`UniformPadding` class for cases when
  all attributes of all node (or edge) stores are going to be padded with
  the same value;
* an object of :class:`AttrNamePadding` class for cases when padding is
  going to differ based on attribute names (but not based on node or edge
  types);
* an object of class :class:`NodeTypePadding` or :class:`EdgeTypePadding`
  for cases when padding values are going to differ based on node or edge
  types. Padding values can also differ based on attribute names for a
  given node or edge type by using :class:`AttrNamePadding` objects as
  values of its `values` argument.

Note that in order to allow for consistent padding across all graphs in a
dataset, below conditions must be met:

* if :obj:`max_num_nodes` is a single value, it must be greater than or
  equal to the maximum number of nodes of any graph in the dataset;
* if :obj:`max_num_nodes` is a dictionary, value for every node type must
  be greater than or equal to the maximum number of this type nodes of any
  graph in the dataset.

Example below shows how to create a :class:`Pad` transform for an
:class:`~torch_geometric.data.HeteroData` object. The object is padded to
have :obj:`10` nodes of type :obj:`v0`, :obj:`20` nodes of type :obj:`v1`
and :obj:`30` nodes of type :obj:`v2`.
It is padded to have :obj:`80` edges of type :obj:`('v0', 'e0', 'v1')`.
All the attributes of the :obj:`v0` nodes are padded using a value of
:obj:`3.0`.
The :obj:`x` attribute of the :obj:`v1` node type is padded using a value
of :obj:`-1.0`, and the other attributes of this node type are padded using
a value of :obj:`0.5`.
All the attributes of node types other than :obj:`v0` and :obj:`v1` are
padded using a value of :obj:`1.0`.
All the attributes of the :obj:`('v0', 'e0', 'v1')` edge type are padded
using a value of :obj:`3.5`.
The :obj:`edge_attr` attributes of the :obj:`('v1', 'e0', 'v0')` edge type
are padded using a value of :obj:`-1.5`, and any other attributes of this
edge type are padded using a value of :obj:`5.5`.
All the attributes of edge types other than these two are padded using a
value of :obj:`1.5`.

.. code-block:: python

    num_nodes = {'v0': 10, 'v1': 20, 'v2':30}
    num_edges = {('v0', 'e0', 'v1'): 80}

    node_padding = NodeTypePadding({
        'v0': 3.0,
        'v1': AttrNamePadding({'x': -1.0}, default=0.5),
    }, default=1.0)

    edge_padding = EdgeTypePadding({
        ('v0', 'e0', 'v1'): 3.5,
        ('v1', 'e0', 'v0'): AttrNamePadding({'edge_attr': -1.5},
                                            default=5.5),
    }, default=1.5)

    transform = Pad(num_nodes, num_edges, node_padding, edge_padding)

Args:
    max_num_nodes (int or dict): The number of nodes after padding.
        In heterogeneous graphs, may also take in a dictionary denoting the
        number of nodes for specific node types.
    max_num_edges (int or dict, optional): The number of edges after
        padding.
        In heterogeneous graphs, may also take in a dictionary denoting the
        number of edges for specific edge types. (default: :obj:`None`)
    node_pad_value (int or float or Padding, optional): The fill value to
        use for node features. (default: :obj:`0.0`)
    edge_pad_value (int or float or Padding, optional): The fill value to
        use for edge features. (default: :obj:`0.0`)
        The :obj:`edge_index` tensor is padded with with the index of the
        first padded node (which represents a set of self-loops on the
        padded node). (default: :obj:`0.0`)
    mask_pad_value (bool, optional): The fill value to use for
        :obj:`train_mask`, :obj:`val_mask` and :obj:`test_mask` attributes
        (default: :obj:`False`).
    add_pad_mask (bool, optional): If set to :obj:`True`, will attach
        node-level :obj:`pad_node_mask` and edge-level :obj:`pad_edge_mask`
        attributes to the output which indicates which elements in the data
        are real (represented by :obj:`True`) and which were added as a
        result of padding (represented by :obj:`False`).
        (default: :obj:`False`)
    exclude_keys ([str], optional): Keys to be removed
        from the input data object. (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `Padding`

An abstract class for specifying padding values.

#### Methods

- **`get_value(self, store_type: Union[str, Tuple[str, str, str], NoneType] = None, attr_name: Optional[str] = None) -> Union[int, float]`**

### `UniformPadding`

Uniform padding independent of attribute name or node/edge type.

Args:
    value (int or float, optional): The value to be used for padding.
        (default: :obj:`0.0`)

#### Methods

- **`get_value(self, store_type: Union[str, Tuple[str, str, str], NoneType] = None, attr_name: Optional[str] = None) -> Union[int, float]`**
