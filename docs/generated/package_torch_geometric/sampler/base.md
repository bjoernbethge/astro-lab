# base

Part of `torch_geometric.sampler`
Module: `torch_geometric.sampler.base`

## Functions (2)

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

### `to_bidirectional(row: torch.Tensor, col: torch.Tensor, rev_row: torch.Tensor, rev_col: torch.Tensor, edge_id: Optional[torch.Tensor] = None, rev_edge_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`

## Classes (22)

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

### `BaseSampler`

An abstract base class that initializes a graph sampler and provides
:meth:`sample_from_nodes` and :meth:`sample_from_edges` routines.

.. note ::

    Any data stored in the sampler will be *replicated* across data loading
    workers that use the sampler since each data loading worker holds its
    own instance of a sampler.
    As such, it is recommended to limit the amount of information stored in
    the sampler.

#### Methods

- **`sample_from_nodes(self, index: torch_geometric.sampler.base.NodeSamplerInput, **kwargs) -> Union[torch_geometric.sampler.base.HeteroSamplerOutput, torch_geometric.sampler.base.SamplerOutput]`**
  Performs sampling from the nodes specified in :obj:`index`,

- **`sample_from_edges(self, index: torch_geometric.sampler.base.EdgeSamplerInput, neg_sampling: Optional[torch_geometric.sampler.base.NegativeSampling] = None) -> Union[torch_geometric.sampler.base.HeteroSamplerOutput, torch_geometric.sampler.base.SamplerOutput]`**
  Performs sampling from the edges specified in :obj:`index`,

### `CastMixin`

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

### `DataType`

The data type a sampler is operating on.

### `EdgeSamplerInput`

The sampling input of
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

Args:
    input_id (torch.Tensor, optional): The indices of the data loader input
        of the current mini-batch.
    row (torch.Tensor): The source node indices of seed links to start
        sampling from.
    col (torch.Tensor): The destination node indices of seed links to start
        sampling from.
    label (torch.Tensor, optional): The label for the seed links.
        (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamp for the seed links.
        (default: :obj:`None`)
    input_type (Tuple[str, str, str], optional): The input edge type (in
        case of sampling in a heterogeneous graph). (default: :obj:`None`)

### `EdgeTypeStr`

A helper class to construct serializable edge types by merging an edge
type tuple into a single string.

#### Methods

- **`to_tuple(self) -> Tuple[str, str, str]`**
  Returns the original edge type.

### `Enum`

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

### `HeteroSamplerOutput`

The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
on heterogeneous graphs.

Args:
    node (Dict[str, torch.Tensor]): The sampled nodes in the original graph
        for each node type.
    row (Dict[Tuple[str, str, str], torch.Tensor]): The source node indices
        of the sampled subgraph for each edge type.
        Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
        corresponding to the nodes in the :obj:`node` tensor of the source
        node type.
    col (Dict[Tuple[str, str, str], torch.Tensor]): The destination node
        indices of the sampled subgraph for each edge type.
        Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
        corresponding to the nodes in the :obj:`node` tensor of the
        destination node type.
    edge (Dict[Tuple[str, str, str], torch.Tensor], optional): The sampled
        edges in the original graph for each edge type.
        This tensor is used to obtain edge features from the original
        graph. If no edge attributes are present, it may be omitted.
    batch (Dict[str, torch.Tensor], optional): The vector to identify the
        seed node for each sampled node for each node type. Can be present
        in case of disjoint subgraph sampling per seed node.
        (default: :obj:`None`)
    num_sampled_nodes (Dict[str, List[int]], optional): The number of
        sampled nodes for each node type and each layer.
        (default: :obj:`None`)
    num_sampled_edges (Dict[EdgeType, List[int]], optional): The number of
        sampled edges for each edge type and each layer.
        (default: :obj:`None`)
    orig_row (Dict[EdgeType, torch.Tensor], optional): The original source
        node indices returned by the sampler.
        Filled in case :meth:`to_bidirectional` is called with the
        :obj:`keep_orig_edges` option. (default: :obj:`None`)
    orig_col (Dict[EdgeType, torch.Tensor], optional): The original
        destination node indices returned by the sampler.
        Filled in case :meth:`to_bidirectional` is called with the
        :obj:`keep_orig_edges` option. (default: :obj:`None`)
    metadata: (Any, optional): Additional metadata information.
        (default: :obj:`None`)

#### Methods

- **`to_bidirectional(self, keep_orig_edges: bool = False) -> 'SamplerOutput'`**
  Converts the sampled subgraph into a bidirectional variant, in

### `NegativeSampling`

The negative sampling configuration of a
:class:`~torch_geometric.sampler.BaseSampler` when calling
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

Args:
    mode (str): The negative sampling mode
        (:obj:`"binary"` or :obj:`"triplet"`).
        If set to :obj:`"binary"`, will randomly sample negative links
        from the graph.
        If set to :obj:`"triplet"`, will randomly sample negative
        destination nodes for each positive source node.
    amount (int or float, optional): The ratio of sampled negative edges to
        the number of positive edges. (default: :obj:`1`)
    src_weight (torch.Tensor, optional): A node-level vector determining
        the sampling of source nodes. Does not necessarily need to sum up
        to one. If not given, negative nodes will be sampled uniformly.
        (default: :obj:`None`)
    dst_weight (torch.Tensor, optional): A node-level vector determining
        the sampling of destination nodes. Does not necessarily need to sum
        up to one. If not given, negative nodes will be sampled uniformly.
        (default: :obj:`None`)

#### Methods

- **`is_binary(self) -> bool`**

- **`is_triplet(self) -> bool`**

- **`sample(self, num_samples: int, endpoint: Literal['src', 'dst'], num_nodes: Optional[int] = None) -> torch.Tensor`**
  Generates :obj:`num_samples` negative samples.

### `NegativeSamplingMode`

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

### `NodeSamplerInput`

The sampling input of
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.

Args:
    input_id (torch.Tensor, optional): The indices of the data loader input
        of the current mini-batch.
    node (torch.Tensor): The indices of seed nodes to start sampling from.
    time (torch.Tensor, optional): The timestamp for the seed nodes.
        (default: :obj:`None`)
    input_type (str, optional): The input node type (in case of sampling in
        a heterogeneous graph). (default: :obj:`None`)

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

### `NumNeighbors`

The number of neighbors to sample in a homogeneous or heterogeneous
graph. In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for individual edge types.

Args:
    values (List[int] or Dict[Tuple[str, str, str], List[int]]): The
        number of neighbors to sample.
        If an entry is set to :obj:`-1`, all neighbors will be included.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for individual edge types.
    default (List[int], optional): The default number of neighbors for edge
        types not specified in :obj:`values`. (default: :obj:`None`)

#### Methods

- **`get_values(self, edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[Tuple[str, str, str], List[int]]]`**
  Returns the number of neighbors.

- **`get_mapped_values(self, edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[str, List[int]]]`**
  Returns the number of neighbors.

### `SamplerOutput`

The sampling output of a :class:`~torch_geometric.sampler.BaseSampler`
on homogeneous graphs.

Args:
    node (torch.Tensor): The sampled nodes in the original graph.
    row (torch.Tensor): The source node indices of the sampled subgraph.
        Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
        corresponding to the nodes in the :obj:`node` tensor.
    col (torch.Tensor): The destination node indices of the sampled
        subgraph.
        Indices must be re-indexed to :obj:`{ 0, ..., num_nodes - 1 }`
        corresponding to the nodes in the :obj:`node` tensor.
    edge (torch.Tensor, optional): The sampled edges in the original graph.
        This tensor is used to obtain edge features from the original
        graph. If no edge attributes are present, it may be omitted.
    batch (torch.Tensor, optional): The vector to identify the seed node
        for each sampled node. Can be present in case of disjoint subgraph
        sampling per seed node. (default: :obj:`None`)
    num_sampled_nodes (List[int], optional): The number of sampled nodes
        per hop. (default: :obj:`None`)
    num_sampled_edges (List[int], optional): The number of sampled edges
        per hop. (default: :obj:`None`)
    orig_row (torch.Tensor, optional): The original source node indices
        returned by the sampler.
        Filled in case :meth:`to_bidirectional` is called with the
        :obj:`keep_orig_edges` option. (default: :obj:`None`)
    orig_col (torch.Tensor, optional): The original destination node
        indices indices returned by the sampler.
        Filled in case :meth:`to_bidirectional` is called with the
        :obj:`keep_orig_edges` option. (default: :obj:`None`)
    metadata: (Any, optional): Additional metadata information.
        (default: :obj:`None`)

#### Methods

- **`to_bidirectional(self, keep_orig_edges: bool = False) -> 'SamplerOutput'`**
  Converts the sampled subgraph into a bidirectional variant, in

### `SubgraphType`

The type of the returned subgraph.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

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
