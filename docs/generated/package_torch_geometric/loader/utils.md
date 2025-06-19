# utils

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.utils`

## Functions (10)

### `filter_custom_hetero_store(feature_store: torch_geometric.data.feature_store.FeatureStore, graph_store: torch_geometric.data.graph_store.GraphStore, node_dict: Dict[str, torch.Tensor], row_dict: Dict[str, torch.Tensor], col_dict: Dict[str, torch.Tensor], edge_dict: Dict[str, Optional[torch.Tensor]], custom_cls: Optional[torch_geometric.data.hetero_data.HeteroData] = None) -> torch_geometric.data.hetero_data.HeteroData`

Constructs a :class:`~torch_geometric.data.HeteroData` object from a
feature store and graph store instance.

### `filter_custom_store(feature_store: torch_geometric.data.feature_store.FeatureStore, graph_store: torch_geometric.data.graph_store.GraphStore, node: torch.Tensor, row: torch.Tensor, col: torch.Tensor, edge: Optional[torch.Tensor], custom_cls: Optional[torch_geometric.data.data.Data] = None) -> torch_geometric.data.data.Data`

Constructs a :class:`~torch_geometric.data.Data` object from a feature
store and graph store instance.

### `filter_data(data: torch_geometric.data.data.Data, node: torch.Tensor, row: torch.Tensor, col: torch.Tensor, edge: Optional[torch.Tensor], perm: Optional[torch.Tensor] = None) -> torch_geometric.data.data.Data`

### `filter_edge_store_(store: torch_geometric.data.storage.EdgeStorage, out_store: torch_geometric.data.storage.EdgeStorage, row: torch.Tensor, col: torch.Tensor, index: Optional[torch.Tensor], perm: Optional[torch.Tensor] = None)`

### `filter_hetero_data(data: torch_geometric.data.hetero_data.HeteroData, node_dict: Dict[str, torch.Tensor], row_dict: Dict[Tuple[str, str, str], torch.Tensor], col_dict: Dict[Tuple[str, str, str], torch.Tensor], edge_dict: Dict[Tuple[str, str, str], Optional[torch.Tensor]], perm_dict: Optional[Dict[Tuple[str, str, str], Optional[torch.Tensor]]] = None) -> torch_geometric.data.hetero_data.HeteroData`

### `filter_node_store_(store: torch_geometric.data.storage.NodeStorage, out_store: torch_geometric.data.storage.NodeStorage, index: torch.Tensor)`

### `get_edge_label_index(data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData, Tuple[torch_geometric.data.feature_store.FeatureStore, torch_geometric.data.graph_store.GraphStore]], edge_label_index: Union[torch.Tensor, NoneType, Tuple[str, str, str], Tuple[Tuple[str, str, str], Optional[torch.Tensor]]]) -> Tuple[Optional[str], torch.Tensor]`

### `get_input_nodes(data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData, Tuple[torch_geometric.data.feature_store.FeatureStore, torch_geometric.data.graph_store.GraphStore]], input_nodes: Union[torch.Tensor, NoneType, str, Tuple[str, Optional[torch.Tensor]], torch_geometric.data.feature_store.TensorAttr], input_id: Optional[torch.Tensor] = None) -> Tuple[Optional[str], torch.Tensor, Optional[torch.Tensor]]`

### `index_select(value: Union[torch.Tensor, numpy.ndarray], index: torch.Tensor, dim: int = 0) -> torch.Tensor`

Indexes the :obj:`value` tensor along dimension :obj:`dim` using the
entries in :obj:`index`.

Args:
    value (torch.Tensor or np.ndarray): The input tensor.
    index (torch.Tensor): The 1-D tensor containing the indices to index.
    dim (int, optional): The dimension in which to index.
        (default: :obj:`0`)

.. warning::

    :obj:`index` is casted to a :obj:`torch.int64` tensor internally, as
    `PyTorch currently only supports indexing
    <https://github.com/pytorch/pytorch/issues/61819>`_ via
    :obj:`torch.int64`.

### `infer_filter_per_worker(data: Any) -> bool`

## Classes (12)

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
