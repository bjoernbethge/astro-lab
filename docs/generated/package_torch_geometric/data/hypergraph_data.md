# hypergraph_data

Part of `torch_geometric.data`
Module: `torch_geometric.data.hypergraph_data`

## Functions (3)

### `hyper_subgraph(subset: Union[torch.Tensor, List[int]], edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, relabel_nodes: bool = False, num_nodes: Optional[int] = None, return_edge_mask: bool = False) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]`

Returns the induced subgraph of the hyper graph of
:obj:`(edge_index, edge_attr)` containing the nodes in :obj:`subset`.

Args:
    subset (torch.Tensor or [int]): The nodes to keep.
    edge_index (LongTensor): Hyperedge tensor
        with shape :obj:`[2, num_edges*num_nodes_per_edge]`, where
        :obj:`edge_index[1]` denotes the hyperedge index and
        :obj:`edge_index[0]` denotes the node indices that are connected
        by the hyperedge.
    edge_attr (torch.Tensor, optional): Edge weights or multi-dimensional
        edge features of shape :obj:`[num_edges, *]`.
        (default: :obj:`None`)
    relabel_nodes (bool, optional): If set to :obj:`True`, the
        resulting :obj:`edge_index` will be relabeled to hold
        consecutive indices starting from zero. (default: :obj:`False`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index[0]) + 1`. (default: :obj:`None`)
    return_edge_mask (bool, optional): If set to :obj:`True`, will return
        the edge mask to filter out additional edge features.
        (default: :obj:`False`)

:rtype: (:class:`LongTensor`, :class:`Tensor`)

Examples:
    >>> edge_index = torch.tensor([[0, 1, 2, 1, 2, 3, 0, 2, 3],
    ...                            [0, 0, 0, 1, 1, 1, 2, 2, 2]])
    >>> edge_attr = torch.tensor([3, 2, 6])
    >>> subset = torch.tensor([0, 3])
    >>> subgraph(subset, edge_index, edge_attr)
    (tensor([[0, 3],
            [0, 0]]),
    tensor([ 6.]))

    >>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
    (tensor([[0, 3],
            [0, 0]]),
    tensor([ 6.]))
    tensor([False, False, True])

### `select(src: Union[torch.Tensor, List[Any], torch_geometric.typing.TensorFrame], index_or_mask: torch.Tensor, dim: int) -> Union[torch.Tensor, List[Any]]`

Selects the input tensor or input list according to a given index or
mask vector.

Args:
    src (torch.Tensor or list): The input tensor or list.
    index_or_mask (torch.Tensor): The index or mask vector.
    dim (int): The dimension along which to select.

### `warn_or_raise(msg: str, raise_on_error: bool = True) -> None`

## Classes (6)

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

### `HyperGraphData`

A data object describing a hypergraph.

The data object can hold node-level, link-level and graph-level attributes.
This object differs from a standard :obj:`~torch_geometric.data.Data`
object by having hyperedges, i.e. edges that connect more
than two nodes. For example, in the hypergraph scenario
:math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
:math:`\mathcal{V} = \{ 0, 1, 2, 3, 4 \}` and
:math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3, 4 \} \}`, the
hyperedge index :obj:`edge_index` is represented as:

.. code-block:: python

    # hyper graph with two hyperedges
    # connecting 3 and 4 nodes, respectively
    edge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3, 4],
        [0, 0, 0, 1, 1, 1, 1],
    ])

Args:
    x (torch.Tensor, optional): Node feature matrix with shape
        :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
    edge_index (LongTensor, optional): Hyperedge tensor
        with shape :obj:`[2, num_edges*num_nodes_per_edge]`.
        Where `edge_index[1]` denotes the hyperedge index and
        `edge_index[0]` denotes the node indicies that are connected
        by the hyperedge. (default: :obj:`None`)
        (default: :obj:`None`)
    edge_attr (torch.Tensor, optional): Edge feature matrix with shape
        :obj:`[num_edges, num_edge_features]`.
        (default: :obj:`None`)
    y (torch.Tensor, optional): Graph-level or node-level ground-truth
        labels with arbitrary shape. (default: :obj:`None`)
    pos (torch.Tensor, optional): Node position matrix with shape
        :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`is_edge_attr(self, key: str) -> bool`**
  Returns :obj:`True` if the object at key :obj:`key` denotes an

- **`subgraph(self, subset: torch.Tensor) -> 'HyperGraphData'`**
  Returns the induced subgraph given by the node indices

- **`edge_subgraph(self, subset: torch.Tensor) -> Self`**
  Returns the induced subgraph given by the edge indices

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
