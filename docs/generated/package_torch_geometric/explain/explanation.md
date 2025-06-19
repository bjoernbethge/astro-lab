# explanation

Part of `torch_geometric.explain`
Module: `torch_geometric.explain.explanation`

## Functions (2)

### `visualize_graph(edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, path: Optional[str] = None, backend: Optional[str] = None, node_labels: Optional[List[str]] = None) -> Any`

Visualizes the graph given via :obj:`edge_index` and (optional)
:obj:`edge_weight`.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_weight (torch.Tensor, optional): The edge weights.
    path (str, optional): The path to where the plot is saved.
        If set to :obj:`None`, will visualize the plot on-the-fly.
        (default: :obj:`None`)
    backend (str, optional): The graph drawing backend to use for
        visualization (:obj:`"graphviz"`, :obj:`"networkx"`).
        If set to :obj:`None`, will use the most appropriate
        visualization backend based on available system packages.
        (default: :obj:`None`)
    node_labels (List[str], optional): The labels/IDs of nodes.
        (default: :obj:`None`)

### `warn_or_raise(msg: str, raise_on_error: bool = True)`

## Classes (9)

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

### `Explanation`

Holds all the obtained explanations of a homogeneous graph.

The explanation object is a :obj:`~torch_geometric.data.Data` object and
can hold node attributions and edge attributions.
It can also hold the original graph if needed.

Args:
    node_mask (Tensor, optional): Node-level mask with shape
        :obj:`[num_nodes, 1]`, :obj:`[1, num_features]` or
        :obj:`[num_nodes, num_features]`. (default: :obj:`None`)
    edge_mask (Tensor, optional): Edge-level mask with shape
        :obj:`[num_edges]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

#### Methods

- **`validate(self, raise_on_error: bool = True) -> bool`**
  Validates the correctness of the :class:`Explanation` object.

- **`get_explanation_subgraph(self) -> 'Explanation'`**
  Returns the induced subgraph, in which all nodes and edges with

- **`get_complement_subgraph(self) -> 'Explanation'`**
  Returns the induced subgraph, in which all nodes and edges with any

### `ExplanationMixin`

#### Methods

- **`validate_masks(self, raise_on_error: bool = True) -> bool`**
  Validates the correctness of the :class:`Explanation` masks.

- **`threshold(self, *args, **kwargs) -> Union[ForwardRef('Explanation'), ForwardRef('HeteroExplanation')]`**
  Thresholds the explanation masks according to the thresholding

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

### `HeteroExplanation`

Holds all the obtained explanations of a heterogeneous graph.

The explanation object is a :obj:`~torch_geometric.data.HeteroData` object
and can hold node attributions and edge attributions.
It can also hold the original graph if needed.

#### Methods

- **`validate(self, raise_on_error: bool = True) -> bool`**
  Validates the correctness of the :class:`Explanation` object.

- **`get_explanation_subgraph(self) -> 'HeteroExplanation'`**
  Returns the induced subgraph, in which all nodes and edges with

- **`get_complement_subgraph(self) -> 'HeteroExplanation'`**
  Returns the induced subgraph, in which all nodes and edges with any

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

### `ThresholdConfig`

Configuration class to store and validate threshold parameters.

Args:
    threshold_type (ThresholdType or str): The type of threshold to apply.
        The possible values are:

            - :obj:`None`: No threshold is applied.

            - :obj:`"hard"`: A hard threshold is applied to each mask.
              The elements of the mask with a value below the :obj:`value`
              are set to :obj:`0`, the others are set to :obj:`1`.

            - :obj:`"topk"`: A soft threshold is applied to each mask.
              The top obj:`value` elements of each mask are kept, the
              others are set to :obj:`0`.

            - :obj:`"topk_hard"`: Same as :obj:`"topk"` but values are set
              to :obj:`1` for all elements which are kept.

    value (int or float, optional): The value to use when thresholding.
        (default: :obj:`None`)

### `ThresholdType`

Enum class for the threshold type.
