# random_link_split

Part of `torch_geometric.transforms`
Module: `torch_geometric.transforms.random_link_split`

## Functions (2)

### `functional_transform(name: str) -> Callable`

### `negative_sampling(edge_index: torch.Tensor, num_nodes: Union[int, Tuple[int, int], NoneType] = None, num_neg_samples: Optional[int] = None, method: str = 'sparse', force_undirected: bool = False) -> torch.Tensor`

Samples random negative edges of a graph given by :attr:`edge_index`.

Args:
    edge_index (LongTensor): The edge indices.
    num_nodes (int or Tuple[int, int], optional): The number of nodes,
        *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
        If given as a tuple, then :obj:`edge_index` is interpreted as a
        bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
        (default: :obj:`None`)
    num_neg_samples (int, optional): The (approximate) number of negative
        samples to return.
        If set to :obj:`None`, will try to return a negative edge for every
        positive edge. (default: :obj:`None`)
    method (str, optional): The method to use for negative sampling,
        *i.e.* :obj:`"sparse"` or :obj:`"dense"`.
        This is a memory/runtime trade-off.
        :obj:`"sparse"` will work on any graph of any size, while
        :obj:`"dense"` can perform faster true-negative checks.
        (default: :obj:`"sparse"`)
    force_undirected (bool, optional): If set to :obj:`True`, sampled
        negative edges will be undirected. (default: :obj:`False`)

:rtype: LongTensor

Examples:
    >>> # Standard usage
    >>> edge_index = torch.as_tensor([[0, 0, 1, 2],
    ...                               [0, 1, 2, 3]])
    >>> negative_sampling(edge_index)
    tensor([[3, 0, 0, 3],
            [2, 3, 2, 1]])

    >>> # For bipartite graph
    >>> negative_sampling(edge_index, num_nodes=(3, 4))
    tensor([[0, 2, 2, 1],
            [2, 2, 1, 3]])

## Classes (6)

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

### `RandomLinkSplit`

Performs an edge-level random split into training, validation and test
sets of a :class:`~torch_geometric.data.Data` or a
:class:`~torch_geometric.data.HeteroData` object
(functional name: :obj:`random_link_split`).
The split is performed such that the training split does not include edges
in validation and test splits; and the validation split does not include
edges in the test split.

.. code-block:: python

    from torch_geometric.transforms import RandomLinkSplit

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)

Args:
    num_val (int or float, optional): The number of validation edges.
        If set to a floating-point value in :math:`[0, 1]`, it represents
        the ratio of edges to include in the validation set.
        (default: :obj:`0.1`)
    num_test (int or float, optional): The number of test edges.
        If set to a floating-point value in :math:`[0, 1]`, it represents
        the ratio of edges to include in the test set.
        (default: :obj:`0.2`)
    is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
        undirected, and positive and negative samples will not leak
        (reverse) edge connectivity across different splits. This only
        affects the graph split, label data will not be returned
        undirected. This option is ignored for bipartite edge types or
        whenever :obj:`edge_type != rev_edge_type`. (default: :obj:`False`)
    key (str, optional): The name of the attribute holding
        ground-truth labels.
        If :obj:`data[key]` does not exist, it will be automatically
        created and represents a binary classification task
        (:obj:`1` = edge, :obj:`0` = no edge).
        If :obj:`data[key]` exists, it has to be a categorical label from
        :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges. (default: :obj:`"edge_label"`)
    split_labels (bool, optional): If set to :obj:`True`, will split
        positive and negative labels and save them in distinct attributes
        :obj:`"pos_edge_label"` and :obj:`"neg_edge_label"`, respectively.
        (default: :obj:`False`)
    add_negative_train_samples (bool, optional): Whether to add negative
        training samples for link prediction.
        If the model already performs negative sampling, then the option
        should be set to :obj:`False`.
        Otherwise, the added negative samples will be the same across
        training iterations unless negative sampling is performed again.
        (default: :obj:`True`)
    neg_sampling_ratio (float, optional): The ratio of sampled negative
        edges to the number of positive edges. (default: :obj:`1.0`)
    disjoint_train_ratio (int or float, optional): If set to a value
        greater than :obj:`0.0`, training edges will not be shared for
        message passing and supervision. Instead,
        :obj:`disjoint_train_ratio` edges are used as ground-truth labels
        for supervision during training. (default: :obj:`0.0`)
    edge_types (Tuple[EdgeType] or List[EdgeType], optional): The edge
        types used for performing edge-level splitting in case of
        operating on :class:`~torch_geometric.data.HeteroData` objects.
        (default: :obj:`None`)
    rev_edge_types (Tuple[EdgeType] or List[Tuple[EdgeType]], optional):
        The reverse edge types of :obj:`edge_types` in case of operating
        on :class:`~torch_geometric.data.HeteroData` objects.
        This will ensure that edges of the reverse direction will be
        split accordingly to prevent any data leakage.
        Can be :obj:`None` in case no reverse connection exists.
        (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Tuple[Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData], Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData], Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]]`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
