# fake

Part of `torch_geometric.datasets`
Module: `torch_geometric.datasets.fake`

## Functions (6)

### `coalesce(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'sum', is_sorted: bool = False, sort_by_row: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
Duplicate entries in :obj:`edge_attr` are merged by scattering them
together according to the given :obj:`reduce` option.

Args:
    edge_index (torch.Tensor): The edge indices.
    edge_attr (torch.Tensor or List[torch.Tensor], optional): Edge weights
        or multi-dimensional edge features.
        If given as a list, will re-shuffle and remove duplicates for all
        its entries. (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`, :obj:`"any"`). (default: :obj:`"sum"`)
    is_sorted (bool, optional): If set to :obj:`True`, will expect
        :obj:`edge_index` to be already sorted row-wise.
    sort_by_row (bool, optional): If set to :obj:`False`, will sort
        :obj:`edge_index` column-wise.

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Example:
    >>> edge_index = torch.tensor([[1, 1, 2, 3],
    ...                            [3, 3, 1, 2]])
    >>> edge_attr = torch.tensor([1., 1., 1., 1.])
    >>> coalesce(edge_index)
    tensor([[1, 2, 3],
            [3, 1, 2]])

    >>> # Sort `edge_index` column-wise
    >>> coalesce(edge_index, sort_by_row=False)
    tensor([[2, 3, 1],
            [1, 2, 3]])

    >>> coalesce(edge_index, edge_attr)
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([2., 1., 1.]))

    >>> # Use 'mean' operation to merge edge features
    >>> coalesce(edge_index, edge_attr, reduce='mean')
    (tensor([[1, 2, 3],
            [3, 1, 2]]),
    tensor([1., 1., 1.]))

### `get_edge_index(num_src_nodes: int, num_dst_nodes: int, avg_degree: float, is_undirected: bool = False, remove_loops: bool = False) -> torch.Tensor`

### `get_num_channels(num_channels: int) -> int`

### `get_num_nodes(avg_num_nodes: int, avg_degree: float) -> int`

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

### `to_undirected(edge_index: torch.Tensor, edge_attr: Union[torch.Tensor, NoneType, List[torch.Tensor], str] = '???', num_nodes: Optional[int] = None, reduce: str = 'add') -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, List[torch.Tensor]]]`

Converts the graph given by :attr:`edge_index` to an undirected graph
such that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
\mathcal{E}`.

Args:
    edge_index (LongTensor): The edge indices.
    edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
        dimensional edge features.
        If given as a list, will remove duplicates for all its entries.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max(edge_index) + 1`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`). (default: :obj:`"add"`)

:rtype: :class:`LongTensor` if :attr:`edge_attr` is not passed, else
    (:class:`LongTensor`, :obj:`Optional[Tensor]` or :obj:`List[Tensor]]`)

.. warning::

    From :pyg:`PyG >= 2.3.0` onwards, this function will always return a
    tuple whenever :obj:`edge_attr` is passed as an argument (even in case
    it is set to :obj:`None`).

Examples:
    >>> edge_index = torch.tensor([[0, 1, 1],
    ...                            [1, 0, 2]])
    >>> to_undirected(edge_index)
    tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]])

    >>> edge_index = torch.tensor([[0, 1, 1],
    ...                            [1, 0, 2]])
    >>> edge_weight = torch.tensor([1., 1., 1.])
    >>> to_undirected(edge_index, edge_weight)
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([2., 2., 1., 1.]))

    >>> # Use 'mean' operation to merge edge features
    >>>  to_undirected(edge_index, edge_weight, reduce='mean')
    (tensor([[0, 1, 1, 2],
            [1, 0, 2, 1]]),
    tensor([1., 1., 1., 1.]))

## Classes (8)

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

### `FakeDataset`

A fake dataset that returns randomly generated
:class:`~torch_geometric.data.Data` objects.

Args:
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    avg_num_nodes (int, optional): The average number of nodes in a graph.
        (default: :obj:`1000`)
    avg_degree (float, optional): The average degree per node.
        (default: :obj:`10.0`)
    num_channels (int, optional): The number of node features.
        (default: :obj:`64`)
    edge_dim (int, optional): The number of edge features.
        (default: :obj:`0`)
    num_classes (int, optional): The number of classes in the dataset.
        (default: :obj:`10`)
    task (str, optional): Whether to return node-level or graph-level
        labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
        If set to :obj:`"auto"`, will return graph-level labels if
        :obj:`num_graphs > 1`, and node-level labels other-wise.
        (default: :obj:`"auto"`)
    is_undirected (bool, optional): Whether the graphs to generate are
        undirected. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    **kwargs (optional): Additional attributes and their shapes
        *e.g.* :obj:`global_features=5`.

#### Methods

- **`generate_data(self) -> torch_geometric.data.data.Data`**

### `FakeHeteroDataset`

A fake dataset that returns randomly generated
:class:`~torch_geometric.data.HeteroData` objects.

Args:
    num_graphs (int, optional): The number of graphs. (default: :obj:`1`)
    num_node_types (int, optional): The number of node types.
        (default: :obj:`3`)
    num_edge_types (int, optional): The number of edge types.
        (default: :obj:`6`)
    avg_num_nodes (int, optional): The average number of nodes in a graph.
        (default: :obj:`1000`)
    avg_degree (float, optional): The average degree per node.
        (default: :obj:`10.0`)
    avg_num_channels (int, optional): The average number of node features.
        (default: :obj:`64`)
    edge_dim (int, optional): The number of edge features.
        (default: :obj:`0`)
    num_classes (int, optional): The number of classes in the dataset.
        (default: :obj:`10`)
    task (str, optional): Whether to return node-level or graph-level
        labels (:obj:`"node"`, :obj:`"graph"`, :obj:`"auto"`).
        If set to :obj:`"auto"`, will return graph-level labels if
        :obj:`num_graphs > 1`, and node-level labels other-wise.
        (default: :obj:`"auto"`)
    transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.HeteroData` object and returns a
        transformed version. The data object will be transformed before
        every access. (default: :obj:`None`)
    **kwargs (optional): Additional attributes and their shapes
        *e.g.* :obj:`global_features=5`.

#### Methods

- **`generate_data(self) -> torch_geometric.data.hetero_data.HeteroData`**

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

### `InMemoryDataset`

Dataset base class for creating graph datasets which easily fit
into CPU memory.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html#creating-in-memory-datasets>`__ for the accompanying
tutorial.

Args:
    root (str, optional): Root directory where the dataset should be saved.
        (optional: :obj:`None`)
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        a :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before being saved to disk.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

#### Methods

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

- **`get(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the data object at index :obj:`idx`.

- **`load(self, path: str, data_cls: Type[torch_geometric.data.data.BaseData] = <class 'torch_geometric.data.data.Data'>) -> None`**
  Loads the dataset from the file path :obj:`path`.

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

### `product`

product(*iterables, repeat=1) --> product object

Cartesian product of input iterables.  Equivalent to nested for-loops.

For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B).
The leftmost iterators are in the outermost for-loop, so the output tuples
cycle in a manner similar to an odometer (with the rightmost element changing
on every iteration).

To compute the product of an iterable with itself, specify the number
of repetitions with the optional repeat keyword argument. For example,
product(A, repeat=4) means the same as product(A, A, A, A).

product('ab', range(3)) --> ('a',0) ('a',1) ('a',2) ('b',0) ('b',1) ('b',2)
product((0,1), (0,1), (0,1)) --> (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) ...
