# random_node_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.random_node_loader`

## Functions (1)

### `to_homogeneous_edge_index(data: torch_geometric.data.hetero_data.HeteroData) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Dict[Tuple[str, str, str], Any]]`

Converts a heterogeneous graph into a homogeneous typed graph.

## Classes (4)

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

### `RandomNodeLoader`

A data loader that randomly samples nodes within a graph and returns
their induced subgraph.

.. note::

    For an example of using
    :class:`~torch_geometric.loader.RandomNodeLoader`, see
    `examples/ogbn_proteins_deepgcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    ogbn_proteins_deepgcn.py>`_.

Args:
    data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
        The :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` graph object.
    num_parts (int): The number of partitions.
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.

#### Methods

- **`collate_fn(self, index)`**

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
