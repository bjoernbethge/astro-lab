# data

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.data`

## Functions (8)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

### `download_google_url(id: str, folder: str, filename: str, log: bool = True)`

Downloads the content of a Google Drive ID to a specific folder.

### `download_url(url: str, folder: str, log: bool = True, filename: Optional[str] = None)`

Downloads the content of an URL to a specific folder.

Args:
    url (str): The URL.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)
    filename (str, optional): The filename of the downloaded file. If set
        to :obj:`None`, will correspond to the filename given by the URL.
        (default: :obj:`None`)

### `extract_bz2(path: str, folder: str, log: bool = True) -> None`

Extracts a bz2 archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_gz(path: str, folder: str, log: bool = True) -> None`

Extracts a gz archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_tar(path: str, folder: str, mode: str = 'r:gz', log: bool = True) -> None`

Extracts a tar archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    mode (str, optional): The compression mode. (default: :obj:`"r:gz"`)
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `extract_zip(path: str, folder: str, log: bool = True) -> None`

Extracts a zip archive to a specific folder.

Args:
    path (str): The path to the tar archive.
    folder (str): The folder.
    log (bool, optional): If :obj:`False`, will not print anything to the
        console. (default: :obj:`True`)

### `makedirs(path: str)`

Recursively creates a directory.

.. warning::

    :meth:`makedirs` is deprecated and will be removed soon.
    Please use :obj:`os.makedirs(path, exist_ok=True)` instead.

Args:
    path (str): The path to create.

## Classes (29)

### `Batch`

A data object describing a batch of graphs as one big (disconnected)
graph.
Inherits from :class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData`.
In addition, single graphs can be identified via the assignment vector
:obj:`batch`, which maps each node to its respective graph identifier.

:pyg:`PyG` allows modification to the underlying batching procedure by
overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
functionalities.
The :meth:`~Data.__inc__` method defines the incremental count between two
consecutive graph attributes.
By default, :pyg:`PyG` increments attributes by the number of nodes
whenever their attribute names contain the substring :obj:`index`
(for historical reasons), which comes in handy for attributes such as
:obj:`edge_index` or :obj:`node_index`.
However, note that this may lead to unexpected behavior for attributes
whose names contain the substring :obj:`index` but should not be
incremented.
To make sure, it is best practice to always double-check the output of
batching.
Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
tensors of the same attribute should be concatenated together.

#### Methods

- **`get_example(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the :class:`~torch_geometric.data.Data` or

- **`index_select(self, idx: Union[slice, torch.Tensor, numpy.ndarray, collections.abc.Sequence]) -> List[torch_geometric.data.data.BaseData]`**
  Creates a subset of :class:`~torch_geometric.data.Data` or

- **`to_data_list(self) -> List[torch_geometric.data.data.BaseData]`**
  Reconstructs the list of :class:`~torch_geometric.data.Data` or

### `ClusterData`

Clusters/partitions a graph data object into multiple subgraphs, as
motivated by the `"Cluster-GCN: An Efficient Algorithm for Training Deep
and Large Graph Convolutional Networks"
<https://arxiv.org/abs/1905.07953>`_ paper.

.. note::
    The underlying METIS algorithm requires undirected graphs as input.

Args:
    data (torch_geometric.data.Data): The graph data object.
    num_parts (int): The number of partitions.
    recursive (bool, optional): If set to :obj:`True`, will use multilevel
        recursive bisection instead of multilevel k-way partitioning.
        (default: :obj:`False`)
    save_dir (str, optional): If set, will save the partitioned data to the
        :obj:`save_dir` directory for faster re-use. (default: :obj:`None`)
    filename (str, optional): Name of the stored partitioned file.
        (default: :obj:`None`)
    log (bool, optional): If set to :obj:`False`, will not log any
        progress. (default: :obj:`True`)
    keep_inter_cluster_edges (bool, optional): If set to :obj:`True`,
        will keep inter-cluster edge connections. (default: :obj:`False`)
    sparse_format (str, optional): The sparse format to use for computing
        partitions. (default: :obj:`"csr"`)

### `ClusterLoader`

The data loader scheme from the `"Cluster-GCN: An Efficient Algorithm
for Training Deep and Large Graph Convolutional Networks"
<https://arxiv.org/abs/1905.07953>`_ paper which merges partioned subgraphs
and their between-cluster links from a large-scale graph data object to
form a mini-batch.

.. note::

    Use :class:`~torch_geometric.loader.ClusterData` and
    :class:`~torch_geometric.loader.ClusterLoader` in conjunction to
    form mini-batches of clusters.
    For an example of using Cluster-GCN, see
    `examples/cluster_gcn_reddit.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py>`_ or
    `examples/cluster_gcn_ppi.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py>`_.

Args:
    cluster_data (torch_geometric.loader.ClusterData): The already
        partioned data object.
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

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

### `DataListLoader`

A data loader which batches data objects from a
:class:`torch_geometric.data.dataset` to a :python:`Python` list.
Data objects can be either of type :class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData`.

.. note::

    This data loader should be used for multi-GPU support via
    :class:`torch_geometric.nn.DataParallel`.

Args:
    dataset (Dataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    shuffle (bool, optional): If set to :obj:`True`, the data will be
        reshuffled at every epoch. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
        :obj:`num_workers`.

### `DataLoader`

A data loader which merges data objects from a
:class:`torch_geometric.data.Dataset` to a mini-batch.
Data objects can be either of type :class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData`.

Args:
    dataset (Dataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    shuffle (bool, optional): If set to :obj:`True`, the data will be
        reshuffled at every epoch. (default: :obj:`False`)
    follow_batch (List[str], optional): Creates assignment batch
        vectors for each key in the list. (default: :obj:`None`)
    exclude_keys (List[str], optional): Will exclude each key in the
        list. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`.

### `Database`

Base class for inserting and retrieving data from a database.

A database acts as a persisted, out-of-memory and index-based key/value
store for tensor and custom data:

.. code-block:: python

    db = Database()
    db[0] = Data(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> Data(x=[5, 16], y=0, z='id_0')

To improve efficiency, it is recommended to specify the underlying
:obj:`schema` of the data:

.. code-block:: python

    db = Database(schema={  # Custom schema:
        # Tensor information can be specified through a dictionary:
        'x': dict(dtype=torch.float, size=(-1, 16)),
        'y': int,
        'z': str,
    })
    db[0] = dict(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> {'x': torch.tensor(...), 'y': 0, 'z': 'id_0'}

In addition, databases support batch-wise insert and get, and support
syntactic sugar known from indexing :python:`Python` lists, *e.g.*:

.. code-block:: python

    db = Database()
    db[2:5] = torch.randn(3, 16)
    print(db[torch.tensor([2, 3])])
    >>> [torch.tensor(...), torch.tensor(...)]

Args:
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

### `Dataset`

Dataset base class for creating graph datasets.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_dataset.html>`__ for the accompanying tutorial.

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

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

- **`len(self) -> int`**
  Returns the number of data objects stored in the dataset.

### `DenseDataLoader`

A data loader which batches data objects from a
:class:`torch_geometric.data.dataset` to a
:class:`torch_geometric.data.Batch` object by stacking all attributes in a
new dimension.

.. note::

    To make use of this data loader, all graph attributes in the dataset
    need to have the same shape.
    In particular, this data loader should only be used when working with
    *dense* adjacency matrices.

Args:
    dataset (Dataset): The dataset from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    shuffle (bool, optional): If set to :obj:`True`, the data will be
        reshuffled at every epoch. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`drop_last` or
        :obj:`num_workers`.

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

### `GraphSAINTEdgeSampler`

The GraphSAINT edge sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

### `GraphSAINTNodeSampler`

The GraphSAINT node sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

### `GraphSAINTRandomWalkSampler`

The GraphSAINT random walk sampler class (see
:class:`~torch_geometric.loader.GraphSAINTSampler`).

Args:
    walk_length (int): The length of each random walk.

### `GraphSAINTSampler`

The GraphSAINT sampler base class from the `"GraphSAINT: Graph
Sampling Based Inductive Learning Method"
<https://arxiv.org/abs/1907.04931>`_ paper.
Given a graph in a :obj:`data` object, this class samples nodes and
constructs subgraphs that can be processed in a mini-batch fashion.
Normalization coefficients for each mini-batch are given via
:obj:`node_norm` and :obj:`edge_norm` data attributes.

.. note::

    See :class:`~torch_geometric.loader.GraphSAINTNodeSampler`,
    :class:`~torch_geometric.loader.GraphSAINTEdgeSampler` and
    :class:`~torch_geometric.loader.GraphSAINTRandomWalkSampler` for
    currently supported samplers.
    For an example of using GraphSAINT sampling, see
    `examples/graph_saint.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/graph_saint.py>`_.

Args:
    data (torch_geometric.data.Data): The graph data object.
    batch_size (int): The approximate number of samples per batch.
    num_steps (int, optional): The number of iterations per epoch.
        (default: :obj:`1`)
    sample_coverage (int): How many samples per node should be used to
        compute normalization statistics. (default: :obj:`0`)
    save_dir (str, optional): If set, will save normalization statistics to
        the :obj:`save_dir` directory for faster re-use.
        (default: :obj:`None`)
    log (bool, optional): If set to :obj:`False`, will not log any
        pre-processing progress. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
        :obj:`num_workers`.

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

### `LazyLoader`

Create a module object.

The name must be a string; the optional doc argument can have any type.

### `NeighborSampler`

The neighbor sampler from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
for mini-batch training of GNNs on large-scale graphs where full-batch
training is not feasible.

Given a GNN with :math:`L` layers and a specific mini-batch of nodes
:obj:`node_idx` for which we want to compute embeddings, this module
iteratively samples neighbors and constructs bipartite graphs that simulate
the actual computation flow of GNNs.

More specifically, :obj:`sizes` denotes how much neighbors we want to
sample for each node in each layer.
This module then takes in these :obj:`sizes` and iteratively samples
:obj:`sizes[l]` for each node involved in layer :obj:`l`.
In the next layer, sampling is repeated for the union of nodes that were
already encountered.
The actual computation graphs are then returned in reverse-mode, meaning
that we pass messages from a larger set of nodes to a smaller one, until we
reach the nodes for which we originally wanted to compute embeddings.

Hence, an item returned by :class:`NeighborSampler` holds the current
:obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
computation, and a list of bipartite graph objects via the tuple
:obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
bipartite edges between source and target nodes, :obj:`e_id` denotes the
IDs of original edges in the full graph, and :obj:`size` holds the shape
of the bipartite graph.
For each bipartite graph, target nodes are also included at the beginning
of the list of source nodes so that one can easily apply skip-connections
or add self-loops.

.. warning::

    :class:`~torch_geometric.loader.NeighborSampler` is deprecated and will
    be removed in a future release.
    Use :class:`torch_geometric.loader.NeighborLoader` instead.

.. note::

    For an example of using :obj:`NeighborSampler`, see
    `examples/reddit.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    reddit.py>`_ or
    `examples/ogbn_products_sage.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    ogbn_products_sage.py>`_.

Args:
    edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
        :class:`torch_sparse.SparseTensor` that defines the underlying
        graph connectivity/message passing flow.
        :obj:`edge_index` holds the indices of a (sparse) symmetric
        adjacency matrix.
        If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
        must be defined as :obj:`[2, num_edges]`, where messages from nodes
        :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
        (in case :obj:`flow="source_to_target"`).
        If :obj:`edge_index` is of type :class:`torch_sparse.SparseTensor`,
        its sparse indices :obj:`(row, col)` should relate to
        :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
        The major difference between both formats is that we need to input
        the *transposed* sparse adjacency matrix.
    sizes ([int]): The number of neighbors to sample for each node in each
        layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
        in layer :obj:`l`.
    node_idx (LongTensor, optional): The nodes that should be considered
        for creating mini-batches. If set to :obj:`None`, all nodes will be
        considered.
    num_nodes (int, optional): The number of nodes in the graph.
        (default: :obj:`None`)
    return_e_id (bool, optional): If set to :obj:`False`, will not return
        original edge indices of sampled edges. This is only useful in case
        when operating on graphs without edge features to save memory.
        (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

### `OnDiskDataset`

Dataset base class for creating large graph datasets which do not
easily fit into CPU memory at once by leveraging a :class:`Database`
backend for on-disk storage and access of data objects.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        transformed version.
        The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object and returns a
        boolean value, indicating whether the data object should be
        included in the final dataset. (default: :obj:`None`)
    backend (str): The :class:`Database` backend to use
        (one of :obj:`"sqlite"` or :obj:`"rocksdb"`).
        (default: :obj:`"sqlite"`)
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. If specified to anything different than
        :obj:`object`, implementations of :class:`OnDiskDataset` need to
        override :meth:`serialize` and :meth:`deserialize` methods.
        (default: :obj:`object`)
    log (bool, optional): Whether to print any console output while
        downloading and processing the dataset. (default: :obj:`True`)

#### Methods

- **`close(self) -> None`**
  Closes the connection to the underlying database.

- **`serialize(self, data: torch_geometric.data.data.BaseData) -> Any`**
  Serializes the :class:`~torch_geometric.data.Data` or

- **`deserialize(self, data: Any) -> torch_geometric.data.data.BaseData`**
  Deserializes the DB entry into a

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

### `RandomNodeSampler`

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

### `RocksDatabase`

An index-based key/value database based on :obj:`RocksDB`.

.. note::
    This database implementation requires the :obj:`rocksdict` package.

.. warning::
    :class:`RocksDatabase` is currently less optimized than
    :class:`SQLiteDatabase`.

Args:
    path (str): The path to where the database should be saved.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`to_key(index: int) -> bytes`**

### `SQLiteDatabase`

An index-based key/value database based on :obj:`sqlite3`.

.. note::
    This database implementation requires the :obj:`sqlite3` package.

Args:
    path (str): The path to where the database should be saved.
    name (str): The name of the table to save the data to.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

### `ShaDowKHopSampler`

The ShaDow :math:`k`-hop sampler from the `"Decoupling the Depth and
Scope of Graph Neural Networks" <https://arxiv.org/abs/2201.07858>`_ paper.
Given a graph in a :obj:`data` object, the sampler will create shallow,
localized subgraphs.
A deep GNN on this local graph then smooths the informative local signals.

.. note::

    For an example of using :class:`ShaDowKHopSampler`, see
    `examples/shadow.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/shadow.py>`_.

Args:
    data (torch_geometric.data.Data): The graph data object.
    depth (int): The depth/number of hops of the localized subgraph.
    num_neighbors (int): The number of neighbors to sample for each node in
        each hop.
    node_idx (LongTensor or BoolTensor, optional): The nodes that should be
        considered for creating mini-batches.
        If set to :obj:`None`, all nodes will be
        considered.
    replace (bool, optional): If set to :obj:`True`, will sample neighbors
        with replacement. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size` or
        :obj:`num_workers`.

### `TemporalData`

A data object composed by a stream of events describing a temporal
graph.
The :class:`~torch_geometric.data.TemporalData` object can hold a list of
events (that can be understood as temporal edges in a graph) with
structured messages.
An event is composed by a source node, a destination node, a timestamp
and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
represented with these four values.

In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
the behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block:: python

    from torch import Tensor
    from torch_geometric.data import TemporalData

    events = TemporalData(
        src=Tensor([1,2,3,4]),
        dst=Tensor([2,3,4,5]),
        t=Tensor([1000,1010,1100,2000]),
        msg=Tensor([1,1,0,0])
    )

    # Add additional arguments to `events`:
    events.y = Tensor([1,1,0,0])

    # It is also possible to set additional arguments in the constructor
    events = TemporalData(
        ...,
        y=Tensor([1,1,0,0])
    )

    # Get the number of events:
    events.num_events
    >>> 4

    # Analyzing the graph structure:
    events.num_nodes
    >>> 5

    # PyTorch tensor functionality:
    events = events.pin_memory()
    events = events.to('cuda:0', non_blocking=True)

Args:
    src (torch.Tensor, optional): A list of source nodes for the events
        with shape :obj:`[num_events]`. (default: :obj:`None`)
    dst (torch.Tensor, optional): A list of destination nodes for the
        events with shape :obj:`[num_events]`. (default: :obj:`None`)
    t (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_events]`. (default: :obj:`None`)
    msg (torch.Tensor, optional): Messages feature matrix with shape
        :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

.. note::
    The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
    of :obj`msg` should be the same (:obj:`num_events`).

#### Methods

- **`index_select(self, idx: Any) -> 'TemporalData'`**

- **`stores_as(self, data: 'TemporalData')`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

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
