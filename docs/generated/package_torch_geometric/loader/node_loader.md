# node_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.node_loader`

## Functions (6)

### `filter_custom_hetero_store(feature_store: torch_geometric.data.feature_store.FeatureStore, graph_store: torch_geometric.data.graph_store.GraphStore, node_dict: Dict[str, torch.Tensor], row_dict: Dict[str, torch.Tensor], col_dict: Dict[str, torch.Tensor], edge_dict: Dict[str, Optional[torch.Tensor]], custom_cls: Optional[torch_geometric.data.hetero_data.HeteroData] = None) -> torch_geometric.data.hetero_data.HeteroData`

Constructs a :class:`~torch_geometric.data.HeteroData` object from a
feature store and graph store instance.

### `filter_custom_store(feature_store: torch_geometric.data.feature_store.FeatureStore, graph_store: torch_geometric.data.graph_store.GraphStore, node: torch.Tensor, row: torch.Tensor, col: torch.Tensor, edge: Optional[torch.Tensor], custom_cls: Optional[torch_geometric.data.data.Data] = None) -> torch_geometric.data.data.Data`

Constructs a :class:`~torch_geometric.data.Data` object from a feature
store and graph store instance.

### `filter_data(data: torch_geometric.data.data.Data, node: torch.Tensor, row: torch.Tensor, col: torch.Tensor, edge: Optional[torch.Tensor], perm: Optional[torch.Tensor] = None) -> torch_geometric.data.data.Data`

### `filter_hetero_data(data: torch_geometric.data.hetero_data.HeteroData, node_dict: Dict[str, torch.Tensor], row_dict: Dict[Tuple[str, str, str], torch.Tensor], col_dict: Dict[Tuple[str, str, str], torch.Tensor], edge_dict: Dict[Tuple[str, str, str], Optional[torch.Tensor]], perm_dict: Optional[Dict[Tuple[str, str, str], Optional[torch.Tensor]]] = None) -> torch_geometric.data.hetero_data.HeteroData`

### `get_input_nodes(data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData, Tuple[torch_geometric.data.feature_store.FeatureStore, torch_geometric.data.graph_store.GraphStore]], input_nodes: Union[torch.Tensor, NoneType, str, Tuple[str, Optional[torch.Tensor]], torch_geometric.data.feature_store.TensorAttr], input_id: Optional[torch.Tensor] = None) -> Tuple[Optional[str], torch.Tensor, Optional[torch.Tensor]]`

### `infer_filter_per_worker(data: Any) -> bool`

## Classes (15)

### `AffinityMixin`

A context manager to enable CPU affinity for data loader workers
(only used when running on CPU devices).

Affinitization places data loader workers threads on specific CPU cores.
In effect, it allows for more efficient local memory allocation and reduces
remote memory calls.
Every time a process or thread moves from one core to another, registers
and caches need to be flushed and reloaded.
This can become very costly if it happens often, and our threads may also
no longer be close to their data, or be able to share data in a cache.

See `here <https://pytorch-geometric.readthedocs.io/en/latest/advanced/
cpu_affinity.html>`__ for the accompanying tutorial.

.. warning::

    To correctly affinitize compute threads (*i.e.* with
    :obj:`KMP_AFFINITY`), please make sure that you exclude
    :obj:`loader_cores` from the list of cores available for the main
    process.
    This will cause core oversubsription and exacerbate performance.

.. code-block:: python

    loader = NeigborLoader(data, num_workers=3)
    with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
        for batch in loader:
            pass

#### Methods

- **`enable_cpu_affinity(self, loader_cores: Union[List[List[int]], List[int], NoneType] = None) -> None`**
  Enables CPU affinity.

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

### `DataLoaderIterator`

A data loader iterator extended by a simple post transformation
function :meth:`transform_fn`. While the iterator may request items from
different sub-processes, :meth:`transform_fn` will always be executed in
the main process.

This iterator is used in PyG's sampler classes, and is responsible for
feature fetching and filtering data objects after sampling has taken place
in a sub-process. This has the following advantages:

* We do not need to share feature matrices across processes which may
  prevent any errors due to too many open file handles.
* We can execute any expensive post-processing commands on the main thread
  with full parallelization power (which usually executes faster).
* It lets us naturally support data already being present on the GPU.

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

### `LogMemoryMixin`

A context manager to enable logging of memory consumption in
:class:`~torch.utils.data.DataLoader` workers.

#### Methods

- **`enable_memory_log(self) -> None`**

### `MultithreadingMixin`

A context manager to enable multi-threading in
:class:`~torch.utils.data.DataLoader` workers.
It changes the default value of threads used in the loader from :obj:`1`
to :obj:`worker_threads`.

#### Methods

- **`enable_multithreading(self, worker_threads: Optional[int] = None) -> None`**
  Enables multithreading in worker subprocesses.

### `NodeLoader`

A data loader that performs mini-batch sampling from node information,
using a generic :class:`~torch_geometric.sampler.BaseSampler`
implementation that defines a
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes` function and
is supported on the provided input :obj:`data` object.

Args:
    data (Any): A :class:`~torch_geometric.data.Data`,
        :class:`~torch_geometric.data.HeteroData`, or
        (:class:`~torch_geometric.data.FeatureStore`,
        :class:`~torch_geometric.data.GraphStore`) data object.
    node_sampler (torch_geometric.sampler.BaseSampler): The sampler
        implementation to be used with this loader.
        Needs to implement
        :meth:`~torch_geometric.sampler.BaseSampler.sample_from_nodes`.
        The sampler implementation must be compatible with the input
        :obj:`data` object.
    input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
        indices of seed nodes to start sampling from.
        Needs to be either given as a :obj:`torch.LongTensor` or
        :obj:`torch.BoolTensor`.
        If set to :obj:`None`, all nodes will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the node type and node indices. (default: :obj:`None`)
    input_time (torch.Tensor, optional): Optional values to override the
        timestamp for the input nodes given in :obj:`input_nodes`. If not
        set, will use the timestamps in :obj:`time_attr` as default (if
        present). The :obj:`time_attr` needs to be set for this to work.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    transform_sampler_output (callable, optional): A function/transform
        that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
        returns a transformed version. (default: :obj:`None`)
    filter_per_worker (bool, optional): If set to :obj:`True`, will filter
        the returned data in each worker's subprocess.
        If set to :obj:`False`, will filter the returned data in the main
        process.
        If set to :obj:`None`, will automatically infer the decision based
        on whether data partially lives on the GPU
        (:obj:`filter_per_worker=True`) or entirely on the CPU
        (:obj:`filter_per_worker=False`).
        There exists different trade-offs for setting this option.
        Specifically, setting this option to :obj:`True` for in-memory
        datasets will move all features to shared memory, which may result
        in too many open file handles. (default: :obj:`None`)
    custom_cls (HeteroData, optional): A custom
        :class:`~torch_geometric.data.HeteroData` class to return for
        mini-batches in case of remote backends. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

#### Methods

- **`collate_fn(self, index: Union[torch.Tensor, List[int]]) -> Any`**
  Samples a subgraph from a batch of input nodes.

- **`filter_fn(self, out: Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**
  Joins the sampled nodes with their corresponding features,

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
