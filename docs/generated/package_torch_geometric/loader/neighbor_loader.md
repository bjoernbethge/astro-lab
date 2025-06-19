# neighbor_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.neighbor_loader`

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

### `NeighborLoader`

A data loader that performs neighbor sampling as introduced in the
`"Inductive Representation Learning on Large Graphs"
<https://arxiv.org/abs/1706.02216>`_ paper.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.

More specifically, :obj:`num_neighbors` denotes how much neighbors are
sampled for each node in each iteration.
:class:`~torch_geometric.loader.NeighborLoader` takes in this list of
:obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
each node involved in iteration :obj:`i - 1`.

Sampled nodes are sorted based on the order in which they were sampled.
In particular, the first :obj:`batch_size` nodes represent the set of
original mini-batch nodes.

.. code-block:: python

    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import NeighborLoader

    data = Planetoid(path, name='Cora')[0]

    loader = NeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        input_nodes=data.train_mask,
    )

    sampled_data = next(iter(loader))
    print(sampled_data.batch_size)
    >>> 128

By default, the data loader will only include the edges that were
originally sampled (:obj:`directed = True`).
This option should only be used in case the number of hops is equivalent to
the number of GNN layers.
In case the number of GNN layers is greater than the number of hops,
consider setting :obj:`directed = False`, which will include all edges
between all sampled nodes (but is slightly slower as a result).

Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
**homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
well as **heterogeneous** graphs stored via
:class:`~torch_geometric.data.HeteroData`.
When operating in heterogeneous graphs, up to :obj:`num_neighbors`
neighbors will be sampled for each :obj:`edge_type`.
However, more fine-grained control over
the amount of sampled neighbors of individual edge types is possible:

.. code-block:: python

    from torch_geometric.datasets import OGB_MAG
    from torch_geometric.loader import NeighborLoader

    hetero_data = OGB_MAG(path)[0]

    loader = NeighborLoader(
        hetero_data,
        # Sample 30 neighbors for each node and edge type for 2 iterations
        num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=128,
        input_nodes=('paper', hetero_data['paper'].train_mask),
    )

    sampled_hetero_data = next(iter(loader))
    print(sampled_hetero_data['paper'].batch_size)
    >>> 128

.. note::

    For an example of using
    :class:`~torch_geometric.loader.NeighborLoader`, see
    `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
where global node indices are mapped to local indices corresponding to this
specific subgraph. However, often times it is desired to map the nodes of
the current subgraph back to the global node indices. The
:class:`~torch_geometric.loader.NeighborLoader` will include this mapping
as part of the :obj:`data` object:

.. code-block:: python

    loader = NeighborLoader(data, ...)
    sampled_data = next(iter(loader))
    print(sampled_data.n_id)  # Global node index of each node in batch.

In particular, the data loader will add the following attributes to the
returned mini-batch:

* :obj:`batch_size` The number of seed nodes (first nodes in the batch)
* :obj:`n_id` The global node index for every sampled node
* :obj:`e_id` The global edge index for every sampled edge
* :obj:`input_id`: The global index of the :obj:`input_nodes`
* :obj:`num_sampled_nodes`: The number of sampled nodes in each hop
* :obj:`num_sampled_edges`: The number of sampled edges in each hop

Args:
    data (Any): A :class:`~torch_geometric.data.Data`,
        :class:`~torch_geometric.data.HeteroData`, or
        (:class:`~torch_geometric.data.FeatureStore`,
        :class:`~torch_geometric.data.GraphStore`) data object.
    num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
        number of neighbors to sample for each node in each iteration.
        If an entry is set to :obj:`-1`, all neighbors will be included.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for each individual edge type.
    input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
        indices of nodes for which neighbors are sampled to create
        mini-batches.
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
    replace (bool, optional): If set to :obj:`True`, will sample with
        replacement. (default: :obj:`False`)
    subgraph_type (SubgraphType or str, optional): The type of the returned
        subgraph.
        If set to :obj:`"directional"`, the returned subgraph only holds
        the sampled (directed) edges which are necessary to compute
        representations for the sampled seed nodes.
        If set to :obj:`"bidirectional"`, sampled edges are converted to
        bidirectional edges.
        If set to :obj:`"induced"`, the returned subgraph contains the
        induced subgraph of all sampled nodes.
        (default: :obj:`"directional"`)
    disjoint (bool, optional): If set to :obj: `True`, each seed node will
        create its own disjoint subgraph.
        If set to :obj:`True`, mini-batch outputs will have a :obj:`batch`
        vector holding the mapping of nodes to their respective subgraph.
        Will get automatically set to :obj:`True` in case of temporal
        sampling. (default: :obj:`False`)
    temporal_strategy (str, optional): The sampling strategy when using
        temporal sampling (:obj:`"uniform"`, :obj:`"last"`).
        If set to :obj:`"uniform"`, will sample uniformly across neighbors
        that fulfill temporal constraints.
        If set to :obj:`"last"`, will sample the last `num_neighbors` that
        fulfill temporal constraints.
        (default: :obj:`"uniform"`)
    time_attr (str, optional): The name of the attribute that denotes
        timestamps for either the nodes or edges in the graph.
        If set, temporal sampling will be used such that neighbors are
        guaranteed to fulfill temporal constraints, *i.e.* neighbors have
        an earlier or equal timestamp than the center node.
        (default: :obj:`None`)
    weight_attr (str, optional): The name of the attribute that denotes
        edge weights in the graph.
        If set, weighted/biased sampling will be used such that neighbors
        are more likely to get sampled the higher their edge weights are.
        Edge weights do not need to sum to one, but must be non-negative,
        finite and have a non-zero sum within local neighborhoods.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    transform_sampler_output (callable, optional): A function/transform
        that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
        returns a transformed version. (default: :obj:`None`)
    is_sorted (bool, optional): If set to :obj:`True`, assumes that
        :obj:`edge_index` is sorted by column.
        If :obj:`time_attr` is set, additionally requires that rows are
        sorted according to time within individual neighborhoods.
        This avoids internal re-sorting of the data and can improve
        runtime and memory efficiency. (default: :obj:`False`)
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
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
        :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.

### `NeighborSampler`

An implementation of an in-memory (heterogeneous) neighbor sampler used
by :class:`~torch_geometric.loader.NeighborLoader`.

#### Methods

- **`sample_from_nodes(self, inputs: torch_geometric.sampler.base.NodeSamplerInput) -> Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]`**
  Performs sampling from the nodes specified in :obj:`index`,

- **`sample_from_edges(self, inputs: torch_geometric.sampler.base.EdgeSamplerInput, neg_sampling: Optional[torch_geometric.sampler.base.NegativeSampling] = None) -> Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]`**
  Performs sampling from the edges specified in :obj:`index`,

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

### `SubgraphType`

The type of the returned subgraph.
