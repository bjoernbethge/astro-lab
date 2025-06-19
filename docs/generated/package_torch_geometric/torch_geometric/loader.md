# loader

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.loader`

## Functions (1)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

## Classes (25)

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

### `CachedLoader`

A loader to cache mini-batch outputs, e.g., obtained during
:class:`NeighborLoader` iterations.

Args:
    loader (torch.utils.data.DataLoader): The data loader.
    device (torch.device, optional): The device to load the data to.
        (default: :obj:`None`)
    transform (callable, optional): A function/transform that takes in
        a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)

#### Methods

- **`clear(self)`**
  Clears the cache.

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

### `DynamicBatchSampler`

Dynamically adds samples to a mini-batch up to a maximum size (either
based on number of nodes or number of edges). When data samples have a
wide range in sizes, specifying a mini-batch size in terms of number of
samples is not ideal and can cause CUDA OOM errors.

Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
ambiguous, depending on the order of the samples. By default the
:meth:`__len__` will be undefined. This is fine for most cases but
progress bars will be infinite. Alternatively, :obj:`num_steps` can be
supplied to cap the number of mini-batches produced by the sampler.

.. code-block:: python

    from torch_geometric.loader import DataLoader, DynamicBatchSampler

    sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
    loader = DataLoader(dataset, batch_sampler=sampler, ...)

Args:
    dataset (Dataset): Dataset to sample from.
    max_num (int): Size of mini-batch to aim for in number of nodes or
        edges.
    mode (str, optional): :obj:`"node"` or :obj:`"edge"` to measure
        batch size. (default: :obj:`"node"`)
    shuffle (bool, optional): If set to :obj:`True`, will have the data
        reshuffled at every epoch. (default: :obj:`False`)
    skip_too_big (bool, optional): If set to :obj:`True`, skip samples
        which cannot fit in a batch by itself. (default: :obj:`False`)
    num_steps (int, optional): The number of mini-batches to draw for a
        single epoch. If set to :obj:`None`, will iterate through all the
        underlying examples, but :meth:`__len__` will be :obj:`None` since
        it is ambiguous. (default: :obj:`None`)

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

### `HGTLoader`

The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.

:class:`~torch_geometric.data.HGTLoader` tries to (1) keep a similar
number of nodes and edges for each type and (2) keep the sampled sub-graph
dense to minimize the information loss and reduce the sample variance.

Methodically, :class:`~torch_geometric.data.HGTLoader` keeps track of a
node budget for each node type, which is then used to determine the
sampling probability of a node.
In particular, the probability of sampling a node is determined by the
number of connections to already sampled nodes and their node degrees.
With this, :class:`~torch_geometric.data.HGTLoader` will sample a fixed
amount of neighbors for each node type in each iteration, as given by the
:obj:`num_samples` argument.

Sampled nodes are sorted based on the order in which they were sampled.
In particular, the first :obj:`batch_size` nodes represent the set of
original mini-batch nodes.

.. note::

    For an example of using :class:`~torch_geometric.data.HGTLoader`, see
    `examples/hetero/to_hetero_mag.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    hetero/to_hetero_mag.py>`_.

.. code-block:: python

    from torch_geometric.loader import HGTLoader
    from torch_geometric.datasets import OGB_MAG

    hetero_data = OGB_MAG(path)[0]

    loader = HGTLoader(
        hetero_data,
        # Sample 512 nodes per type and per iteration for 4 iterations
        num_samples={key: [512] * 4 for key in hetero_data.node_types},
        # Use a batch size of 128 for sampling training nodes of type paper
        batch_size=128,
        input_nodes=('paper', hetero_data['paper'].train_mask),
    )

    sampled_hetero_data = next(iter(loader))
    print(sampled_data.batch_size)
    >>> 128

Args:
    data (Any): A :class:`~torch_geometric.data.Data`,
        :class:`~torch_geometric.data.HeteroData`, or
        (:class:`~torch_geometric.data.FeatureStore`,
        :class:`~torch_geometric.data.GraphStore`) data object.
    num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
        sample in each iteration and for each node type.
        If given as a list, will sample the same amount of nodes for each
        node type.
    input_nodes (str or Tuple[str, torch.Tensor]): The indices of nodes for
        which neighbors are sampled to create mini-batches.
        Needs to be passed as a tuple that holds the node type and
        corresponding node indices.
        Node indices need to be either given as a :obj:`torch.LongTensor`
        or :obj:`torch.BoolTensor`.
        If node indices are set to :obj:`None`, all nodes of this specific
        type will be considered.
    transform (callable, optional): A function/transform that takes in
        an a sampled mini-batch and returns a transformed version.
        (default: :obj:`None`)
    transform_sampler_output (callable, optional): A function/transform
        that takes in a :class:`torch_geometric.sampler.SamplerOutput` and
        returns a transformed version. (default: :obj:`None`)
    is_sorted (bool, optional): If set to :obj:`True`, assumes that
        :obj:`edge_index` is sorted by column. This avoids internal
        re-sorting of the data and can improve runtime and memory
        efficiency. (default: :obj:`False`)
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

### `ImbalancedSampler`

A weighted random sampler that randomly samples elements according to
class distribution.
As such, it will either remove samples from the majority class
(under-sampling) or add more examples from the minority class
(over-sampling).

**Graph-level sampling:**

.. code-block:: python

    from torch_geometric.loader import DataLoader, ImbalancedSampler

    sampler = ImbalancedSampler(dataset)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)

**Node-level sampling:**

.. code-block:: python

    from torch_geometric.loader import NeighborLoader, ImbalancedSampler

    sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
    loader = NeighborLoader(data, input_nodes=data.train_mask,
                            batch_size=64, num_neighbors=[-1, -1],
                            sampler=sampler, ...)

You can also pass in the class labels directly as a :class:`torch.Tensor`:

.. code-block:: python

    from torch_geometric.loader import NeighborLoader, ImbalancedSampler

    sampler = ImbalancedSampler(data.y)
    loader = NeighborLoader(data, input_nodes=data.train_mask,
                            batch_size=64, num_neighbors=[-1, -1],
                            sampler=sampler, ...)

Args:
    dataset (Dataset or Data or Tensor): The dataset or class distribution
        from which to sample the data, given either as a
        :class:`~torch_geometric.data.Dataset`,
        :class:`~torch_geometric.data.Data`, or :class:`torch.Tensor`
        object.
    input_nodes (Tensor, optional): The indices of nodes that are used by
        the corresponding loader, *e.g.*, by
        :class:`~torch_geometric.loader.NeighborLoader`.
        If set to :obj:`None`, all nodes will be considered.
        This argument should only be set for node-level loaders and does
        not have any effect when operating on a set of graphs as given by
        :class:`~torch_geometric.data.Dataset`. (default: :obj:`None`)
    num_samples (int, optional): The number of samples to draw for a single
        epoch. If set to :obj:`None`, will sample as much elements as there
        exists in the underlying data. (default: :obj:`None`)

### `LinkLoader`

A data loader that performs mini-batch sampling from link information,
using a generic :class:`~torch_geometric.sampler.BaseSampler`
implementation that defines a
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges` function and
is supported on the provided input :obj:`data` object.

.. note::
    Negative sampling is currently implemented in an approximate
    way, *i.e.* negative edges may contain false negatives.

Args:
    data (Any): A :class:`~torch_geometric.data.Data`,
        :class:`~torch_geometric.data.HeteroData`, or
        (:class:`~torch_geometric.data.FeatureStore`,
        :class:`~torch_geometric.data.GraphStore`) data object.
    link_sampler (torch_geometric.sampler.BaseSampler): The sampler
        implementation to be used with this loader.
        Needs to implement
        :meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.
        The sampler implementation must be compatible with the input
        :obj:`data` object.
    edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The edge indices, holding source and destination nodes to start
        sampling from.
        If set to :obj:`None`, all edges will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the edge type and corresponding edge indices.
        (default: :obj:`None`)
    edge_label (Tensor, optional): The labels of edge indices from which to
        start sampling from. Must be the same length as
        the :obj:`edge_label_index`. (default: :obj:`None`)
    edge_label_time (Tensor, optional): The timestamps of edge indices from
        which to start sampling from. Must be the same length as
        :obj:`edge_label_index`. If set, temporal sampling will be
        used such that neighbors are guaranteed to fulfill temporal
        constraints, *i.e.*, neighbors have an earlier timestamp than
        the ouput edge. The :obj:`time_attr` needs to be set for this
        to work. (default: :obj:`None`)
    neg_sampling (NegativeSampling, optional): The negative sampling
        configuration.
        For negative sampling mode :obj:`"binary"`, samples can be accessed
        via the attributes :obj:`edge_label_index` and :obj:`edge_label` in
        the respective edge type of the returned mini-batch.
        In case :obj:`edge_label` does not exist, it will be automatically
        created and represents a binary classification task (:obj:`0` =
        negative edge, :obj:`1` = positive edge).
        In case :obj:`edge_label` does exist, it has to be a categorical
        label from :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges.
        Note that returned labels are of type :obj:`torch.float` for binary
        classification (to facilitate the ease-of-use of
        :meth:`F.binary_cross_entropy`) and of type
        :obj:`torch.long` for multi-class classification (to facilitate the
        ease-of-use of :meth:`F.cross_entropy`).
        For negative sampling mode :obj:`"triplet"`, samples can be
        accessed via the attributes :obj:`src_index`, :obj:`dst_pos_index`
        and :obj:`dst_neg_index` in the respective node types of the
        returned mini-batch.
        :obj:`edge_label` needs to be :obj:`None` for :obj:`"triplet"`
        negative sampling mode.
        If set to :obj:`None`, no negative sampling strategy is applied.
        (default: :obj:`None`)
    neg_sampling_ratio (int or float, optional): The ratio of sampled
        negative edges to the number of positive edges.
        Deprecated in favor of the :obj:`neg_sampling` argument.
        (default: :obj:`None`).
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
  Samples a subgraph from a batch of input edges.

- **`filter_fn(self, out: Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**
  Joins the sampled nodes with their corresponding features,

### `LinkNeighborLoader`

A link-based data loader derived as an extension of the node-based
:class:`torch_geometric.loader.NeighborLoader`.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.

More specifically, this loader first selects a sample of edges from the
set of input edges :obj:`edge_label_index` (which may or not be edges in
the original graph) and then constructs a subgraph from all the nodes
present in this list by sampling :obj:`num_neighbors` neighbors in each
iteration.

.. code-block:: python

    from torch_geometric.datasets import Planetoid
    from torch_geometric.loader import LinkNeighborLoader

    data = Planetoid(path, name='Cora')[0]

    loader = LinkNeighborLoader(
        data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=128,
        edge_label_index=data.edge_index,
    )

    sampled_data = next(iter(loader))
    print(sampled_data)
    >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
             train_mask=[1368], val_mask=[1368], test_mask=[1368],
             edge_label_index=[2, 128])

It is additionally possible to provide edge labels for sampled edges, which
are then added to the batch:

.. code-block:: python

    loader = LinkNeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=128,
        edge_label_index=data.edge_index,
        edge_label=torch.ones(data.edge_index.size(1))
    )

    sampled_data = next(iter(loader))
    print(sampled_data)
    >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
             train_mask=[1368], val_mask=[1368], test_mask=[1368],
             edge_label_index=[2, 128], edge_label=[128])

The rest of the functionality mirrors that of
:class:`~torch_geometric.loader.NeighborLoader`, including support for
heterogeneous graphs.
In particular, the data loader will add the following attributes to the
returned mini-batch:

* :obj:`n_id` The global node index for every sampled node
* :obj:`e_id` The global edge index for every sampled edge
* :obj:`input_id`: The global index of the :obj:`edge_label_index`
* :obj:`num_sampled_nodes`: The number of sampled nodes in each hop
* :obj:`num_sampled_edges`: The number of sampled edges in each hop

.. note::
    Negative sampling is currently implemented in an approximate
    way, *i.e.* negative edges may contain false negatives.

.. warning::
    Note that the sampling scheme is independent from the edge we are
    making a prediction for.
    That is, by default supervision edges in :obj:`edge_label_index`
    **will not** get masked out during sampling.
    In case there exists an overlap between message passing edges in
    :obj:`data.edge_index` and supervision edges in
    :obj:`edge_label_index`, you might end up sampling an edge you are
    making a prediction for.
    You can generally avoid this behavior (if desired) by making
    :obj:`data.edge_index` and :obj:`edge_label_index` two disjoint sets of
    edges, *e.g.*, via the
    :class:`~torch_geometric.transforms.RandomLinkSplit` transformation and
    its :obj:`disjoint_train_ratio` argument.

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
    edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
        The edge indices for which neighbors are sampled to create
        mini-batches.
        If set to :obj:`None`, all edges will be considered.
        In heterogeneous graphs, needs to be passed as a tuple that holds
        the edge type and corresponding edge indices.
        (default: :obj:`None`)
    edge_label (Tensor, optional): The labels of edge indices for
        which neighbors are sampled. Must be the same length as
        the :obj:`edge_label_index`. If set to :obj:`None` its set to
        `torch.zeros(...)` internally. (default: :obj:`None`)
    edge_label_time (Tensor, optional): The timestamps for edge indices
        for which neighbors are sampled. Must be the same length as
        :obj:`edge_label_index`. If set, temporal sampling will be
        used such that neighbors are guaranteed to fulfill temporal
        constraints, *i.e.*, neighbors have an earlier timestamp than
        the ouput edge. The :obj:`time_attr` needs to be set for this
        to work. (default: :obj:`None`)
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
    neg_sampling (NegativeSampling, optional): The negative sampling
        configuration.
        For negative sampling mode :obj:`"binary"`, samples can be accessed
        via the attributes :obj:`edge_label_index` and :obj:`edge_label` in
        the respective edge type of the returned mini-batch.
        In case :obj:`edge_label` does not exist, it will be automatically
        created and represents a binary classification task (:obj:`0` =
        negative edge, :obj:`1` = positive edge).
        In case :obj:`edge_label` does exist, it has to be a categorical
        label from :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges.
        Note that returned labels are of type :obj:`torch.float` for binary
        classification (to facilitate the ease-of-use of
        :meth:`F.binary_cross_entropy`) and of type
        :obj:`torch.long` for multi-class classification (to facilitate the
        ease-of-use of :meth:`F.cross_entropy`).
        For negative sampling mode :obj:`"triplet"`, samples can be
        accessed via the attributes :obj:`src_index`, :obj:`dst_pos_index`
        and :obj:`dst_neg_index` in the respective node types of the
        returned mini-batch.
        :obj:`edge_label` needs to be :obj:`None` for :obj:`"triplet"`
        negative sampling mode.
        If set to :obj:`None`, no negative sampling strategy is applied.
        (default: :obj:`None`)
    neg_sampling_ratio (int or float, optional): The ratio of sampled
        negative edges to the number of positive edges.
        Deprecated in favor of the :obj:`neg_sampling` argument.
        (default: :obj:`None`)
    time_attr (str, optional): The name of the attribute that denotes
        timestamps for either the nodes or edges in the graph.
        If set, temporal sampling will be used such that neighbors are
        guaranteed to fulfill temporal constraints, *i.e.* neighbors have
        an earlier or equal timestamp than the center node.
        Only used if :obj:`edge_label_time` is set. (default: :obj:`None`)
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

#### Methods

- **`sample(self, batch)`**

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

### `PrefetchLoader`

A GPU prefetcher class for asynchronously transferring data of a
:class:`torch.utils.data.DataLoader` from host memory to device memory.

Args:
    loader (torch.utils.data.DataLoader): The data loader.
    device (torch.device, optional): The device to load the data to.
        (default: :obj:`None`)

#### Methods

- **`non_blocking_transfer(self, batch: Any) -> Any`**

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

### `TemporalDataLoader`

A data loader which merges succesive events of a
:class:`torch_geometric.data.TemporalData` to a mini-batch.

Args:
    data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
        from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    neg_sampling_ratio (float, optional): The ratio of sampled negative
        destination nodes to the number of postive destination nodes.
        (default: :obj:`0.0`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`.

### `ZipLoader`

A loader that returns a tuple of data objects by sampling from multiple
:class:`NodeLoader` or :class:`LinkLoader` instances.

Args:
    loaders (List[NodeLoader] or List[LinkLoader]): The loader instances.
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

#### Methods

- **`collate_fn(self, index: List[int]) -> Tuple[Any, ...]`**

- **`filter_fn(self, outs: Tuple[Any, ...]) -> Tuple[Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData], ...]`**
