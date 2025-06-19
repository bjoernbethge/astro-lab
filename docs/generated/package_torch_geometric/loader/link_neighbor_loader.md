# link_neighbor_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.link_neighbor_loader`

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

### `NegativeSampling`

The negative sampling configuration of a
:class:`~torch_geometric.sampler.BaseSampler` when calling
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

Args:
    mode (str): The negative sampling mode
        (:obj:`"binary"` or :obj:`"triplet"`).
        If set to :obj:`"binary"`, will randomly sample negative links
        from the graph.
        If set to :obj:`"triplet"`, will randomly sample negative
        destination nodes for each positive source node.
    amount (int or float, optional): The ratio of sampled negative edges to
        the number of positive edges. (default: :obj:`1`)
    src_weight (torch.Tensor, optional): A node-level vector determining
        the sampling of source nodes. Does not necessarily need to sum up
        to one. If not given, negative nodes will be sampled uniformly.
        (default: :obj:`None`)
    dst_weight (torch.Tensor, optional): A node-level vector determining
        the sampling of destination nodes. Does not necessarily need to sum
        up to one. If not given, negative nodes will be sampled uniformly.
        (default: :obj:`None`)

#### Methods

- **`is_binary(self) -> bool`**

- **`is_triplet(self) -> bool`**

- **`sample(self, num_samples: int, endpoint: Literal['src', 'dst'], num_nodes: Optional[int] = None) -> torch.Tensor`**
  Generates :obj:`num_samples` negative samples.

### `NeighborSampler`

An implementation of an in-memory (heterogeneous) neighbor sampler used
by :class:`~torch_geometric.loader.NeighborLoader`.

#### Methods

- **`sample_from_nodes(self, inputs: torch_geometric.sampler.base.NodeSamplerInput) -> Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]`**
  Performs sampling from the nodes specified in :obj:`index`,

- **`sample_from_edges(self, inputs: torch_geometric.sampler.base.EdgeSamplerInput, neg_sampling: Optional[torch_geometric.sampler.base.NegativeSampling] = None) -> Union[torch_geometric.sampler.base.SamplerOutput, torch_geometric.sampler.base.HeteroSamplerOutput]`**
  Performs sampling from the edges specified in :obj:`index`,

### `SubgraphType`

The type of the returned subgraph.
