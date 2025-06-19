# hgt_loader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.hgt_loader`

## Classes (8)

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

### `HGTSampler`

An implementation of an in-memory heterogeneous layer-wise sampler
user by :class:`~torch_geometric.loader.HGTLoader`.

#### Methods

- **`sample_from_nodes(self, inputs: torch_geometric.sampler.base.NodeSamplerInput) -> torch_geometric.sampler.base.HeteroSamplerOutput`**
  Performs sampling from the nodes specified in :obj:`index`,

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
