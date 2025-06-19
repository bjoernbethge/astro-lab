# loader

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.loader`

## Functions (15)

### `create_dataset()`

Create dataset object.

Returns: PyG dataset object

### `create_link_label(pos_edge_index, neg_edge_index)`

Create labels for link prediction, based on positive and negative edges.

Args:
    pos_edge_index (torch.tensor): Positive edge index [2, num_edges]
    neg_edge_index (torch.tensor): Negative edge index [2, num_edges]

Returns: Link label tensor, [num_positive_edges + num_negative_edges]

### `create_loader()`

Create data loader object.

Returns: List of PyTorch data loaders

### `get_loader(dataset, sampler, batch_size, shuffle=True)`

### `index2mask(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

Converts indices to a mask representation.

Args:
    index (Tensor): The indices.
    size (int, optional): The size of the mask. If set to :obj:`None`, a
        minimal sized output mask is returned.

Example:
    >>> index = torch.tensor([1, 3, 5])
    >>> index_to_mask(index)
    tensor([False,  True, False,  True, False,  True])

    >>> index_to_mask(index, size=7)
    tensor([False,  True, False,  True, False,  True, False])

### `index_to_mask(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

Converts indices to a mask representation.

Args:
    index (Tensor): The indices.
    size (int, optional): The size of the mask. If set to :obj:`None`, a
        minimal sized output mask is returned.

Example:
    >>> index = torch.tensor([1, 3, 5])
    >>> index_to_mask(index)
    tensor([False,  True, False,  True, False,  True])

    >>> index_to_mask(index, size=7)
    tensor([False,  True, False,  True, False,  True, False])

### `load_dataset()`

Load dataset objects.

Returns: PyG dataset object

### `load_ogb(name, dataset_dir)`

Load OGB dataset objects.

Args:
    name (str): dataset name
    dataset_dir (str): data directory

Returns: PyG dataset object

### `load_pyg(name, dataset_dir)`

Load PyG dataset objects. (More PyG datasets will be supported).

Args:
    name (str): dataset name
    dataset_dir (str): data directory

Returns: PyG dataset object

### `neg_sampling_transform(data)`

Do negative sampling for link prediction tasks.

Args:
    data (torch_geometric.data): Input data object

Returns: Transformed data object with negative edges + link pred labels

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

### `planetoid_dataset(name: str) -> Callable`

### `set_dataset_attr(dataset, name, value, size)`

### `set_dataset_info(dataset)`

Set global dataset information.

Args:
    dataset: PyG dataset object

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

## Classes (15)

### `Amazon`

The Amazon Computers and Amazon Photo networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent goods and edges represent that two goods are frequently
bought together.
Given product reviews as bag-of-words node features, the task is to
map goods to their respective product category.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Computers"`,
        :obj:`"Photo"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - Computers
      - 13,752
      - 491,722
      - 767
      - 10
    * - Photo
      - 7,650
      - 238,162
      - 745
      - 8

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `Coauthor`

The Coauthor CS and Coauthor Physics networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent authors that are connected by an edge if they co-authored a
paper.
Given paper keywords for each author's papers, the task is to map authors
to their respective field of study.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"CS"`, :obj:`"Physics"`).
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - CS
      - 18,333
      - 163,788
      - 6,805
      - 15
    * - Physics
      - 34,493
      - 495,924
      - 8,415
      - 5

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `KarateClub`

Zachary's karate club network from the `"An Information Flow Model for
Conflict and Fission in Small Groups"
<https://www.journals.uchicago.edu/doi/abs/10.1086/jar.33.4.3629752>`_
paper, containing 34 nodes,
connected by 156 (undirected and unweighted) edges.
Every node is labeled by one of four classes obtained via modularity-based
clustering, following the `"Semi-supervised Classification with Graph
Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_ paper.
Training is based on a single labeled example per class, *i.e.* a total
number of 4 labeled nodes.

Args:
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10
    :header-rows: 1

    * - #nodes
      - #edges
      - #features
      - #classes
    * - 34
      - 156
      - 34
      - 4

### `MNISTSuperpixels`

MNIST superpixels dataset from the `"Geometric Deep Learning on
Graphs and Manifolds Using Mixture Model CNNs"
<https://arxiv.org/abs/1611.08402>`_ paper, containing 70,000 graphs with
75 nodes each.
Every graph is labeled by one of 10 classes.

Args:
    root (str): Root directory where the dataset should be saved.
    train (bool, optional): If :obj:`True`, loads the training dataset,
        otherwise the test dataset. (default: :obj:`True`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - 70,000
      - 75
      - ~1,393.0
      - 1
      - 10

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `PPI`

The protein-protein interaction networks from the `"Predicting
Multicellular Function through Multi-layer Tissue Networks"
<https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
sets, motif gene sets and immunological signatures as features (50 in
total) and gene ontology sets as labels (121 in total).

Args:
    root (str): Root directory where the dataset should be saved.
    split (str, optional): If :obj:`"train"`, loads the training dataset.
        If :obj:`"val"`, loads the validation dataset.
        If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 20
      - ~2,245.3
      - ~61,318.4
      - 50
      - 121

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `Planetoid`

The citation network datasets :obj:`"Cora"`, :obj:`"CiteSeer"` and
:obj:`"PubMed"` from the `"Revisiting Semi-Supervised Learning with Graph
Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
Nodes represent documents and edges represent citation links.
Training, validation and test splits are given by binary masks.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The name of the dataset (:obj:`"Cora"`, :obj:`"CiteSeer"`,
        :obj:`"PubMed"`).
    split (str, optional): The type of dataset split (:obj:`"public"`,
        :obj:`"full"`, :obj:`"geom-gcn"`, :obj:`"random"`).
        If set to :obj:`"public"`, the split will be the public fixed split
        from the `"Revisiting Semi-Supervised Learning with Graph
        Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
        If set to :obj:`"full"`, all nodes except those in the validation
        and test sets will be used for training (as in the
        `"FastGCN: Fast Learning with Graph Convolutional Networks via
        Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
        If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
        `"Geom-GCN: Geometric Graph Convolutional Networks"
        <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
        If set to :obj:`"random"`, train, validation, and test sets will be
        randomly generated, according to :obj:`num_train_per_class`,
        :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
    num_train_per_class (int, optional): The number of training samples
        per class in case of :obj:`"random"` split. (default: :obj:`20`)
    num_val (int, optional): The number of validation samples in case of
        :obj:`"random"` split. (default: :obj:`500`)
    num_test (int, optional): The number of test samples in case of
        :obj:`"random"` split. (default: :obj:`1000`)
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #nodes
      - #edges
      - #features
      - #classes
    * - Cora
      - 2,708
      - 10,556
      - 1,433
      - 7
    * - CiteSeer
      - 3,327
      - 9,104
      - 3,703
      - 6
    * - PubMed
      - 19,717
      - 88,648
      - 500
      - 3

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

### `QM7b`

The QM7b dataset from the `"MoleculeNet: A Benchmark for Molecular
Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
7,211 molecules with 14 regression targets.

Args:
    root (str): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 10 10 10 10 10
    :header-rows: 1

    * - #graphs
      - #nodes
      - #edges
      - #features
      - #tasks
    * - 7,211
      - ~15.4
      - ~245.0
      - 0
      - 14

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.

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

### `TUDataset`

A variety of graph kernel benchmark datasets, *.e.g.*,
:obj:`"IMDB-BINARY"`, :obj:`"REDDIT-BINARY"` or :obj:`"PROTEINS"`,
collected from the `TU Dortmund University
<https://chrsmrrs.github.io/datasets>`_.
In addition, this dataset wrapper provides `cleaned dataset versions
<https://github.com/nd7141/graph_datasets>`_ as motivated by the
`"Understanding Isomorphism Bias in Graph Data Sets"
<https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
graphs.

.. note::
    Some datasets may not come with any node labels.
    You can then either make use of the argument :obj:`use_node_attr`
    to load additional continuous node attributes (if present) or provide
    synthetic node features using transforms such as
    :class:`torch_geometric.transforms.Constant` or
    :class:`torch_geometric.transforms.OneHotDegree`.

Args:
    root (str): Root directory where the dataset should be saved.
    name (str): The `name
        <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
        dataset.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
    force_reload (bool, optional): Whether to re-process the dataset.
        (default: :obj:`False`)
    use_node_attr (bool, optional): If :obj:`True`, the dataset will
        contain additional continuous node attributes (if present).
        (default: :obj:`False`)
    use_edge_attr (bool, optional): If :obj:`True`, the dataset will
        contain additional continuous edge attributes (if present).
        (default: :obj:`False`)
    cleaned (bool, optional): If :obj:`True`, the dataset will
        contain only non-isomorphic graphs. (default: :obj:`False`)

**STATS:**

.. list-table::
    :widths: 20 10 10 10 10 10
    :header-rows: 1

    * - Name
      - #graphs
      - #nodes
      - #edges
      - #features
      - #classes
    * - MUTAG
      - 188
      - ~17.9
      - ~39.6
      - 7
      - 2
    * - ENZYMES
      - 600
      - ~32.6
      - ~124.3
      - 3
      - 6
    * - PROTEINS
      - 1,113
      - ~39.1
      - ~145.6
      - 3
      - 2
    * - COLLAB
      - 5,000
      - ~74.5
      - ~4914.4
      - 0
      - 3
    * - IMDB-BINARY
      - 1,000
      - ~19.8
      - ~193.1
      - 0
      - 2
    * - REDDIT-BINARY
      - 2,000
      - ~429.6
      - ~995.5
      - 0
      - 2
    * - ...
      -
      -
      -
      -
      -

#### Methods

- **`download(self) -> None`**
  Downloads the dataset to the :obj:`self.raw_dir` folder.

- **`process(self) -> None`**
  Processes the dataset to the :obj:`self.processed_dir` folder.
