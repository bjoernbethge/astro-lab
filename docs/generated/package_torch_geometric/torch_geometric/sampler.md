# sampler

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.sampler`

## Description

Graph sampler package.

## Classes (9)

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

### `EdgeSamplerInput`

The sampling input of
:meth:`~torch_geometric.sampler.BaseSampler.sample_from_edges`.

Args:
    input_id (torch.Tensor, optional): The indices of the data loader input
        of the current mini-batch.
    row (torch.Tensor): The source node indices of seed links to start
        sampling from.
    col (torch.Tensor): The destination node indices of seed links to start
        sampling from.
    label (torch.Tensor, optional): The label for the seed links.
        (default: :obj:`None`)
    time (torch.Tensor, optional): The timestamp for the seed links.
        (default: :obj:`None`)
    input_type (Tuple[str, str, str], optional): The input edge type (in
        case of sampling in a heterogeneous graph). (default: :obj:`None`)

### `HGTSampler`

An implementation of an in-memory heterogeneous layer-wise sampler
user by :class:`~torch_geometric.loader.HGTLoader`.

#### Methods

- **`sample_from_nodes(self, inputs: torch_geometric.sampler.base.NodeSamplerInput) -> torch_geometric.sampler.base.HeteroSamplerOutput`**
  Performs sampling from the nodes specified in :obj:`index`,

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

### `NumNeighbors`

The number of neighbors to sample in a homogeneous or heterogeneous
graph. In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for individual edge types.

Args:
    values (List[int] or Dict[Tuple[str, str, str], List[int]]): The
        number of neighbors to sample.
        If an entry is set to :obj:`-1`, all neighbors will be included.
        In heterogeneous graphs, may also take in a dictionary denoting
        the amount of neighbors to sample for individual edge types.
    default (List[int], optional): The default number of neighbors for edge
        types not specified in :obj:`values`. (default: :obj:`None`)

#### Methods

- **`get_values(self, edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[Tuple[str, str, str], List[int]]]`**
  Returns the number of neighbors.

- **`get_mapped_values(self, edge_types: Optional[List[Tuple[str, str, str]]] = None) -> Union[List[int], Dict[str, List[int]]]`**
  Returns the number of neighbors.

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
