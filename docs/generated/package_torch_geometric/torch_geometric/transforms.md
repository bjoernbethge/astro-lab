# transforms

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.transforms`

## Functions (1)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

## Classes (66)

### `AddLaplacianEigenvectorPE`

Adds the Laplacian eigenvector positional encoding from the
`"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
paper to the given graph
(functional name: :obj:`add_laplacian_eigenvector_pe`).

Args:
    k (int): The number of non-trivial eigenvectors to consider.
    attr_name (str, optional): The attribute name of the data object to add
        positional encodings to. If set to :obj:`None`, will be
        concatenated to :obj:`data.x`.
        (default: :obj:`"laplacian_eigenvector_pe"`)
    is_undirected (bool, optional): If set to :obj:`True`, this transform
        expects undirected graphs as input, and can hence speed up the
        computation of eigenvectors. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
        :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
        :attr:`is_undirected` is :obj:`True`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `AddMetaPaths`

Adds additional edge types to a
:class:`~torch_geometric.data.HeteroData` object between the source node
type and the destination node type of a given :obj:`metapath`, as described
in the `"Heterogenous Graph Attention Networks"
<https://arxiv.org/abs/1903.07293>`_ paper
(functional name: :obj:`add_metapaths`).

Meta-path based neighbors can exploit different aspects of structure
information in heterogeneous graphs.
Formally, a metapath is a path of the form

.. math::

    \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
    \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
represents the edge type connecting two node types.
The added edge type is given by the sequential multiplication  of
adjacency matrices along the metapath, and is added to the
:class:`~torch_geometric.data.HeteroData` object as edge type
:obj:`(src_node_type, "metapath_*", dst_node_type)`, where
:obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
and :math:`\mathcal{V}_{\ell}`, respectively.

In addition, a :obj:`metapath_dict` object is added to the
:class:`~torch_geometric.data.HeteroData` object which maps the
metapath-based edge type to its original metapath.

.. code-block:: python

    from torch_geometric.datasets import DBLP
    from torch_geometric.data import HeteroData
    from torch_geometric.transforms import AddMetaPaths

    data = DBLP(root)[0]
    # 4 node types: "paper", "author", "conference", and "term"
    # 6 edge types: ("paper","author"), ("author", "paper"),
    #               ("paper, "term"), ("paper", "conference"),
    #               ("term, "paper"), ("conference", "paper")

    # Add two metapaths:
    # 1. From "paper" to "paper" through "conference"
    # 2. From "author" to "conference" through "paper"
    metapaths = [[("paper", "conference"), ("conference", "paper")],
                 [("author", "paper"), ("paper", "conference")]]
    data = AddMetaPaths(metapaths)(data)

    print(data.edge_types)
    >>> [("author", "to", "paper"), ("paper", "to", "author"),
         ("paper", "to", "term"), ("paper", "to", "conference"),
         ("term", "to", "paper"), ("conference", "to", "paper"),
         ("paper", "metapath_0", "paper"),
         ("author", "metapath_1", "conference")]

    print(data.metapath_dict)
    >>> {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                            ("conference", "paper")],
         ("author", "metapath_1", "conference"): [("author", "paper"),
                                                  ("paper", "conference")]}

Args:
    metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
        by a list of lists of
        :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
        edge types will be dropped. (default: :obj:`False`)
    keep_same_node_type (bool, optional): If set to :obj:`True`, existing
        edge types between the same node type are not dropped even in case
        :obj:`drop_orig_edge_types` is set to :obj:`True`.
        (default: :obj:`False`)
    drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
        will drop node types not connected by any edge type.
        (default: :obj:`False`)
    max_sample (int, optional): If set, will sample at maximum
        :obj:`max_sample` neighbors within metapaths. Useful in order to
        tackle very dense metapath edges. (default: :obj:`None`)
    weighted (bool, optional): If set to :obj:`True`, computes weights for
        each metapath edge and stores them in :obj:`edge_weight`. The
        weight of each metapath edge is computed as the number of metapaths
        from the start to the end of the metapath edge.
        (default :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.hetero_data.HeteroData) -> torch_geometric.data.hetero_data.HeteroData`**

### `AddRandomMetaPaths`

Adds additional edge types similar to :class:`AddMetaPaths`.
The key difference is that the added edge type is given by
multiple random walks along the metapath.
One might want to increase the number of random walks
via :obj:`walks_per_node` to achieve competitive performance with
:class:`AddMetaPaths`.

Args:
    metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
        by a list of lists of
        :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
        edge types will be dropped. (default: :obj:`False`)
    keep_same_node_type (bool, optional): If set to :obj:`True`, existing
        edge types between the same node type are not dropped even in case
        :obj:`drop_orig_edge_types` is set to :obj:`True`.
        (default: :obj:`False`)
    drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
        will drop node types not connected by any edge type.
        (default: :obj:`False`)
    walks_per_node (int, List[int], optional): The number of random walks
        for each starting node in a metapath. (default: :obj:`1`)
    sample_ratio (float, optional): The ratio of source nodes to start
        random walks from. (default: :obj:`1.0`)

#### Methods

- **`forward(self, data: torch_geometric.data.hetero_data.HeteroData) -> torch_geometric.data.hetero_data.HeteroData`**

- **`sample(edge_index: torch_geometric.edge_index.EdgeIndex, subset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**
  Sample neighbors from :obj:`edge_index` for each node in

### `AddRandomWalkPE`

Adds the random walk positional encoding from the `"Graph Neural
Networks with Learnable Structural and Positional Representations"
<https://arxiv.org/abs/2110.07875>`_ paper to the given graph
(functional name: :obj:`add_random_walk_pe`).

Args:
    walk_length (int): The number of random walk steps.
    attr_name (str, optional): The attribute name of the data object to add
        positional encodings to. If set to :obj:`None`, will be
        concatenated to :obj:`data.x`.
        (default: :obj:`"random_walk_pe"`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `AddRemainingSelfLoops`

Adds remaining self-loops to the given homogeneous or heterogeneous
graph (functional name: :obj:`add_remaining_self_loops`).

Args:
    attr (str, optional): The name of the attribute of edge weights
        or multi-dimensional edge features to pass to
        :meth:`torch_geometric.utils.add_remaining_self_loops`.
        (default: :obj:`"edge_weight"`)
    fill_value (float or Tensor or str, optional): The way to generate
        edge features of self-loops (in case :obj:`attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `AddSelfLoops`

Adds self-loops to the given homogeneous or heterogeneous graph
(functional name: :obj:`add_self_loops`).

Args:
    attr (str, optional): The name of the attribute of edge weights
        or multi-dimensional edge features to pass to
        :meth:`torch_geometric.utils.add_self_loops`.
        (default: :obj:`"edge_weight"`)
    fill_value (float or Tensor or str, optional): The way to generate
        edge features of self-loops (in case :obj:`attr != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `BaseTransform`

An abstract base class for writing transforms.

Transforms are a general way to modify and customize
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects, either by implicitly
passing them as an argument to a :class:`~torch_geometric.data.Dataset`, or
by applying them explicitly to individual
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` objects:

.. code-block:: python

    import torch_geometric.transforms as T
    from torch_geometric.datasets import TUDataset

    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    dataset = TUDataset(path, name='MUTAG', transform=transform)
    data = dataset[0]  # Implicitly transform data on every access.

    data = TUDataset(path, name='MUTAG')[0]
    data = transform(data)  # Explicitly transform data.

#### Methods

- **`forward(self, data: Any) -> Any`**

### `Cartesian`

Saves the relative Cartesian coordinates of linked nodes in its edge
attributes (functional name: :obj:`cartesian`). Each coordinate gets
globally normalized to a specified interval (:math:`[0, 1]` by default).

Args:
    norm (bool, optional): If set to :obj:`False`, the output will not be
        normalized. (default: :obj:`True`)
    max_value (float, optional): If set and :obj:`norm=True`, normalization
        will be performed based on this value instead of the maximum value
        found in the data. (default: :obj:`None`)
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)
    interval ((float, float), optional): A tuple specifying the lower and
        upper bound for normalization. (default: :obj:`(0.0, 1.0)`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Center`

Centers node positions :obj:`data.pos` around the origin
(functional name: :obj:`center`).

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `Compose`

Composes several transforms together.

Args:
    transforms (List[Callable]): List of transforms to compose.

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `ComposeFilters`

Composes several filters together.

Args:
    filters (List[Callable]): List of filters to compose.

### `Constant`

Appends a constant value to each node feature :obj:`x`
(functional name: :obj:`constant`).

Args:
    value (float, optional): The value to add. (default: :obj:`1.0`)
    cat (bool, optional): If set to :obj:`False`, existing node features
        will be replaced. (default: :obj:`True`)
    node_types (str or List[str], optional): The specified node type(s) to
        append constant values for if used on heterogeneous graphs.
        If set to :obj:`None`, constants will be added to each node feature
        :obj:`x` for all existing node types. (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `Delaunay`

Computes the delaunay triangulation of a set of points
(functional name: :obj:`delaunay`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Distance`

Saves the Euclidean distance of linked nodes in its edge attributes
(functional name: :obj:`distance`). Each distance gets globally normalized
to a specified interval (:math:`[0, 1]` by default).

Args:
    norm (bool, optional): If set to :obj:`False`, the output will not be
        normalized. (default: :obj:`True`)
    max_value (float, optional): If set and :obj:`norm=True`, normalization
        will be performed based on this value instead of the maximum value
        found in the data. (default: :obj:`None`)
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)
    interval ((float, float), optional): A tuple specifying the lower and
        upper bound for normalization. (default: :obj:`(0.0, 1.0)`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `FaceToEdge`

Converts mesh faces :obj:`[3, num_faces]` to edge indices
:obj:`[2, num_edges]` (functional name: :obj:`face_to_edge`).

Args:
    remove_faces (bool, optional): If set to :obj:`False`, the face tensor
        will not be removed.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `FeaturePropagation`

The feature propagation operator from the `"On the Unreasonable
Effectiveness of Feature propagation in Learning on Graphs with Missing
Node Features" <https://arxiv.org/abs/2111.12128>`_ paper
(functional name: :obj:`feature_propagation`).

.. math::
    \mathbf{X}^{(0)} &= (1 - \mathbf{M}) \cdot \mathbf{X}

    \mathbf{X}^{(\ell + 1)} &= \mathbf{X}^{(0)} + \mathbf{M} \cdot
    (\mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \mathbf{X}^{(\ell)})

where missing node features are inferred by known features via propagation.

.. code-block:: python

    from torch_geometric.transforms import FeaturePropagation

    transform = FeaturePropagation(missing_mask=torch.isnan(data.x))
    data = transform(data)

Args:
    missing_mask (torch.Tensor): Mask matrix
        :math:`\mathbf{M} \in {\{ 0, 1 \}}^{N\times F}` indicating missing
        node features.
    num_iterations (int, optional): The number of propagations.
        (default: :obj:`40`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `FixedPoints`

Samples a fixed number of points and features from a point cloud
(functional name: :obj:`fixed_points`).

Args:
    num (int): The number of points to sample.
    replace (bool, optional): If set to :obj:`False`, samples points
        without replacement. (default: :obj:`True`)
    allow_duplicates (bool, optional): In case :obj:`replace` is
        :obj`False` and :obj:`num` is greater than the number of points,
        this option determines whether to add duplicated nodes to the
        output points or not.
        In case :obj:`allow_duplicates` is :obj:`False`, the number of
        output points might be smaller than :obj:`num`.
        In case :obj:`allow_duplicates` is :obj:`True`, the number of
        duplicated points are kept to a minimum. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `GCNNorm`

Applies the GCN normalization from the `"Semi-supervised Classification
with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
paper (functional name: :obj:`gcn_norm`).

.. math::
    \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
    \mathbf{\hat{D}}^{-1/2}

where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `GDC`

Processes the graph via Graph Diffusion Convolution (GDC) from the
`"Diffusion Improves Graph Learning" <https://arxiv.org/abs/1911.05485>`_
paper (functional name: :obj:`gdc`).

.. note::

    The paper offers additional advice on how to choose the
    hyperparameters.
    For an example of using GCN with GDC, see `examples/gcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    gcn.py>`_.

Args:
    self_loop_weight (float, optional): Weight of the added self-loop.
        Set to :obj:`None` to add no self-loops. (default: :obj:`1`)
    normalization_in (str, optional): Normalization of the transition
        matrix on the original (input) graph. Possible values:
        :obj:`"sym"`, :obj:`"col"`, and :obj:`"row"`.
        See :func:`GDC.transition_matrix` for details.
        (default: :obj:`"sym"`)
    normalization_out (str, optional): Normalization of the transition
        matrix on the transformed GDC (output) graph. Possible values:
        :obj:`"sym"`, :obj:`"col"`, :obj:`"row"`, and :obj:`None`.
        See :func:`GDC.transition_matrix` for details.
        (default: :obj:`"col"`)
    diffusion_kwargs (dict, optional): Dictionary containing the parameters
        for diffusion.
        `method` specifies the diffusion method (:obj:`"ppr"`,
        :obj:`"heat"` or :obj:`"coeff"`).
        Each diffusion method requires different additional parameters.
        See :func:`GDC.diffusion_matrix_exact` or
        :func:`GDC.diffusion_matrix_approx` for details.
        (default: :obj:`dict(method='ppr', alpha=0.15)`)
    sparsification_kwargs (dict, optional): Dictionary containing the
        parameters for sparsification.
        `method` specifies the sparsification method (:obj:`"threshold"` or
        :obj:`"topk"`).
        Each sparsification method requires different additional
        parameters.
        See :func:`GDC.sparsify_dense` for details.
        (default: :obj:`dict(method='threshold', avg_degree=64)`)
    exact (bool, optional): Whether to exactly calculate the diffusion
        matrix.
        Note that the exact variants are not scalable.
        They densify the adjacency matrix and calculate either its inverse
        or its matrix exponential.
        However, the approximate variants do not support edge weights and
        currently only personalized PageRank and sparsification by
        threshold are implemented as fast, approximate versions.
        (default: :obj:`True`)

:rtype: :class:`torch_geometric.data.Data`

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

- **`transition_matrix(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, normalization: str) -> Tuple[torch.Tensor, torch.Tensor]`**
  Calculate the approximate, sparse diffusion on a given sparse

- **`diffusion_matrix_exact(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int, method: str, **kwargs: Any) -> torch.Tensor`**
  Calculate the (dense) diffusion on a given sparse graph.

### `GenerateMeshNormals`

Generate normal vectors for each mesh node based on neighboring
faces (functional name: :obj:`generate_mesh_normals`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `GridSampling`

Clusters points into fixed-sized voxels
(functional name: :obj:`grid_sampling`).
Each cluster returned is a new point based on the mean of all points
inside the given cluster.

Args:
    size (float or [float] or Tensor): Size of a voxel (in each dimension).
    start (float or [float] or Tensor, optional): Start coordinates of the
        grid (in each dimension). If set to :obj:`None`, will be set to the
        minimum coordinates found in :obj:`data.pos`.
        (default: :obj:`None`)
    end (float or [float] or Tensor, optional): End coordinates of the grid
        (in each dimension). If set to :obj:`None`, will be set to the
        maximum coordinates found in :obj:`data.pos`.
        (default: :obj:`None`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `HalfHop`

The graph upsampling augmentation from the
`"Half-Hop: A Graph Upsampling Approach for Slowing Down Message Passing"
<https://openreview.net/forum?id=lXczFIwQkv>`_ paper.
The graph is augmented by adding artificial slow nodes between neighbors
to slow down message propagation. (functional name: :obj:`half_hop`).

.. note::
    :class:`HalfHop` augmentation is not supported if :obj:`data` has
    :attr:`edge_weight` or :attr:`edge_attr`.

Args:
    alpha (float, optional): The interpolation factor
        used to compute slow node features
        :math:`x = \alpha*x_src + (1-\alpha)*x_dst` (default: :obj:`0.5`)
    p (float, optional): The probability of half-hopping
        an edge. (default: :obj:`1.0`)

.. code-block:: python

    import torch_geometric.transforms as T

    transform = T.HalfHop(alpha=0.5)
    data = transform(data)  # Apply transformation.
    out = model(data.x, data.edge_index)  # Feed-forward.
    out = out[~data.slow_node_mask]  # Get rid of slow nodes.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `IndexToMask`

Converts indices to a mask representation
(functional name: :obj:`index_to_mask`).

Args:
    attrs (str, [str], optional): If given, will only perform index to mask
        conversion for the given attributes. If omitted, will infer the
        attributes from the suffix :obj:`_index`. (default: :obj:`None`)
    sizes (int, [int], optional): The size of the mask. If set to
        :obj:`None`, an automatically sized tensor is returned. The number
        of nodes will be used by default, except for edge attributes which
        will use the number of edges as the mask size.
        (default: :obj:`None`)
    replace (bool, optional): if set to :obj:`True` replaces the index
        attributes with mask tensors. (default: :obj:`False`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `KNNGraph`

Creates a k-NN graph based on node positions :obj:`data.pos`
(functional name: :obj:`knn_graph`).

Args:
    k (int, optional): The number of neighbors. (default: :obj:`6`)
    loop (bool, optional): If :obj:`True`, the graph will contain
        self-loops. (default: :obj:`False`)
    force_undirected (bool, optional): If set to :obj:`True`, new edges
        will be undirected. (default: :obj:`False`)
    flow (str, optional): The flow direction when used in combination with
        message passing (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`).
        If set to :obj:`"source_to_target"`, every target node will have
        exactly :math:`k` source nodes pointing to it.
        (default: :obj:`"source_to_target"`)
    cosine (bool, optional): If :obj:`True`, will use the cosine
        distance instead of euclidean distance to find nearest neighbors.
        (default: :obj:`False`)
    num_workers (int): Number of workers to use for computation. Has no
        effect in case :obj:`batch` is not :obj:`None`, or the input lies
        on the GPU. (default: :obj:`1`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `LaplacianLambdaMax`

Computes the highest eigenvalue of the graph Laplacian given by
:meth:`torch_geometric.utils.get_laplacian`
(functional name: :obj:`laplacian_lambda_max`).

Args:
    normalization (str, optional): The normalization scheme for the graph
        Laplacian (default: :obj:`None`):

        1. :obj:`None`: No normalization
        :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

        2. :obj:`"sym"`: Symmetric normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}`

        3. :obj:`"rw"`: Random-walk normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
    is_undirected (bool, optional): If set to :obj:`True`, this transform
        expects undirected graphs as input, and can hence speed up the
        computation of the largest eigenvalue. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `LargestConnectedComponents`

Selects the subgraph that corresponds to the
largest connected components in the graph
(functional name: :obj:`largest_connected_components`).

Args:
    num_components (int, optional): Number of largest components to keep
        (default: :obj:`1`)
    connection (str, optional): Type of connection to use for directed
        graphs, can be either :obj:`'strong'` or :obj:`'weak'`.
        Nodes `i` and `j` are strongly connected if a path
        exists both from `i` to `j` and from `j` to `i`. A directed graph
        is weakly connected if replacing all of its directed edges with
        undirected edges produces a connected (undirected) graph.
        (default: :obj:`'weak'`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `LineGraph`

Converts a graph to its corresponding line-graph
(functional name: :obj:`line_graph`).

.. math::
    L(\mathcal{G}) &= (\mathcal{V}^{\prime}, \mathcal{E}^{\prime})

    \mathcal{V}^{\prime} &= \mathcal{E}

    \mathcal{E}^{\prime} &= \{ (e_1, e_2) : e_1 \cap e_2 \neq \emptyset \}

Line-graph node indices are equal to indices in the original graph's
coalesced :obj:`edge_index`.
For undirected graphs, the maximum line-graph node index is
:obj:`(data.edge_index.size(1) // 2) - 1`.

New node features are given by old edge attributes.
For undirected graphs, edge attributes for reciprocal edges
:obj:`(row, col)` and :obj:`(col, row)` get summed together.

Args:
    force_directed (bool, optional): If set to :obj:`True`, the graph will
        be always treated as a directed graph. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `LinearTransformation`

Transforms node positions :obj:`data.pos` with a square transformation
matrix computed offline (functional name: :obj:`linear_transformation`).

Args:
    matrix (Tensor): Tensor with shape :obj:`[D, D]` where :obj:`D`
        corresponds to the dimensionality of node positions.

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `LocalCartesian`

Saves the relative Cartesian coordinates of linked nodes in its edge
attributes (functional name: :obj:`local_cartesian`). Each coordinate gets
*neighborhood-normalized* to a specified interval
(:math:`[0, 1]` by default).

Args:
    norm (bool, optional): If set to :obj:`False`, the output will not be
        normalized. (default: :obj:`True`)
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)
    interval ((float, float), optional): A tuple specifying the lower and
        upper bound for normalization. (default: :obj:`(0.0, 1.0)`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `LocalDegreeProfile`

Appends the Local Degree Profile (LDP) from the `"A Simple yet
Effective Baseline for Non-attribute Graph Classification"
<https://arxiv.org/abs/1811.03508>`_ paper
(functional name: :obj:`local_degree_profile`).

.. math::
    \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
    \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
\mathcal{N}(i) \}`.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `MaskToIndex`

Converts a mask to an index representation
(functional name: :obj:`mask_to_index`).

Args:
    attrs (str, [str], optional): If given, will only perform mask to index
        conversion for the given attributes.  If omitted, will infer the
        attributes from the suffix :obj:`_mask` (default: :obj:`None`)
    replace (bool, optional): if set to :obj:`True` replaces the mask
        attributes with index tensors. (default: :obj:`False`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `NodePropertySplit`

Creates a node-level split with distributional shift based on a given
node property, as proposed in the `"Evaluating Robustness and Uncertainty
of Graph Models Under Structural Distributional Shifts"
<https://arxiv.org/abs/2302.13875>`__ paper
(functional name: :obj:`node_property_split`).

It splits the nodes in a given graph into five non-intersecting parts
based on their structural properties.
This can be used for transductive node prediction tasks with distributional
shifts.
It considers the in-distribution (ID) and out-of-distribution (OOD) subsets
of nodes.
The ID subset includes training, validation and testing parts, while
the OOD subset includes validation and testing parts.
As a result, it creates five associated node mask vectors for each graph,
three which are for the ID nodes (:obj:`id_train_mask`,
:obj:`id_val_mask`, :obj:`id_test_mask`), and two which are for the OOD
nodes (:obj:`ood_val_mask`, :obj:`ood_test_mask`).

This class implements three particular strategies for inducing
distributional shifts in a graph — based on **popularity**, **locality**
or **density**.

Args:
    property_name (str): The name of the node property to be used
        (:obj:`"popularity"`, :obj:`"locality"`, :obj:`"density"`).
    ratios ([float]): A list of five ratio values for ID training,
        ID validation, ID test, OOD validation and OOD test parts.
        The values must sum to :obj:`1.0`.
    ascending (bool, optional): Whether to sort nodes in ascending order
        of the node property, so that nodes with greater values of the
        property are considered to be OOD (default: :obj:`True`)

.. code-block:: python

    from torch_geometric.transforms import NodePropertySplit
    from torch_geometric.datasets.graph_generator import ERGraph

    data = ERGraph(num_nodes=1000, edge_prob=0.4)()

    property_name = 'popularity'
    ratios = [0.3, 0.1, 0.1, 0.3, 0.2]
    transform = NodePropertySplit(property_name, ratios)

    data = transform(data)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `NormalizeFeatures`

Row-normalizes the attributes given in :obj:`attrs` to sum-up to one
(functional name: :obj:`normalize_features`).

Args:
    attrs (List[str]): The names of attributes to normalize.
        (default: :obj:`["x"]`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `NormalizeRotation`

Rotates all points according to the eigenvectors of the point cloud
(functional name: :obj:`normalize_rotation`).
If the data additionally holds normals saved in :obj:`data.normal`, these
will be rotated accordingly.

Args:
    max_points (int, optional): If set to a value greater than :obj:`0`,
        only a random number of :obj:`max_points` points are sampled and
        used to compute eigenvectors. (default: :obj:`-1`)
    sort (bool, optional): If set to :obj:`True`, will sort eigenvectors
        according to their eigenvalues. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `NormalizeScale`

Centers and normalizes node positions to the interval :math:`(-1, 1)`
(functional name: :obj:`normalize_scale`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `OneHotDegree`

Adds the node degree as one hot encodings to the node features
(functional name: :obj:`one_hot_degree`).

Args:
    max_degree (int): Maximum degree.
    in_degree (bool, optional): If set to :obj:`True`, will compute the
        in-degree of nodes instead of the out-degree.
        (default: :obj:`False`)
    cat (bool, optional): Concat node degrees to node features instead
        of replacing them. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Pad`

Applies padding to enforce consistent tensor shapes
(functional name: :obj:`pad`).

This transform will pad node and edge features up to a maximum allowed size
in the node or edge feature dimension. By default :obj:`0.0` is used as the
padding value and can be configured by setting :obj:`node_pad_value` and
:obj:`edge_pad_value`.

In case of applying :class:`Pad` to a :class:`~torch_geometric.data.Data`
object, the :obj:`node_pad_value` value (or :obj:`edge_pad_value`) can be
either:

* an int, float or object of :class:`UniformPadding` class for cases when
  all attributes are going to be padded with the same value;
* an object of :class:`AttrNamePadding` class for cases when padding is
  going to differ based on attribute names.

In case of applying :class:`Pad` to a
:class:`~torch_geometric.data.HeteroData` object, the :obj:`node_pad_value`
value (or :obj:`edge_pad_value`) can be either:

* an int, float or object of :class:`UniformPadding` class for cases when
  all attributes of all node (or edge) stores are going to be padded with
  the same value;
* an object of :class:`AttrNamePadding` class for cases when padding is
  going to differ based on attribute names (but not based on node or edge
  types);
* an object of class :class:`NodeTypePadding` or :class:`EdgeTypePadding`
  for cases when padding values are going to differ based on node or edge
  types. Padding values can also differ based on attribute names for a
  given node or edge type by using :class:`AttrNamePadding` objects as
  values of its `values` argument.

Note that in order to allow for consistent padding across all graphs in a
dataset, below conditions must be met:

* if :obj:`max_num_nodes` is a single value, it must be greater than or
  equal to the maximum number of nodes of any graph in the dataset;
* if :obj:`max_num_nodes` is a dictionary, value for every node type must
  be greater than or equal to the maximum number of this type nodes of any
  graph in the dataset.

Example below shows how to create a :class:`Pad` transform for an
:class:`~torch_geometric.data.HeteroData` object. The object is padded to
have :obj:`10` nodes of type :obj:`v0`, :obj:`20` nodes of type :obj:`v1`
and :obj:`30` nodes of type :obj:`v2`.
It is padded to have :obj:`80` edges of type :obj:`('v0', 'e0', 'v1')`.
All the attributes of the :obj:`v0` nodes are padded using a value of
:obj:`3.0`.
The :obj:`x` attribute of the :obj:`v1` node type is padded using a value
of :obj:`-1.0`, and the other attributes of this node type are padded using
a value of :obj:`0.5`.
All the attributes of node types other than :obj:`v0` and :obj:`v1` are
padded using a value of :obj:`1.0`.
All the attributes of the :obj:`('v0', 'e0', 'v1')` edge type are padded
using a value of :obj:`3.5`.
The :obj:`edge_attr` attributes of the :obj:`('v1', 'e0', 'v0')` edge type
are padded using a value of :obj:`-1.5`, and any other attributes of this
edge type are padded using a value of :obj:`5.5`.
All the attributes of edge types other than these two are padded using a
value of :obj:`1.5`.

.. code-block:: python

    num_nodes = {'v0': 10, 'v1': 20, 'v2':30}
    num_edges = {('v0', 'e0', 'v1'): 80}

    node_padding = NodeTypePadding({
        'v0': 3.0,
        'v1': AttrNamePadding({'x': -1.0}, default=0.5),
    }, default=1.0)

    edge_padding = EdgeTypePadding({
        ('v0', 'e0', 'v1'): 3.5,
        ('v1', 'e0', 'v0'): AttrNamePadding({'edge_attr': -1.5},
                                            default=5.5),
    }, default=1.5)

    transform = Pad(num_nodes, num_edges, node_padding, edge_padding)

Args:
    max_num_nodes (int or dict): The number of nodes after padding.
        In heterogeneous graphs, may also take in a dictionary denoting the
        number of nodes for specific node types.
    max_num_edges (int or dict, optional): The number of edges after
        padding.
        In heterogeneous graphs, may also take in a dictionary denoting the
        number of edges for specific edge types. (default: :obj:`None`)
    node_pad_value (int or float or Padding, optional): The fill value to
        use for node features. (default: :obj:`0.0`)
    edge_pad_value (int or float or Padding, optional): The fill value to
        use for edge features. (default: :obj:`0.0`)
        The :obj:`edge_index` tensor is padded with with the index of the
        first padded node (which represents a set of self-loops on the
        padded node). (default: :obj:`0.0`)
    mask_pad_value (bool, optional): The fill value to use for
        :obj:`train_mask`, :obj:`val_mask` and :obj:`test_mask` attributes
        (default: :obj:`False`).
    add_pad_mask (bool, optional): If set to :obj:`True`, will attach
        node-level :obj:`pad_node_mask` and edge-level :obj:`pad_edge_mask`
        attributes to the output which indicates which elements in the data
        are real (represented by :obj:`True`) and which were added as a
        result of padding (represented by :obj:`False`).
        (default: :obj:`False`)
    exclude_keys ([str], optional): Keys to be removed
        from the input data object. (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `PointPairFeatures`

Computes the rotation-invariant Point Pair Features
(functional name: :obj:`point_pair_features`).

.. math::
    \left( \| \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
    \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
    \mathbf{n}_j) \right)

of linked nodes in its edge attributes, where :math:`\mathbf{d}_{j,i}`
denotes the difference vector between, and :math:`\mathbf{n}_i` and
:math:`\mathbf{n}_j` denote the surface normals of node :math:`i` and
:math:`j` respectively.

Args:
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Polar`

Saves the polar coordinates of linked nodes in its edge attributes
(functional name: :obj:`polar`).

Args:
    norm (bool, optional): If set to :obj:`False`, the output will not be
        normalized to the interval :math:`{[0, 1]}^2`.
        (default: :obj:`True`)
    max_value (float, optional): If set and :obj:`norm=True`, normalization
        will be performed based on this value instead of the maximum value
        found in the data. (default: :obj:`None`)
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RadiusGraph`

Creates edges based on node positions :obj:`data.pos` to all points
within a given distance (functional name: :obj:`radius_graph`).

Args:
    r (float): The distance.
    loop (bool, optional): If :obj:`True`, the graph will contain
        self-loops. (default: :obj:`False`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        return for each element in :obj:`y`.
        This flag is only needed for CUDA tensors. (default: :obj:`32`)
    flow (str, optional): The flow direction when using in combination with
        message passing (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    num_workers (int): Number of workers to use for computation. Has no
        effect in case :obj:`batch` is not :obj:`None`, or the input lies
        on the GPU. (default: :obj:`1`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomFlip`

Flips node positions along a given axis randomly with a given
probability (functional name: :obj:`random_flip`).

Args:
    axis (int): The axis along the position of nodes being flipped.
    p (float, optional): Probability that node positions will be flipped.
        (default: :obj:`0.5`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomJitter`

Translates node positions by randomly sampled translation values
within a given interval (functional name: :obj:`random_jitter`).
In contrast to other random transformations,
translation is applied separately at each position.

Args:
    translate (sequence or float or int): Maximum translation in each
        dimension, defining the range
        :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
        If :obj:`translate` is a number instead of a sequence, the same
        range is used for each dimension.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomLinkSplit`

Performs an edge-level random split into training, validation and test
sets of a :class:`~torch_geometric.data.Data` or a
:class:`~torch_geometric.data.HeteroData` object
(functional name: :obj:`random_link_split`).
The split is performed such that the training split does not include edges
in validation and test splits; and the validation split does not include
edges in the test split.

.. code-block:: python

    from torch_geometric.transforms import RandomLinkSplit

    transform = RandomLinkSplit(is_undirected=True)
    train_data, val_data, test_data = transform(data)

Args:
    num_val (int or float, optional): The number of validation edges.
        If set to a floating-point value in :math:`[0, 1]`, it represents
        the ratio of edges to include in the validation set.
        (default: :obj:`0.1`)
    num_test (int or float, optional): The number of test edges.
        If set to a floating-point value in :math:`[0, 1]`, it represents
        the ratio of edges to include in the test set.
        (default: :obj:`0.2`)
    is_undirected (bool): If set to :obj:`True`, the graph is assumed to be
        undirected, and positive and negative samples will not leak
        (reverse) edge connectivity across different splits. This only
        affects the graph split, label data will not be returned
        undirected. This option is ignored for bipartite edge types or
        whenever :obj:`edge_type != rev_edge_type`. (default: :obj:`False`)
    key (str, optional): The name of the attribute holding
        ground-truth labels.
        If :obj:`data[key]` does not exist, it will be automatically
        created and represents a binary classification task
        (:obj:`1` = edge, :obj:`0` = no edge).
        If :obj:`data[key]` exists, it has to be a categorical label from
        :obj:`0` to :obj:`num_classes - 1`.
        After negative sampling, label :obj:`0` represents negative edges,
        and labels :obj:`1` to :obj:`num_classes` represent the labels of
        positive edges. (default: :obj:`"edge_label"`)
    split_labels (bool, optional): If set to :obj:`True`, will split
        positive and negative labels and save them in distinct attributes
        :obj:`"pos_edge_label"` and :obj:`"neg_edge_label"`, respectively.
        (default: :obj:`False`)
    add_negative_train_samples (bool, optional): Whether to add negative
        training samples for link prediction.
        If the model already performs negative sampling, then the option
        should be set to :obj:`False`.
        Otherwise, the added negative samples will be the same across
        training iterations unless negative sampling is performed again.
        (default: :obj:`True`)
    neg_sampling_ratio (float, optional): The ratio of sampled negative
        edges to the number of positive edges. (default: :obj:`1.0`)
    disjoint_train_ratio (int or float, optional): If set to a value
        greater than :obj:`0.0`, training edges will not be shared for
        message passing and supervision. Instead,
        :obj:`disjoint_train_ratio` edges are used as ground-truth labels
        for supervision during training. (default: :obj:`0.0`)
    edge_types (Tuple[EdgeType] or List[EdgeType], optional): The edge
        types used for performing edge-level splitting in case of
        operating on :class:`~torch_geometric.data.HeteroData` objects.
        (default: :obj:`None`)
    rev_edge_types (Tuple[EdgeType] or List[Tuple[EdgeType]], optional):
        The reverse edge types of :obj:`edge_types` in case of operating
        on :class:`~torch_geometric.data.HeteroData` objects.
        This will ensure that edges of the reverse direction will be
        split accordingly to prevent any data leakage.
        Can be :obj:`None` in case no reverse connection exists.
        (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Tuple[Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData], Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData], Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]]`**

### `RandomNodeSplit`

Performs a node-level random split by adding :obj:`train_mask`,
:obj:`val_mask` and :obj:`test_mask` attributes to the
:class:`~torch_geometric.data.Data` or
:class:`~torch_geometric.data.HeteroData` object
(functional name: :obj:`random_node_split`).

Args:
    split (str, optional): The type of dataset split (:obj:`"train_rest"`,
        :obj:`"test_rest"`, :obj:`"random"`).
        If set to :obj:`"train_rest"`, all nodes except those in the
        validation and test sets will be used for training (as in the
        `"FastGCN: Fast Learning with Graph Convolutional Networks via
        Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
        If set to :obj:`"test_rest"`, all nodes except those in the
        training and validation sets will be used for test (as in the
        `"Pitfalls of Graph Neural Network Evaluation"
        <https://arxiv.org/abs/1811.05868>`_ paper).
        If set to :obj:`"random"`, train, validation, and test sets will be
        randomly generated, according to :obj:`num_train_per_class`,
        :obj:`num_val` and :obj:`num_test` (as in the `"Semi-supervised
        Classification with Graph Convolutional Networks"
        <https://arxiv.org/abs/1609.02907>`_ paper).
        (default: :obj:`"train_rest"`)
    num_splits (int, optional): The number of splits to add. If bigger
        than :obj:`1`, the shape of masks will be
        :obj:`[num_nodes, num_splits]`, and :obj:`[num_nodes]` otherwise.
        (default: :obj:`1`)
    num_train_per_class (int, optional): The number of training samples
        per class in case of :obj:`"test_rest"` and :obj:`"random"` split.
        (default: :obj:`20`)
    num_val (int or float, optional): The number of validation samples.
        If float, it represents the ratio of samples to include in the
        validation set. (default: :obj:`500`)
    num_test (int or float, optional): The number of test samples in case
        of :obj:`"train_rest"` and :obj:`"random"` split. If float, it
        represents the ratio of samples to include in the test set.
        (default: :obj:`1000`)
    key (str, optional): The name of the attribute holding ground-truth
        labels. By default, will only add node-level splits for node-level
        storages in which :obj:`key` is present. (default: :obj:`"y"`).

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `RandomRotate`

Rotates node positions around a specific axis by a randomly sampled
factor within a given interval (functional name: :obj:`random_rotate`).

Args:
    degrees (tuple or float): Rotation interval from which the rotation
        angle is sampled. If :obj:`degrees` is a number instead of a
        tuple, the interval is given by :math:`[-\mathrm{degrees},
        \mathrm{degrees}]`.
    axis (int, optional): The rotation axis. (default: :obj:`0`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomScale`

Scales node positions by a randomly sampled factor :math:`s` within a
given interval, *e.g.*, resulting in the transformation matrix
(functional name: :obj:`random_scale`).

.. math::
    \begin{bmatrix}
        s & 0 & 0 \\
        0 & s & 0 \\
        0 & 0 & s \\
    \end{bmatrix}

for three-dimensional positions.

Args:
    scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
        is randomly sampled from the range
        :math:`a \leq \mathrm{scale} \leq b`.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomShear`

Shears node positions by randomly sampled factors :math:`s` within a
given interval, *e.g.*, resulting in the transformation matrix
(functional name: :obj:`random_shear`).

.. math::
    \begin{bmatrix}
        1      & s_{xy} & s_{xz} \\
        s_{yx} & 1      & s_{yz} \\
        s_{zx} & z_{zy} & 1      \\
    \end{bmatrix}

for three-dimensional positions.

Args:
    shear (float or int): maximum shearing factor defining the range
        :math:`(-\mathrm{shear}, +\mathrm{shear})` to sample from.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RandomTranslate`

Translates node positions by randomly sampled translation values
within a given interval (functional name: :obj:`random_jitter`).
In contrast to other random transformations,
translation is applied separately at each position.

Args:
    translate (sequence or float or int): Maximum translation in each
        dimension, defining the range
        :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
        If :obj:`translate` is a number instead of a sequence, the same
        range is used for each dimension.

### `RemoveDuplicatedEdges`

Removes duplicated edges from a given homogeneous or heterogeneous
graph. Useful to clean-up known repeated edges/self-loops in common
benchmark datasets, *e.g.*, in :obj:`ogbn-products`.
(functional name: :obj:`remove_duplicated_edges`).

Args:
    key (str or [str], optional): The name of edge attribute(s) to merge in
        case of duplication. (default: :obj:`["edge_weight", "edge_attr"]`)
    reduce (str, optional): The reduce operation to use for merging edge
        attributes (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`,
        :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"add"`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `RemoveIsolatedNodes`

Removes isolated nodes from the graph
(functional name: :obj:`remove_isolated_nodes`).

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `RemoveSelfLoops`

Removes all self-loops in the given homogeneous or heterogeneous
graph (functional name: :obj:`remove_self_loops`).

Args:
    attr (str, optional): The name of the attribute of edge weights
        or multi-dimensional edge features to pass to
        :meth:`torch_geometric.utils.remove_self_loops`.
        (default: :obj:`"edge_weight"`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `RemoveTrainingClasses`

Removes classes from the node-level training set as given by
:obj:`data.train_mask`, *e.g.*, in order to get a zero-shot label scenario
(functional name: :obj:`remove_training_classes`).

Args:
    classes (List[int]): The classes to remove from the training set.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `RootedEgoNets`

Collects rooted :math:`k`-hop EgoNets for each node in the graph, as
described in the `"From Stars to Subgraphs: Uplifting Any GNN with Local
Structure Awareness" <https://arxiv.org/abs/2110.03753>`_ paper.

Args:
    num_hops (int): the number of hops :math:`k`.

#### Methods

- **`extract(self, data: torch_geometric.data.data.Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`**

### `RootedRWSubgraph`

Collects rooted random-walk based subgraphs for each node in the graph,
as described in the `"From Stars to Subgraphs: Uplifting Any GNN with Local
Structure Awareness" <https://arxiv.org/abs/2110.03753>`_ paper.

Args:
    walk_length (int): the length of the random walk.
    repeat (int, optional): The number of times of repeating the random
        walk to reduce randomness. (default: :obj:`1`)

#### Methods

- **`extract(self, data: torch_geometric.data.data.Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`**

### `SIGN`

The Scalable Inception Graph Neural Network module (SIGN) from the
`"SIGN: Scalable Inception Graph Neural Networks"
<https://arxiv.org/abs/2004.11198>`_ paper (functional name: :obj:`sign`),
which precomputes the fixed representations.

.. math::
    \mathbf{X}^{(i)} = {\left( \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2} \right)}^i \mathbf{X}

for :math:`i \in \{ 1, \ldots, K \}` and saves them in
:obj:`data.x1`, :obj:`data.x2`, ...

.. note::

    Since intermediate node representations are pre-computed, this operator
    is able to scale well to large graphs via classic mini-batching.
    For an example of using SIGN, see `examples/sign.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    sign.py>`_.

Args:
    K (int): The number of hops/layer.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `SVDFeatureReduction`

Dimensionality reduction of node features via Singular Value
Decomposition (SVD) (functional name: :obj:`svd_feature_reduction`).

Args:
    out_channels (int): The dimensionality of node features after
        reduction.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `SamplePoints`

Uniformly samples a fixed number of points on the mesh faces according
to their face area (functional name: :obj:`sample_points`).

Args:
    num (int): The number of points to sample.
    remove_faces (bool, optional): If set to :obj:`False`, the face tensor
        will not be removed. (default: :obj:`True`)
    include_normals (bool, optional): If set to :obj:`True`, then compute
        normals for each sampled point. (default: :obj:`False`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `Spherical`

Saves the spherical coordinates of linked nodes in its edge attributes
(functional name: :obj:`spherical`).

Args:
    norm (bool, optional): If set to :obj:`False`, the output will not be
        normalized to the interval :math:`{[0, 1]}^3`.
        (default: :obj:`True`)
    max_value (float, optional): If set and :obj:`norm=True`, normalization
        will be performed based on this value instead of the maximum value
        found in the data. (default: :obj:`None`)
    cat (bool, optional): If set to :obj:`False`, all existing edge
        attributes will be replaced. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `TargetIndegree`

Saves the globally normalized degree of target nodes
(functional name: :obj:`target_indegree`).

.. math::

    \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

in its edge attributes.

Args:
    cat (bool, optional): Concat pseudo-coordinates to edge attributes
        instead of replacing them. (default: :obj:`True`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `ToDense`

Converts a sparse adjacency matrix to a dense adjacency matrix with
shape :obj:`[num_nodes, num_nodes, *]` (functional name: :obj:`to_dense`).

Args:
    num_nodes (int, optional): The number of nodes. If set to :obj:`None`,
        the number of nodes will get automatically inferred.
        (default: :obj:`None`)

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `ToDevice`

Performs tensor device conversion, either for all attributes of the
:obj:`~torch_geometric.data.Data` object or only the ones given by
:obj:`attrs` (functional name: :obj:`to_device`).

Args:
    device (torch.device): The destination device.
    attrs (List[str], optional): If given, will only perform tensor device
        conversion for the given attributes. (default: :obj:`None`)
    non_blocking (bool, optional): If set to :obj:`True` and tensor
        values are in pinned memory, the copy will be asynchronous with
        respect to the host. (default: :obj:`False`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `ToSLIC`

Converts an image to a superpixel representation using the
:meth:`skimage.segmentation.slic` algorithm, resulting in a
:obj:`torch_geometric.data.Data` object holding the centroids of
superpixels in :obj:`data.pos` and their mean color in :obj:`data.x`
(functional name: :obj:`to_slic`).

This transform can be used with any :obj:`torchvision` dataset.

.. code-block:: python

    from torchvision.datasets import MNIST
    import torchvision.transforms as T
    from torch_geometric.transforms import ToSLIC

    transform = T.Compose([T.ToTensor(), ToSLIC(n_segments=75)])
    dataset = MNIST('/tmp/MNIST', download=True, transform=transform)

Args:
    add_seg (bool, optional): If set to `True`, will add the segmentation
        result to the data object. (default: :obj:`False`)
    add_img (bool, optional): If set to `True`, will add the input image
        to the data object. (default: :obj:`False`)
    **kwargs (optional): Arguments to adjust the output of the SLIC
        algorithm. See the `SLIC documentation
        <https://scikit-image.org/docs/dev/api/skimage.segmentation.html
        #skimage.segmentation.slic>`_ for an overview.

#### Methods

- **`forward(self, img: torch.Tensor) -> torch_geometric.data.data.Data`**

### `ToSparseTensor`

Converts the :obj:`edge_index` attributes of a homogeneous or
heterogeneous data object into a **transposed**
:class:`torch_sparse.SparseTensor` or :pytorch:`PyTorch`
:class:`torch.sparse.Tensor` object with key :obj:`adj_t`
(functional name: :obj:`to_sparse_tensor`).

.. note::

    In case of composing multiple transforms, it is best to convert the
    :obj:`data` object via :class:`ToSparseTensor` as late as possible,
    since there exist some transforms that are only able to operate on
    :obj:`data.edge_index` for now.

Args:
    attr (str, optional): The name of the attribute to add as a value to
        the :class:`~torch_sparse.SparseTensor` or
        :class:`torch.sparse.Tensor` object (if present).
        (default: :obj:`edge_weight`)
    remove_edge_index (bool, optional): If set to :obj:`False`, the
        :obj:`edge_index` tensor will not be removed.
        (default: :obj:`True`)
    fill_cache (bool, optional): If set to :obj:`True`, will fill the
        underlying :class:`torch_sparse.SparseTensor` cache (if used).
        (default: :obj:`True`)
    layout (torch.layout, optional): Specifies the layout of the returned
        sparse tensor (:obj:`None`, :obj:`torch.sparse_coo` or
        :obj:`torch.sparse_csr`).
        If set to :obj:`None` and the :obj:`torch_sparse` dependency is
        installed, will convert :obj:`edge_index` into a
        :class:`torch_sparse.SparseTensor` object.
        If set to :obj:`None` and the :obj:`torch_sparse` dependency is
        not installed, will convert :obj:`edge_index` into a
        :class:`torch.sparse.Tensor` object with layout
        :obj:`torch.sparse_csr`. (default: :obj:`None`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `ToUndirected`

Converts a homogeneous or heterogeneous graph to an undirected graph
such that :math:`(j,i) \in \mathcal{E}` for every edge
:math:`(i,j) \in \mathcal{E}` (functional name: :obj:`to_undirected`).
In heterogeneous graphs, will add "reverse" connections for *all* existing
edge types.

Args:
    reduce (str, optional): The reduce operation to use for merging edge
        features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`). (default: :obj:`"add"`)
    merge (bool, optional): If set to :obj:`False`, will create reverse
        edge types for connections pointing to the same source and target
        node type.
        If set to :obj:`True`, reverse edges will be merged into the
        original relation.
        This option only has effects in
        :class:`~torch_geometric.data.HeteroData` graph data.
        (default: :obj:`True`)

#### Methods

- **`forward(self, data: Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]) -> Union[torch_geometric.data.data.Data, torch_geometric.data.hetero_data.HeteroData]`**

### `TwoHop`

Adds the two hop edges to the edge indices
(functional name: :obj:`two_hop`).

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**

### `VirtualNode`

Appends a virtual node to the given homogeneous graph that is connected
to all other nodes, as described in the `"Neural Message Passing for
Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper
(functional name: :obj:`virtual_node`).
The virtual node serves as a global scratch space that each node both reads
from and writes to in every step of message passing.
This allows information to travel long distances during the propagation
phase.

Node and edge features of the virtual node are added as zero-filled input
features.
Furthermore, special edge types will be added both for in-coming and
out-going information to and from the virtual node.

#### Methods

- **`forward(self, data: torch_geometric.data.data.Data) -> torch_geometric.data.data.Data`**
