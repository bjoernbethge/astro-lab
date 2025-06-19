# nn Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.nn`

## Functions (33)

### `approx_knn(x: torch.Tensor, y: torch.Tensor, k: int, batch_x: torch.Tensor = None, batch_y: torch.Tensor = None) -> torch.Tensor`

Finds for each element in :obj:`y` the :obj:`k` approximated nearest
points in :obj:`x`.

.. note::

    Approximated :math:`k`-nearest neighbor search is performed via the
    `pynndescent <https://pynndescent.readthedocs.io/en/latest>`_ library.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    y (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
    k (int): The number of neighbors.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
        node to a specific example. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `approx_knn_graph(x: torch.Tensor, k: int, batch: torch.Tensor = None, loop: bool = False, flow: str = 'source_to_target') -> torch.Tensor`

Computes graph edges to the nearest approximated :obj:`k` points.

.. note::

    Approximated :math:`k`-nearest neighbor search is performed via the
    `pynndescent <https://pynndescent.readthedocs.io/en/latest>`_ library.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    k (int): The number of neighbors.
    batch (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    loop (bool, optional): If :obj:`True`, the graph will contain
        self-loops. (default: :obj:`False`)
    flow (str, optional): The flow direction when using in combination with
        message passing (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

:rtype: :class:`torch.Tensor`

### `avg_pool(cluster: torch.Tensor, data: torch_geometric.data.data.Data, transform: Optional[Callable] = None) -> torch_geometric.data.data.Data`

Pools and coarsens a graph given by the
:class:`torch_geometric.data.Data` object according to the clustering
defined in :attr:`cluster`.
Final node features are defined by the *average* features of all nodes
within the same cluster.
See :meth:`torch_geometric.nn.pool.max_pool` for more details.

Args:
    cluster (torch.Tensor): The cluster vector
        :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
        node to a specific cluster.
    data (Data): Graph data object.
    transform (callable, optional): A function/transform that takes in the
        coarsened and pooled :obj:`torch_geometric.data.Data` object and
        returns a transformed version. (default: :obj:`None`)

:rtype: :class:`torch_geometric.data.Data`

### `avg_pool_neighbor_x(data: torch_geometric.data.data.Data, flow: Optional[str] = 'source_to_target') -> torch_geometric.data.data.Data`

Average pools neighboring node features, where each feature in
:obj:`data.x` is replaced by the average feature values from the central
node and its neighbors.

### `avg_pool_x(cluster: torch.Tensor, x: torch.Tensor, batch: torch.Tensor, batch_size: Optional[int] = None, size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Average pools node features according to the clustering defined in
:attr:`cluster`.
See :meth:`torch_geometric.nn.pool.max_pool_x` for more details.

Args:
    cluster (torch.Tensor): The cluster vector
        :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
        node to a specific cluster.
    x (Tensor): The node feature matrix.
    batch (torch.Tensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example.
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)
    size (int, optional): The maximum number of clusters in a single
        example. (default: :obj:`None`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`) if :attr:`size` is
    :obj:`None`, else :class:`torch.Tensor`

### `bro(x: torch.Tensor, batch: torch.Tensor, p: Union[int, str] = 2) -> torch.Tensor`

The Batch Representation Orthogonality penalty from the `"Improving
Molecular Graph Neural Network Explainability with Orthonormalization
and Induced Sparsity" <https://arxiv.org/abs/2105.04854>`_ paper.

Computes a regularization for each graph representation in a mini-batch
according to

.. math::
    \mathcal{L}_{\textrm{BRO}}^\mathrm{graph} =
      || \mathbf{HH}^T - \mathbf{I}||_p

and returns an average over all graphs in the batch.

Args:
    x (torch.Tensor): The node feature matrix.
    batch (torch.Tensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    p (int or str, optional): The norm order to use. (default: :obj:`2`)

### `captum_output_to_dicts(captum_attrs: Tuple[torch.Tensor, ...], mask_type: Union[str, torch_geometric.explain.algorithm.captum.MaskLevelType], metadata: Tuple[List[str], List[Tuple[str, str, str]]]) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict[Tuple[str, str, str], torch.Tensor]]]`

Convert the output of `Captum <https://captum.ai/>`_ attribution
methods which is a tuple of attributions to two dictionaries with node and
edge attribution tensors. This function is used while explaining
:class:`~torch_geometric.data.HeteroData` objects.
See :meth:`~torch_geometric.nn.models.to_captum_model` for example usage.

Args:
    captum_attrs (tuple[torch.Tensor]): The output of attribution methods.
    mask_type (str): Denotes the type of mask to be created with
        a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
        and :obj:`"node_and_edge"`:

        1. :obj:`"edge"`: :obj:`captum_attrs` contains only edge
           attributions. The returned tuple has no node attributions, and
           an edge attribution dictionary edge types as keys and edge mask
           tensors of shape :obj:`[num_edges]` as values.

        2. :obj:`"node"`: :obj:`captum_attrs` contains only node
           attributions. The returned tuple has a node attribution
           dictionary with node types as keys and node mask tensors of
           shape :obj:`[num_nodes, num_features]` as values, and no edge
           attributions.

        3. :obj:`"node_and_edge"`: :obj:`captum_attrs` contains node and
            edge attributions.

    metadata (Metadata): The metadata of the heterogeneous graph.

### `dense_diff_pool(x: torch.Tensor, adj: torch.Tensor, s: torch.Tensor, mask: Optional[torch.Tensor] = None, normalize: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

The differentiable pooling operator from the `"Hierarchical Graph
Representation Learning with Differentiable Pooling"
<https://arxiv.org/abs/1806.08804>`_ paper.

.. math::
    \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{X}

    \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
\times N \times C}`.
Returns the pooled node feature matrix, the coarsened adjacency matrix and
two auxiliary objectives: (1) The link prediction loss

.. math::
    \mathcal{L}_{LP} = {\| \mathbf{A} -
    \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
    \|}_F,

and (2) the entropy regularization

.. math::
    \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

Args:
    x (torch.Tensor): Node feature tensor
        :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
        each graph, and feature dimension :math:`F`.
    adj (torch.Tensor): Adjacency tensor
        :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
    s (torch.Tensor): Assignment tensor
        :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
        with number of clusters :math:`C`.
        The softmax does not have to be applied before-hand, since it is
        executed within this method.
    mask (torch.Tensor, optional): Mask matrix
        :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
        the valid nodes for each graph. (default: :obj:`None`)
    normalize (bool, optional): If set to :obj:`False`, the link
        prediction loss is not divided by :obj:`adj.numel()`.
        (default: :obj:`True`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
    :class:`torch.Tensor`, :class:`torch.Tensor`)

### `dense_mincut_pool(x: torch.Tensor, adj: torch.Tensor, s: torch.Tensor, mask: Optional[torch.Tensor] = None, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

The MinCut pooling operator from the `"Spectral Clustering in Graph
Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
paper.

.. math::
    \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{X}

    \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
\times N \times C}`.
Returns the pooled node feature matrix, the coarsened and symmetrically
normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
loss

.. math::
    \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
    \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
    \mathbf{S})}

where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
loss

.. math::
    \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
    {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
    \right\|}_F.

Args:
    x (torch.Tensor): Node feature tensor
        :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
        batch-size :math:`B`, (maximum) number of nodes :math:`N` for
        each graph, and feature dimension :math:`F`.
    adj (torch.Tensor): Adjacency tensor
        :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
    s (torch.Tensor): Assignment tensor
        :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
        with number of clusters :math:`C`.
        The softmax does not have to be applied before-hand, since it is
        executed within this method.
    mask (torch.Tensor, optional): Mask matrix
        :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
        the valid nodes for each graph. (default: :obj:`None`)
    temp (float, optional): Temperature parameter for softmax function.
        (default: :obj:`1.0`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
    :class:`torch.Tensor`, :class:`torch.Tensor`)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

### `fps(x: torch.Tensor, batch: Optional[torch.Tensor] = None, ratio: float = 0.5, random_start: bool = True, batch_size: Optional[int] = None) -> torch.Tensor`

A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
Learning on Point Sets in a Metric Space"
<https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
most distant point with regard to the rest points.

.. code-block:: python

    import torch
    from torch_geometric.nn import fps

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch = torch.tensor([0, 0, 0, 0])
    index = fps(x, batch, ratio=0.5)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    batch (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    ratio (float, optional): Sampling ratio. (default: :obj:`0.5`)
    random_start (bool, optional): If set to :obj:`False`, use the first
        node in :math:`\mathbf{X}` as starting node. (default: obj:`True`)
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `gini(w: torch.Tensor) -> torch.Tensor`

The Gini coefficient from the `"Improving Molecular Graph Neural
Network Explainability with Orthonormalization and Induced Sparsity"
<https://arxiv.org/abs/2105.04854>`_ paper.

Computes a regularization penalty :math:`\in [0, 1]` for each row of a
matrix according to

.. math::
    \mathcal{L}_\textrm{Gini}^i = \sum_j^n \sum_{j'}^n \frac{|w_{ij}
     - w_{ij'}|}{2 (n^2 - n)\bar{w_i}}

and returns an average over all rows.

Args:
    w (torch.Tensor): A two-dimensional tensor.

### `global_add_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by adding node features
across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.SumAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `global_max_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by taking the channel-wise
maximum across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.MaxAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each element to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None) -> torch.Tensor`

Returns batch-wise graph-level-outputs by averaging node features
across the node dimension.

For a single graph :math:`\mathcal{G}_i`, its output is computed by

.. math::
    \mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.

Functional method of the
:class:`~torch_geometric.nn.aggr.MeanAggregation` module.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node to a specific example.
    size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

### `global_sort_pool(x, index, k)`

### `graclus(edge_index: torch.Tensor, weight: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None)`

A greedy clustering algorithm from the `"Weighted Graph Cuts without
Eigenvectors: A Multilevel Approach" <http://www.cs.utexas.edu/users/
inderjit/public_papers/multilevel_pami.pdf>`_ paper of picking an unmarked
vertex and matching it with one of its unmarked neighbors (that maximizes
its edge weight).
The GPU algorithm is adapted from the `"A GPU Algorithm for Greedy Graph
Matching" <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`_
paper.

Args:
    edge_index (torch.Tensor): The edge indices.
    weight (torch.Tensor, optional): One-dimensional edge weights.
        (default: :obj:`None`)
    num_nodes (int, optional): The number of nodes, *i.e.*
        :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `knn(x: torch.Tensor, y: torch.Tensor, k: int, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, cosine: bool = False, num_workers: int = 1, batch_size: Optional[int] = None) -> torch.Tensor`

Finds for each element in :obj:`y` the :obj:`k` nearest points in
:obj:`x`.

.. code-block:: python

    import torch
    from torch_geometric.nn import knn

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch_x = torch.tensor([0, 0, 0, 0])
    y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    batch_y = torch.tensor([0, 0])
    assign_index = knn(x, y, 2, batch_x, batch_y)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    y (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
    k (int): The number of neighbors.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
        node to a specific example. (default: :obj:`None`)
    cosine (bool, optional): If :obj:`True`, will use the cosine
        distance instead of euclidean distance to find nearest neighbors.
        (default: :obj:`False`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
        :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None, loop: bool = False, flow: str = 'source_to_target', cosine: bool = False, num_workers: int = 1, batch_size: Optional[int] = None) -> torch.Tensor`

Computes graph edges to the nearest :obj:`k` points.

.. code-block:: python

    import torch
    from torch_geometric.nn import knn_graph

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = knn_graph(x, k=2, batch=batch, loop=False)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    k (int): The number of neighbors.
    batch (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    loop (bool, optional): If :obj:`True`, the graph will contain
        self-loops. (default: :obj:`False`)
    flow (str, optional): The flow direction when using in combination with
        message passing (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    cosine (bool, optional): If :obj:`True`, will use the cosine
        distance instead of euclidean distance to find nearest neighbors.
        (default: :obj:`False`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch` is not :obj:`None`, or the input
        lies on the GPU. (default: :obj:`1`)
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `knn_interpolate(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, k: int = 3, num_workers: int = 1)`

The k-NN interpolation from the `"PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space"
<https://arxiv.org/abs/1706.02413>`_ paper.

For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
interpolated features :math:`\mathbf{f}(y)` are given by

.. math::
    \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
    w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
    \mathbf{p}(x_i))^2}

and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
to :math:`y`.

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    pos_x (torch.Tensor): Node position matrix
        :math:`\in \mathbb{R}^{N \times d}`.
    pos_y (torch.Tensor): Upsampled node position matrix
        :math:`\in \mathbb{R}^{M \times d}`.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node from :math:`\mathbf{X}` to a specific example.
        (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
        each node from :math:`\mathbf{Y}` to a specific example.
        (default: :obj:`None`)
    k (int, optional): Number of neighbors. (default: :obj:`3`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
        :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)

### `max_pool(cluster: torch.Tensor, data: torch_geometric.data.data.Data, transform: Optional[Callable] = None) -> torch_geometric.data.data.Data`

Pools and coarsens a graph given by the
:class:`torch_geometric.data.Data` object according to the clustering
defined in :attr:`cluster`.
All nodes within the same cluster will be represented as one node.
Final node features are defined by the *maximum* features of all nodes
within the same cluster, node positions are averaged and edge indices are
defined to be the union of the edge indices of all nodes within the same
cluster.

Args:
    cluster (torch.Tensor): The cluster vector
        :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
        node to a specific cluster.
    data (Data): Graph data object.
    transform (callable, optional): A function/transform that takes in the
        coarsened and pooled :obj:`torch_geometric.data.Data` object and
        returns a transformed version. (default: :obj:`None`)

:rtype: :class:`torch_geometric.data.Data`

### `max_pool_neighbor_x(data: torch_geometric.data.data.Data, flow: Optional[str] = 'source_to_target') -> torch_geometric.data.data.Data`

Max pools neighboring node features, where each feature in
:obj:`data.x` is replaced by the feature value with the maximum value from
the central node and its neighbors.

### `max_pool_x(cluster: torch.Tensor, x: torch.Tensor, batch: torch.Tensor, batch_size: Optional[int] = None, size: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`

Max-Pools node features according to the clustering defined in
:attr:`cluster`.

Args:
    cluster (torch.Tensor): The cluster vector
        :math:`\mathbf{c} \in \{ 0, \ldots, N - 1 \}^N`, which assigns each
        node to a specific cluster.
    x (Tensor): The node feature matrix.
    batch (torch.Tensor): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example.
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)
    size (int, optional): The maximum number of clusters in a single
        example. This property is useful to obtain a batch-wise dense
        representation, *e.g.* for applying FC layers, but should only be
        used if the size of the maximum number of clusters per example is
        known in advance. (default: :obj:`None`)

:rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`) if :attr:`size` is
    :obj:`None`, else :class:`torch.Tensor`

### `nearest(x: torch.Tensor, y: torch.Tensor, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None) -> torch.Tensor`

Finds for each element in :obj:`y` the :obj:`k` nearest point in
:obj:`x`.

.. code-block:: python

    import torch
    from torch_geometric.nn import nearest

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch_x = torch.tensor([0, 0, 0, 0])
    y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    batch_y = torch.tensor([0, 0])
    cluster = nearest(x, y, batch_x, batch_y)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    y (torch.Tensor): Node feature matrix
        :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
        node to a specific example. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

### `radius(x: torch.Tensor, y: torch.Tensor, r: float, batch_x: Optional[torch.Tensor] = None, batch_y: Optional[torch.Tensor] = None, max_num_neighbors: int = 32, num_workers: int = 1, batch_size: Optional[int] = None) -> torch.Tensor`

Finds for each element in :obj:`y` all points in :obj:`x` within
distance :obj:`r`.

.. code-block:: python

    import torch
    from torch_geometric.nn import radius

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch_x = torch.tensor([0, 0, 0, 0])
    y = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    batch_y = torch.tensor([0, 0])
    assign_index = radius(x, y, 1.5, batch_x, batch_y)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    y (torch.Tensor): Node feature matrix
        :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
    r (float): The radius.
    batch_x (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    batch_y (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
        node to a specific example. (default: :obj:`None`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        return for each element in :obj:`y`. (default: :obj:`32`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch_x` or :obj:`batch_y` is not
        :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

.. warning::

    The CPU implementation of :meth:`radius` with :obj:`max_num_neighbors`
    is biased towards certain quadrants.
    Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
    inputs to GPU before proceeding.

### `radius_graph(x: torch.Tensor, r: float, batch: Optional[torch.Tensor] = None, loop: bool = False, max_num_neighbors: int = 32, flow: str = 'source_to_target', num_workers: int = 1, batch_size: Optional[int] = None) -> torch.Tensor`

Computes graph edges to all points within a given distance.

.. code-block:: python

    import torch
    from torch_geometric.nn import radius_graph

    x = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    batch = torch.tensor([0, 0, 0, 0])
    edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)

Args:
    x (torch.Tensor): Node feature matrix
        :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
    r (float): The radius.
    batch (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    loop (bool, optional): If :obj:`True`, the graph will contain
        self-loops. (default: :obj:`False`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        return for each element in :obj:`y`. (default: :obj:`32`)
    flow (str, optional): The flow direction when using in combination with
        message passing (:obj:`"source_to_target"` or
        :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    num_workers (int, optional): Number of workers to use for computation.
        Has no effect in case :obj:`batch` is not :obj:`None`, or the input
        lies on the GPU. (default: :obj:`1`)
    batch_size (int, optional): The number of examples :math:`B`.
        Automatically calculated if not given. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

.. warning::

    The CPU implementation of :meth:`radius_graph` with
    :obj:`max_num_neighbors` is biased towards certain quadrants.
    Consider setting :obj:`max_num_neighbors` to :obj:`None` or moving
    inputs to GPU before proceeding.

### `summary(model: torch.nn.modules.module.Module, *args, max_depth: int = 3, leaf_module: Union[torch.nn.modules.module.Module, List[torch.nn.modules.module.Module], NoneType] = 'MessagePassing', **kwargs) -> str`

Summarizes a given :class:`torch.nn.Module`.
The summarized information includes (1) layer names, (2) input and output
shapes, and (3) the number of parameters.

.. code-block:: python

    import torch
    from torch_geometric.nn import GCN, summary

    model = GCN(128, 64, num_layers=2, out_channels=32)
    x = torch.randn(100, 128)
    edge_index = torch.randint(100, size=(2, 20))

    print(summary(model, x, edge_index))

.. code-block::

    +---------------------+---------------------+--------------+--------+
    | Layer               | Input Shape         | Output Shape | #Param |
    |---------------------+---------------------+--------------+--------|
    | GCN                 | [100, 128], [2, 20] | [100, 32]    | 10,336 |
    | ├─(act)ReLU         | [100, 64]           | [100, 64]    | --     |
    | ├─(convs)ModuleList | --                  | --           | 10,336 |
    | │    └─(0)GCNConv   | [100, 128], [2, 20] | [100, 64]    | 8,256  |
    | │    └─(1)GCNConv   | [100, 64], [2, 20]  | [100, 32]    | 2,080  |
    +---------------------+---------------------+--------------+--------+

Args:
    model (torch.nn.Module): The model to summarize.
    *args: The arguments of the :obj:`model`.
    max_depth (int, optional): The depth of nested layers to display.
        Any layers deeper than this depth will not be displayed in the
        summary. (default: :obj:`3`)
    leaf_module (torch.nn.Module or [torch.nn.Module], optional): The
        modules to be treated as leaf modules, whose submodules are
        excluded from the summary.
        (default: :class:`~torch_geometric.nn.conv.MessagePassing`)
    **kwargs: Additional arguments of the :obj:`model`.

### `to_captum_input(x: Union[torch.Tensor, Dict[str, torch.Tensor]], edge_index: Union[torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]], mask_type: Union[str, torch_geometric.explain.algorithm.captum.MaskLevelType], *args) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]`

Given :obj:`x`, :obj:`edge_index` and :obj:`mask_type`, converts it
to a format to use in `Captum <https://captum.ai/>`_ attribution
methods. Returns :obj:`inputs` and :obj:`additional_forward_args`
required for :captum:`Captum's` :obj:`attribute` functions.
See :meth:`~torch_geometric.nn.models.to_captum_model` for example usage.

Args:
    x (torch.Tensor or Dict[NodeType, torch.Tensor]): The node features.
        For heterogeneous graphs this is a dictionary holding node featues
        for each node type.
    edge_index(torch.Tensor or Dict[EdgeType, torch.Tensor]): The edge
        indices. For heterogeneous graphs this is a dictionary holding the
        :obj:`edge index` for each edge type.
    mask_type (str): Denotes the type of mask to be created with
        a Captum explainer. Valid inputs are :obj:`"edge"`, :obj:`"node"`,
        and :obj:`"node_and_edge"`.
    *args: Additional forward arguments of the model being explained
        which will be added to :obj:`additional_forward_args`.

### `to_captum_model(model: torch.nn.modules.module.Module, mask_type: Union[str, torch_geometric.explain.algorithm.captum.MaskLevelType] = <MaskLevelType.edge: 'edge'>, output_idx: Optional[int] = None, metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None) -> Union[torch_geometric.explain.algorithm.captum.CaptumModel, torch_geometric.explain.algorithm.captum.CaptumHeteroModel]`

Converts a model to a model that can be used for
`Captum <https://captum.ai/>`_ attribution methods.

Sample code for homogeneous graphs:

.. code-block:: python

    from captum.attr import IntegratedGradients

    from torch_geometric.data import Data
    from torch_geometric.nn import GCN
    from torch_geometric.nn import to_captum_model, to_captum_input

    data = Data(x=(...), edge_index(...))
    model = GCN(...)
    ...  # Train the model.

    # Explain predictions for node `10`:
    mask_type="edge"
    output_idx = 10
    captum_model = to_captum_model(model, mask_type, output_idx)
    inputs, additional_forward_args = to_captum_input(data.x,
                                        data.edge_index,mask_type)

    ig = IntegratedGradients(captum_model)
    ig_attr = ig.attribute(inputs = inputs,
                           target=int(y[output_idx]),
                           additional_forward_args=additional_forward_args,
                           internal_batch_size=1)


Sample code for heterogeneous graphs:

.. code-block:: python

    from captum.attr import IntegratedGradients

    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv
    from torch_geometric.nn import (captum_output_to_dicts,
                                    to_captum_model, to_captum_input)

    data = HeteroData(...)
    model = HeteroConv(...)
    ...  # Train the model.

    # Explain predictions for node `10`:
    mask_type="edge"
    metadata = data.metadata
    output_idx = 10
    captum_model = to_captum_model(model, mask_type, output_idx, metadata)
    inputs, additional_forward_args = to_captum_input(data.x_dict,
                                        data.edge_index_dict, mask_type)

    ig = IntegratedGradients(captum_model)
    ig_attr = ig.attribute(inputs=inputs,
                           target=int(y[output_idx]),
                           additional_forward_args=additional_forward_args,
                           internal_batch_size=1)
    edge_attr_dict = captum_output_to_dicts(ig_attr, mask_type, metadata)


.. note::
    For an example of using a :captum:`Captum` attribution method within
    :pyg:`PyG`, see `examples/explain/captum_explainer.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    explain/captum_explainer.py>`_.

Args:
    model (torch.nn.Module): The model to be explained.
    mask_type (str, optional): Denotes the type of mask to be created with
        a :captum:`Captum` explainer. Valid inputs are :obj:`"edge"`,
        :obj:`"node"`, and :obj:`"node_and_edge"`. (default: :obj:`"edge"`)
    output_idx (int, optional): Index of the output element (node or link
        index) to be explained. With :obj:`output_idx` set, the forward
        function will return the output of the model for the element at
        the index specified. (default: :obj:`None`)
    metadata (Metadata, optional): The metadata of the heterogeneous graph.
        Only required if explaning a
        :class:`~torch_geometric.data.HeteroData` object.
        (default: :obj:`None`)

### `to_fixed_size(module: torch.nn.modules.module.Module, batch_size: int, debug: bool = False) -> torch.fx.graph_module.GraphModule`

Converts a model and injects a pre-computed and fixed batch size to all
global pooling operators.

Args:
    module (torch.nn.Module): The model to transform.
    batch_size (int): The fixed batch size used in global pooling modules.
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

### `to_hetero(module: torch.nn.modules.module.Module, metadata: Tuple[List[str], List[Tuple[str, str, str]]], aggr: str = 'sum', input_map: Optional[Dict[str, str]] = None, debug: bool = False) -> torch.fx.graph_module.GraphModule`

Converts a homogeneous GNN model into its heterogeneous equivalent in
which node representations are learned for each node type in
:obj:`metadata[0]`, and messages are exchanged between each edge type in
:obj:`metadata[1]`, as denoted in the `"Modeling Relational Data with Graph
Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ paper.

.. code-block:: python

    import torch
    from torch_geometric.nn import SAGEConv, to_hetero

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), 32)
            self.conv2 = SAGEConv((32, 32), 32)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return x

    model = GNN()

    node_types = ['paper', 'author']
    edge_types = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
    ]
    metadata = (node_types, edge_types)

    model = to_hetero(model, metadata)
    model(x_dict, edge_index_dict)

where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
hold node features and edge connectivity information for each node type and
edge type, respectively.

The below illustration shows the original computation graph of the
homogeneous model on the left, and the newly obtained computation graph of
the heterogeneous model on the right:

.. figure:: ../_figures/to_hetero.svg
  :align: center
  :width: 90%

  Transforming a model via :func:`to_hetero`.

Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
:math:`f_{\theta}^{(\ell)}` is duplicated and stored in a set
:math:`\{ f_{\theta}^{(\ell, r)} : r \in \mathcal{R} \}` (one instance for
each relation in :math:`\mathcal{R}`), and message passing in layer
:math:`\ell` is performed via

.. math::

    \mathbf{h}^{(\ell)}_v = \bigoplus_{r \in \mathcal{R}}
    f_{\theta}^{(\ell, r)} ( \mathbf{h}^{(\ell - 1)}_v, \{
    \mathbf{h}^{(\ell - 1)}_w : w \in \mathcal{N}^{(r)}(v) \}),

where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
\mathcal{V}` under relation :math:`r \in \mathcal{R}`, and
:math:`\bigoplus` denotes the aggregation scheme :attr:`aggr` to use for
grouping node embeddings generated by different relations
(:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`).

Args:
    module (torch.nn.Module): The homogeneous model to transform.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    aggr (str, optional): The aggregation scheme to use for grouping node
        embeddings generated by different relations
        (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"mul"`). (default: :obj:`"sum"`)
    input_map (Dict[str, str], optional): A dictionary holding information
        about the type of input arguments of :obj:`module.forward`.
        For example, in case :obj:`arg` is a node-level argument, then
        :obj:`input_map['arg'] = 'node'`, and
        :obj:`input_map['arg'] = 'edge'` otherwise.
        In case :obj:`input_map` is not further specified, will try to
        automatically determine the correct type of input arguments.
        (default: :obj:`None`)
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

### `to_hetero_with_bases(module: torch.nn.modules.module.Module, metadata: Tuple[List[str], List[Tuple[str, str, str]]], num_bases: int, in_channels: Optional[Dict[str, int]] = None, input_map: Optional[Dict[str, str]] = None, debug: bool = False) -> torch.fx.graph_module.GraphModule`

Converts a homogeneous GNN model into its heterogeneous equivalent
via the basis-decomposition technique introduced in the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

For this, the heterogeneous graph is mapped to a typed homogeneous graph,
in which its feature representations are aligned and grouped to a single
representation.
All GNN layers inside the model will then perform message passing via
basis-decomposition regularization.
This transformation is especially useful in highly multi-relational data,
such that the number of parameters no longer depend on the number of
relations of the input graph:

.. code-block:: python

    import torch
    from torch_geometric.nn import SAGEConv, to_hetero_with_bases

    class GNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv((16, 16), 32)
            self.conv2 = SAGEConv((32, 32), 32)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return x

    model = GNN()

    node_types = ['paper', 'author']
    edge_types = [
        ('paper', 'cites', 'paper'),
        ('paper', 'written_by', 'author'),
        ('author', 'writes', 'paper'),
    ]
    metadata = (node_types, edge_types)

    model = to_hetero_with_bases(model, metadata, num_bases=3,
                                 in_channels={'x': 16})
    model(x_dict, edge_index_dict)

where :obj:`x_dict` and :obj:`edge_index_dict` denote dictionaries that
hold node features and edge connectivity information for each node type and
edge type, respectively.
In case :obj:`in_channels` is given for a specific input argument, its
heterogeneous feature information is first aligned to the given
dimensionality.

The below illustration shows the original computation graph of the
homogeneous model on the left, and the newly obtained computation graph of
the regularized heterogeneous model on the right:

.. figure:: ../_figures/to_hetero_with_bases.svg
  :align: center
  :width: 90%

  Transforming a model via :func:`to_hetero_with_bases`.

Here, each :class:`~torch_geometric.nn.conv.MessagePassing` instance
:math:`f_{\theta}^{(\ell)}` is duplicated :obj:`num_bases` times and
stored in a set :math:`\{ f_{\theta}^{(\ell, b)} : b \in \{ 1, \ldots, B \}
\}` (one instance for each basis in
:obj:`num_bases`), and message passing in layer :math:`\ell` is performed
via

.. math::

    \mathbf{h}^{(\ell)}_v = \sum_{r \in \mathcal{R}} \sum_{b=1}^B
    f_{\theta}^{(\ell, b)} ( \mathbf{h}^{(\ell - 1)}_v, \{
    a^{(\ell)}_{r, b} \cdot \mathbf{h}^{(\ell - 1)}_w :
    w \in \mathcal{N}^{(r)}(v) \}),

where :math:`\mathcal{N}^{(r)}(v)` denotes the neighborhood of :math:`v \in
\mathcal{V}` under relation :math:`r \in \mathcal{R}`.
Notably, only the trainable basis coefficients :math:`a^{(\ell)}_{r, b}`
depend on the relations in :math:`\mathcal{R}`.

Args:
    module (torch.nn.Module): The homogeneous model to transform.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    num_bases (int): The number of bases to use.
    in_channels (Dict[str, int], optional): A dictionary holding
        information about the desired input feature dimensionality of
        input arguments of :obj:`module.forward`.
        In case :obj:`in_channels` is given for a specific input argument,
        its heterogeneous feature information is first aligned to the given
        dimensionality.
        This allows handling of node and edge features with varying feature
        dimensionality across different types. (default: :obj:`None`)
    input_map (Dict[str, str], optional): A dictionary holding information
        about the type of input arguments of :obj:`module.forward`.
        For example, in case :obj:`arg` is a node-level argument, then
        :obj:`input_map['arg'] = 'node'`, and
        :obj:`input_map['arg'] = 'edge'` otherwise.
        In case :obj:`input_map` is not further specified, will try to
        automatically determine the correct type of input arguments.
        (default: :obj:`None`)
    debug (bool, optional): If set to :obj:`True`, will perform
        transformation in debug mode. (default: :obj:`False`)

### `voxel_grid(pos: torch.Tensor, size: Union[float, List[float], torch.Tensor], batch: Optional[torch.Tensor] = None, start: Union[float, List[float], torch.Tensor, NoneType] = None, end: Union[float, List[float], torch.Tensor, NoneType] = None) -> torch.Tensor`

Voxel grid pooling from the, *e.g.*, `Dynamic Edge-Conditioned Filters
in Convolutional Networks on Graphs <https://arxiv.org/abs/1704.02901>`_
paper, which overlays a regular grid of user-defined size over a point
cloud and clusters all points within the same voxel.

Args:
    pos (torch.Tensor): Node position matrix
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times D}`.
    size (float or [float] or Tensor): Size of a voxel (in each dimension).
    batch (torch.Tensor, optional): Batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N`, which assigns each
        node to a specific example. (default: :obj:`None`)
    start (float or [float] or Tensor, optional): Start coordinates of the
        grid (in each dimension). If set to :obj:`None`, will be set to the
        minimum coordinates found in :attr:`pos`. (default: :obj:`None`)
    end (float or [float] or Tensor, optional): End coordinates of the grid
        (in each dimension). If set to :obj:`None`, will be set to the
        maximum coordinates found in :attr:`pos`. (default: :obj:`None`)

:rtype: :class:`torch.Tensor`

## Important Data Types (15)

### `GAE`
**Type**: `<class 'type'>`

The Graph Auto-Encoder model from the
`"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
paper based on user-defined encoder and decoder models.

Args:
    encoder (torch.nn.Module): The encoder module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

*(has methods, callable)*

### `GAT`
**Type**: `<class 'type'>`

The Graph Neural Network from `"Graph Attention Networks"
<https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
:class:`~torch_geometric.nn.GATConv` or
:class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
respectively.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    v2 (bool, optional): If set to :obj:`True`, will make use of
        :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
        :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GATConv` or
        :class:`torch_geometric.nn.conv.GATv2Conv`.

*(has methods, callable)*

### `GCN`
**Type**: `<class 'type'>`

The Graph Neural Network from the `"Semi-supervised
Classification with Graph Convolutional Networks"
<https://arxiv.org/abs/1609.02907>`_ paper, using the
:class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GCNConv`.

*(has methods, callable)*

### `GIN`
**Type**: `<class 'type'>`

The Graph Neural Network from the `"How Powerful are Graph Neural
Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
:class:`~torch_geometric.nn.GINConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GINConv`.

*(has methods, callable)*

### `MLP`
**Type**: `<class 'type'>`

A Multi-Layer Perception (MLP) model.

There exists two ways to instantiate an :class:`MLP`:

1. By specifying explicit channel sizes, *e.g.*,

   .. code-block:: python

      mlp = MLP([16, 32, 64, 128])

   creates a three-layer MLP with **differently** sized hidden layers.

1. By specifying fixed hidden channel sizes over a number of layers,
   *e.g.*,

   .. code-block:: python

      mlp = MLP(in_channels=16, hidden_channels=32,
                out_channels=128, num_layers=3)

   creates a three-layer MLP with **equally** sized hidden layers.

Args:
    channel_list (List[int] or int, optional): List of input, intermediate
        and output channels such that :obj:`len(channel_list) - 1` denotes
        the number of layers of the MLP (default: :obj:`None`)
    in_channels (int, optional): Size of each input sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    hidden_channels (int, optional): Size of each hidden sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    out_channels (int, optional): Size of each output sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    num_layers (int, optional): The number of layers.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    dropout (float or List[float], optional): Dropout probability of each
        hidden embedding. If a list is provided, sets the dropout value per
        layer. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`"batch_norm"`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    plain_last (bool, optional): If set to :obj:`False`, will apply
        non-linearity, batch normalization and dropout to the last layer as
        well. (default: :obj:`True`)
    bias (bool or List[bool], optional): If set to :obj:`False`, the module
        will not learn additive biases. If a list is provided, sets the
        bias per layer. (default: :obj:`True`)
    **kwargs (optional): Additional deprecated arguments of the MLP layer.

*(has methods, callable)*

### `PNA`
**Type**: `<class 'type'>`

The Graph Neural Network from the `"Principal Neighbourhood Aggregation
for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
:class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.PNAConv`.

*(has methods, callable)*

### `ARGA`
**Type**: `<class 'type'>`

The Adversarially Regularized Graph Auto-Encoder model from the
`"Adversarially Regularized Graph Autoencoder for Graph Embedding"
<https://arxiv.org/abs/1802.04407>`_ paper.

Args:
    encoder (torch.nn.Module): The encoder module.
    discriminator (torch.nn.Module): The discriminator module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

*(has methods, callable)*

### `PMLP`
**Type**: `<class 'type'>`

The P(ropagational)MLP model from the `"Graph Neural Networks are
Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
<https://arxiv.org/abs/2212.09034>`_ paper.
:class:`PMLP` is identical to a standard MLP during training, but then
adopts a GNN architecture during testing.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output sample.
    num_layers (int): The number of layers.
    dropout (float, optional): Dropout probability of each hidden
        embedding. (default: :obj:`0.`)
    norm (bool, optional): If set to :obj:`False`, will not apply batch
        normalization. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the module
        will not learn additive biases. (default: :obj:`True`)

*(has methods, callable)*

### `VGAE`
**Type**: `<class 'type'>`

The Variational Graph Auto-Encoder model from the
`"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
paper.

Args:
    encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
        and :math:`\log\sigma^2`.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

*(has methods, callable)*

### `APPNP`
**Type**: `<class 'type'>`

The approximate personalized propagation of neural predictions layer
from the `"Predict then Propagate: Graph Neural Networks meet Personalized
PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.

.. math::
    \mathbf{X}^{(0)} &= \mathbf{X}

    \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
    \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
    \mathbf{X}^{(0)}

    \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Args:
    K (int): Number of iterations :math:`K`.
    alpha (float): Teleport probability :math:`\alpha`.
    dropout (float, optional): Dropout probability of edges during
        training. (default: :obj:`0`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    normalize (bool, optional): Whether to add self-loops and apply
        symmetric normalization. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)`

*(has methods, callable)*

### `ARGVA`
**Type**: `<class 'type'>`

The Adversarially Regularized Variational Graph Auto-Encoder model from
the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
<https://arxiv.org/abs/1802.04407>`_ paper.

Args:
    encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
        and :math:`\log\sigma^2`.
    discriminator (torch.nn.Module): The discriminator module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

*(has methods, callable)*

### `GNNFF`
**Type**: `<class 'type'>`

The Graph Neural Network Force Field (GNNFF) from the
`"Accurate and scalable graph neural network force field and molecular
dynamics with direct force architecture"
<https://www.nature.com/articles/s41524-021-00543-3>`_ paper.
:class:`GNNFF` directly predicts atomic forces from automatically
extracted features of the local atomic environment that are
translationally-invariant, but rotationally-covariant to the coordinate of
the atoms.

Args:
    hidden_node_channels (int): Hidden node embedding size.
    hidden_edge_channels (int): Hidden edge embedding size.
    num_layers (int): Number of message passing blocks.
    cutoff (float, optional): Cutoff distance for interatomic
        interactions. (default: :obj:`5.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        collect for each node within the :attr:`cutoff` distance.
        (default: :obj:`32`)

*(has methods, callable)*

### `LINKX`
**Type**: `<class 'type'>`

The LINKX model from the `"Large Scale Learning on Non-Homophilous
Graphs: New Benchmarks and Strong Simple Methods"
<https://arxiv.org/abs/2110.14446>`_ paper.

.. math::
    \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

    \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

    \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
    [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
    \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

.. note::

    For an example of using LINKX, see `examples/linkx.py <https://
    github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

Args:
    num_nodes (int): The number of nodes in the graph.
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output sample.
    num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
    num_edge_layers (int, optional): Number of layers of
        :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
    num_node_layers (int, optional): Number of layers of
        :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
    dropout (float, optional): Dropout probability of each hidden
        embedding. (default: :obj:`0.0`)

*(has methods, callable)*

### `RENet`
**Type**: `<class 'type'>`

The Recurrent Event Network model from the `"Recurrent Event Network
for Reasoning over Temporal Knowledge Graphs"
<https://arxiv.org/abs/1904.05530>`_ paper.

.. math::
    f_{\mathbf{\Theta}}(\mathbf{e}_s, \mathbf{e}_r,
    \mathbf{h}^{(t-1)}(s, r))

based on a RNN encoder

.. math::
    \mathbf{h}^{(t)}(s, r) = \textrm{RNN}(\mathbf{e}_s, \mathbf{e}_r,
    g(\mathcal{O}^{(t)}_r(s)), \mathbf{h}^{(t-1)}(s, r))

where :math:`\mathbf{e}_s` and :math:`\mathbf{e}_r` denote entity and
relation embeddings, and :math:`\mathcal{O}^{(t)}_r(s)` represents the set
of objects interacted with subject :math:`s` under relation :math:`r` at
timestamp :math:`t`.
This model implements :math:`g` as the **Mean Aggregator** and
:math:`f_{\mathbf{\Theta}}` as a linear projection.

Args:
    num_nodes (int): The number of nodes in the knowledge graph.
    num_rels (int): The number of relations in the knowledge graph.
    hidden_channels (int): Hidden size of node and relation embeddings.
    seq_len (int): The sequence length of past events.
    num_layers (int, optional): The number of recurrent layers.
        (default: :obj:`1`)
    dropout (float): If non-zero, introduces a dropout layer before the
        final prediction. (default: :obj:`0.`)
    bias (bool, optional): If set to :obj:`False`, all layers will not
        learn an additive bias. (default: :obj:`True`)

*(has methods, callable)*

### `XConv`
**Type**: `<class 'type'>`

The convolutional operator on :math:`\mathcal{X}`-transformed points
from the `"PointCNN: Convolution On X-Transformed Points"
<https://arxiv.org/abs/1801.07791>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathrm{Conv}\left(\mathbf{K},
    \gamma_{\mathbf{\Theta}}(\mathbf{P}_i - \mathbf{p}_i) \times
    \left( h_\mathbf{\Theta}(\mathbf{P}_i - \mathbf{p}_i) \, \Vert \,
    \mathbf{x}_i \right) \right),

where :math:`\mathbf{K}` and :math:`\mathbf{P}_i` denote the trainable
filter and neighboring point positions of :math:`\mathbf{x}_i`,
respectively.
:math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}` describe
neural networks, *i.e.* MLPs, where :math:`h_{\mathbf{\Theta}}`
individually lifts each point into a higher-dimensional space, and
:math:`\gamma_{\mathbf{\Theta}}` computes the :math:`\mathcal{X}`-
transformation matrix based on *all* points in a neighborhood.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    dim (int): Point cloud dimensionality.
    kernel_size (int): Size of the convolving kernel, *i.e.* number of
        neighbors including self-loops.
    hidden_channels (int, optional): Output size of
        :math:`h_{\mathbf{\Theta}}`, *i.e.* dimensionality of lifted
        points. If set to :obj:`None`, will be automatically set to
        :obj:`in_channels / 4`. (default: :obj:`None`)
    dilation (int, optional): The factor by which the neighborhood is
        extended, from which :obj:`kernel_size` neighbors are then
        uniformly sampled. Can be interpreted as the dilation rate of
        classical convolutional operators. (default: :obj:`1`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    num_workers (int): Number of workers to use for k-NN computation.
        Has no effect in case :obj:`batch` is not :obj:`None`, or the input
        lies on the GPU. (default: :obj:`1`)

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      positions :math:`(|\mathcal{V}|, D)`,
      batch vector :math:`(|\mathcal{V}|)` *(optional)*
    - **output:**
      node features :math:`(|\mathcal{V}|, F_{out})`

*(has methods, callable)*

## Classes (175)

### `AGNNConv`

The graph attentional propagation layer from the
`"Attention-based Graph Neural Network for Semi-Supervised Learning"
<https://arxiv.org/abs/1803.03735>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \mathbf{P} \mathbf{X},

where the propagation matrix :math:`\mathbf{P}` is computed as

.. math::
    P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
    {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
    \cos(\mathbf{x}_i, \mathbf{x}_k))}

with trainable parameter :math:`\beta`.

Args:
    requires_grad (bool, optional): If set to :obj:`False`, :math:`\beta`
        will not be trainable. (default: :obj:`True`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F)`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, x_norm_i: torch.Tensor, x_norm_j: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `APPNP`

The approximate personalized propagation of neural predictions layer
from the `"Predict then Propagate: Graph Neural Networks meet Personalized
PageRank" <https://arxiv.org/abs/1810.05997>`_ paper.

.. math::
    \mathbf{X}^{(0)} &= \mathbf{X}

    \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
    \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
    \mathbf{X}^{(0)}

    \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Args:
    K (int): Number of iterations :math:`K`.
    alpha (float): Teleport probability :math:`\alpha`.
    dropout (float, optional): Dropout probability of edges during
        training. (default: :obj:`0`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    normalize (bool, optional): Whether to add self-loops and apply
        symmetric normalization. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `ARGA`

The Adversarially Regularized Graph Auto-Encoder model from the
`"Adversarially Regularized Graph Autoencoder for Graph Embedding"
<https://arxiv.org/abs/1802.04407>`_ paper.

Args:
    encoder (torch.nn.Module): The encoder module.
    discriminator (torch.nn.Module): The discriminator module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`reg_loss(self, z: torch.Tensor) -> torch.Tensor`**
  Computes the regularization loss of the encoder.

- **`discriminator_loss(self, z: torch.Tensor) -> torch.Tensor`**
  Computes the loss of the discriminator.

### `ARGVA`

The Adversarially Regularized Variational Graph Auto-Encoder model from
the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
<https://arxiv.org/abs/1802.04407>`_ paper.

Args:
    encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
        and :math:`\log\sigma^2`.
    discriminator (torch.nn.Module): The discriminator module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

#### Methods

- **`reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor`**

- **`encode(self, *args, **kwargs) -> torch.Tensor`**

- **`kl_loss(self, mu: Optional[torch.Tensor] = None, logstd: Optional[torch.Tensor] = None) -> torch.Tensor`**

### `ARMAConv`

The ARMA graph convolutional operator from the `"Graph Neural Networks
with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_
paper.

.. math::
    \mathbf{X}^{\prime} = \frac{1}{K} \sum_{k=1}^K \mathbf{X}_k^{(T)},

with :math:`\mathbf{X}_k^{(T)}` being recursively defined by

.. math::
    \mathbf{X}_k^{(t+1)} = \sigma \left( \mathbf{\hat{L}}
    \mathbf{X}_k^{(t)} \mathbf{W} + \mathbf{X}^{(0)} \mathbf{V} \right),

where :math:`\mathbf{\hat{L}} = \mathbf{I} - \mathbf{L} = \mathbf{D}^{-1/2}
\mathbf{A} \mathbf{D}^{-1/2}` denotes the
modified Laplacian :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}
\mathbf{A} \mathbf{D}^{-1/2}`.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample
        :math:`\mathbf{x}^{(t+1)}`.
    num_stacks (int, optional): Number of parallel stacks :math:`K`.
        (default: :obj:`1`).
    num_layers (int, optional): Number of layers :math:`T`.
        (default: :obj:`1`)
    act (callable, optional): Activation function :math:`\sigma`.
        (default: :meth:`torch.nn.ReLU()`)
    shared_weights (int, optional): If set to :obj:`True` the layers in
        each stack will share the same parameters. (default: :obj:`False`)
    dropout (float, optional): Dropout probability of the skip connection.
        (default: :obj:`0.`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

- **`initialize_parameters(self, module, input)`**

### `ASAPooling`

The Adaptive Structure Aware Pooling operator from the
`"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

Args:
    in_channels (int): Size of each input sample.
    ratio (float or int): Graph pooling ratio, which is used to compute
        :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
        of :math:`k` itself, depending on whether the type of :obj:`ratio`
        is :obj:`float` or :obj:`int`. (default: :obj:`0.5`)
    GNN (torch.nn.Module, optional): A graph neural network layer for
        using intra-cluster properties.
        Especially helpful for graphs with higher degree of neighborhood
        (one of :class:`torch_geometric.nn.conv.GraphConv`,
        :class:`torch_geometric.nn.conv.GCNConv` or
        any GNN which supports the :obj:`edge_weight` parameter).
        (default: :obj:`None`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    add_self_loops (bool, optional): If set to :obj:`True`, will add self
        loops to the new graph connectivity. (default: :obj:`False`)
    **kwargs (optional): Additional parameters for initializing the
        graph neural network layer.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]`**
  Forward pass.

### `Aggregation`

An abstract base class for implementing custom aggregations.

Aggregation can be either performed via an :obj:`index` vector, which
defines the mapping from input elements to their location in the output:

|

.. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
        master/docs/source/_figures/add.svg?sanitize=true
    :align: center
    :width: 400px

|

Notably, :obj:`index` does not have to be sorted (for most aggregation
operators):

.. code-block:: python

   # Feature matrix holding 10 elements with 64 features each:
   x = torch.randn(10, 64)

   # Assign each element to one of three sets:
   index = torch.tensor([0, 0, 1, 0, 2, 0, 2, 1, 0, 2])

   output = aggr(x, index)  #  Output shape: [3, 64]

Alternatively, aggregation can be achieved via a "compressed" index vector
called :obj:`ptr`. Here, elements within the same set need to be grouped
together in the input, and :obj:`ptr` defines their boundaries:

.. code-block:: python

   # Feature matrix holding 10 elements with 64 features each:
   x = torch.randn(10, 64)

   # Define the boundary indices for three sets:
   ptr = torch.tensor([0, 4, 7, 10])

   output = aggr(x, ptr=ptr)  #  Output shape: [3, 64]

Note that at least one of :obj:`index` or :obj:`ptr` must be defined.

Shapes:
    - **input:**
      node features :math:`(*, |\mathcal{V}|, F_{in})` or edge features
      :math:`(*, |\mathcal{E}|, F_{in})`,
      index vector :math:`(|\mathcal{V}|)` or :math:`(|\mathcal{E}|)`,
    - **output:** graph features :math:`(*, |\mathcal{G}|, F_{out})` or
      node features :math:`(*, |\mathcal{V}|, F_{out})`

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`assert_index_present(self, index: Optional[torch.Tensor])`**

- **`assert_sorted_index(self, index: Optional[torch.Tensor])`**

- **`assert_two_dimensional_input(self, x: torch.Tensor, dim: int)`**

### `AntiSymmetricConv`

The anti-symmetric graph convolutional operator from the
`"Anti-Symmetric DGN: a stable architecture for Deep Graph Networks"
<https://openreview.net/forum?id=J3Y7cgZOOS>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{x}_i + \epsilon \cdot \sigma \left(
        (\mathbf{W}-\mathbf{W}^T-\gamma \mathbf{I}) \mathbf{x}_i +
        \Phi(\mathbf{X}, \mathcal{N}_i) + \mathbf{b}\right),

where :math:`\Phi(\mathbf{X}, \mathcal{N}_i)` denotes a
:class:`~torch.nn.conv.MessagePassing` layer.

Args:
    in_channels (int): Size of each input sample.
    phi (MessagePassing, optional): The message passing module
        :math:`\Phi`. If set to :obj:`None`, will use a
        :class:`~torch_geometric.nn.conv.GCNConv` layer as default.
        (default: :obj:`None`)
    num_iters (int, optional): The number of times the anti-symmetric deep
        graph network operator is called. (default: :obj:`1`)
    epsilon (float, optional): The discretization step size
        :math:`\epsilon`. (default: :obj:`0.1`)
    gamma (float, optional): The strength of the diffusion :math:`\gamma`.
        It regulates the stability of the method. (default: :obj:`0.1`)
    act (str, optional): The non-linear activation function :math:`\sigma`,
        *e.g.*, :obj:`"tanh"` or :obj:`"relu"`. (default: :class:`"tanh"`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{in})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], *args, **kwargs) -> torch.Tensor`**
  Runs the forward pass of the module.

### `ApproxL2KNNIndex`

Performs fast approximate :math:`k`-nearest neighbor search
(:math:`k`-NN) based on the the :math:`L_2` metric via the :obj:`faiss`
library.
Hyperparameters needs to be tuned for speed-accuracy trade-off.

Args:
    num_cells (int): The number of cells.
    num_cells_to_visit (int): The number of cells that are visited to
        perform to search.
    bits_per_vector (int): The number of bits per sub-vector.
    emb (torch.Tensor, optional): The data points to add.
        (default: :obj:`None`)
    reserve (int, optional): The number of elements to reserve memory for
        before re-allocating (GPU only). (default: :obj:`None`)

### `ApproxMIPSKNNIndex`

Performs fast approximate :math:`k`-nearest neighbor search
(:math:`k`-NN) based on the maximum inner product via the :obj:`faiss`
library.
Hyperparameters needs to be tuned for speed-accuracy trade-off.

Args:
    num_cells (int): The number of cells.
    num_cells_to_visit (int): The number of cells that are visited to
        perform to search.
    bits_per_vector (int): The number of bits per sub-vector.
    emb (torch.Tensor, optional): The data points to add.
        (default: :obj:`None`)
    reserve (int, optional): The number of elements to reserve memory for
        before re-allocating (GPU only). (default: :obj:`None`)

### `AttentionalAggregation`

The soft attention aggregation layer from the `"Graph Matching Networks
for Learning the Similarity of Graph Structured Objects"
<https://arxiv.org/abs/1904.12787>`_ paper.

.. math::
    \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
    h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \cdot
    h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
\mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
MLPs.

Args:
    gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
        that computes attention scores by mapping node features :obj:`x` of
        shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
        node-level gating) or :obj:`[1, out_channels]` (for feature-level
        gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
    nn (torch.nn.Module, optional): A neural network
        :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
        shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
        before combining them with the attention scores, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `AttentiveFP`

The Attentive FP model for molecular representation learning from the
`"Pushing the Boundaries of Molecular Representation for Drug Discovery
with the Graph Attention Mechanism"
<https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
graph attention mechanisms.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Hidden node feature dimensionality.
    out_channels (int): Size of each output sample.
    edge_dim (int): Edge feature dimensionality.
    num_layers (int): Number of GNN layers.
    num_timesteps (int): Number of iterative refinement steps for global
        readout.
    dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor`**

### `BatchNorm`

Applies batch normalization over a batch of features as described in
the `"Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
    \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
    \odot \gamma + \beta

The mean and standard-deviation are calculated per-dimension over all nodes
inside the mini-batch.

Args:
    in_channels (int): Size of each input sample.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    momentum (float, optional): The value used for the running mean and
        running variance computation. (default: :obj:`0.1`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`True`)
    track_running_stats (bool, optional): If set to :obj:`True`, this
        module tracks the running mean and variance, and when set to
        :obj:`False`, this module does not track such statistics and always
        uses batch statistics in both training and eval modes.
        (default: :obj:`True`)
    allow_single_element (bool, optional): If set to :obj:`True`, batches
        with only a single element will work as during in evaluation.
        That is the running mean and variance will be used.
        Requires :obj:`track_running_stats=True`. (default: :obj:`False`)

#### Methods

- **`reset_running_stats(self)`**
  Resets all running statistics of the module.

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**
  Forward pass.

### `CGConv`

The crystal graph convolutional operator from the
`"Crystal Graph Convolutional Neural Networks for an
Accurate and Interpretable Prediction of Material Properties"
<https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)}
    \sigma \left( \mathbf{z}_{i,j} \mathbf{W}_f + \mathbf{b}_f \right)
    \odot g \left( \mathbf{z}_{i,j} \mathbf{W}_s + \mathbf{b}_s  \right)

where :math:`\mathbf{z}_{i,j} = [ \mathbf{x}_i, \mathbf{x}_j,
\mathbf{e}_{i,j} ]` denotes the concatenation of central node features,
neighboring node features and edge features.
In addition, :math:`\sigma` and :math:`g` denote the sigmoid and softplus
functions, respectively.

Args:
    channels (int or tuple): Size of each input sample. A tuple
        corresponds to the sizes of source and target dimensionalities.
    dim (int, optional): Edge feature dimensionality. (default: :obj:`0`)
    aggr (str, optional): The aggregation operator to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"add"`)
    batch_norm (bool, optional): If set to :obj:`True`, will make use of
        batch normalization. (default: :obj:`False`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)` or
      :math:`(|\mathcal{V_t}|, F_{t})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i, x_j, edge_attr: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `ChebConv`

The chebyshev spectral graph convolutional operator from the
`"Convolutional Neural Networks on Graphs with Fast Localized Spectral
Filtering" <https://arxiv.org/abs/1606.09375>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
    \mathbf{\Theta}^{(k)}

where :math:`\mathbf{Z}^{(k)}` is computed recursively by

.. math::
    \mathbf{Z}^{(1)} &= \mathbf{X}

    \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}

    \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
    \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}

and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
:math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    K (int): Chebyshev filter size :math:`K`.
    normalization (str, optional): The normalization scheme for the graph
        Laplacian (default: :obj:`"sym"`):

        1. :obj:`None`: No normalization
        :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

        2. :obj:`"sym"`: Symmetric normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}`

        3. :obj:`"rw"`: Random-walk normalization
        :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`

        :obj:`\lambda_max` should be a :class:`torch.Tensor` of size
        :obj:`[num_graphs]` in a mini-batch scenario and a
        scalar/zero-dimensional tensor when operating on single graphs.
        You can pre-compute :obj:`lambda_max` via the
        :class:`torch_geometric.transforms.LaplacianLambdaMax` transform.
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*,
      batch vector :math:`(|\mathcal{V}|)` *(optional)*,
      maximum :obj:`lambda` value :math:`(|\mathcal{G}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None, lambda_max: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `ClusterGCNConv`

The ClusterGCN graph convolutional operator from the
`"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph
Convolutional Networks" <https://arxiv.org/abs/1905.07953>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \left( \mathbf{\hat{A}} + \lambda \cdot
    \textrm{diag}(\mathbf{\hat{A}}) \right) \mathbf{X} \mathbf{W}_1 +
    \mathbf{X} \mathbf{W}_2

where :math:`\mathbf{\hat{A}} = {(\mathbf{D} + \mathbf{I})}^{-1}(\mathbf{A}
+ \mathbf{I})`.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    diag_lambda (float, optional): Diagonal enhancement value
        :math:`\lambda`. (default: :obj:`0.`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `ClusterPooling`

The cluster pooling operator from the `"Edge-Based Graph Component
Pooling" <paper url>`_ paper.

:class:`ClusterPooling` computes a score for each edge.
Based on the selected edges, graph clusters are calculated and compressed
to one node using the injective :obj:`"sum"` aggregation function.
Edges are remapped based on the nodes created by each cluster and the
original edges.

Args:
    in_channels (int): Size of each input sample.
    edge_score_method (str, optional): The function to apply
        to compute the edge score from raw edge scores (:obj:`"tanh"`,
        :obj:`"sigmoid"`, :obj:`"log_softmax"`). (default: :obj:`"tanh"`)
    dropout (float, optional): The probability with
        which to drop edge scores during training. (default: :obj:`0.0`)
    threshold (float, optional): The threshold of edge scores. If set to
        :obj:`None`, will be automatically inferred depending on
        :obj:`edge_score_method`. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch_geometric.nn.pool.cluster_pool.UnpoolInfo]`**
  Forward pass.

### `ComplEx`

The ComplEx model from the `"Complex Embeddings for Simple Link
Prediction" <https://arxiv.org/abs/1606.06357>`_ paper.

:class:`ComplEx` models relations as complex-valued bilinear mappings
between head and tail entities using the Hermetian dot product.
The entities and relations are embedded in different dimensional spaces,
resulting in the scoring function:

.. math::
    d(h, r, t) = Re(< \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t>)

.. note::

    For an example of using the :class:`ComplEx` model, see
    `examples/kge_fb15k_237.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    kge_fb15k_237.py>`_.

Args:
    num_nodes (int): The number of nodes/entities in the graph.
    num_relations (int): The number of relations in the graph.
    hidden_channels (int): The hidden embedding size.
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
        the embedding matrices will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the score for the given triplet.

- **`loss(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the loss value for the given triplet.

### `CorrectAndSmooth`

The correct and smooth (C&S) post-processing model from the
`"Combining Label Propagation And Simple Models Out-performs Graph Neural
Networks"
<https://arxiv.org/abs/2010.13993>`_ paper, where soft predictions
:math:`\mathbf{Z}` (obtained from a simple base predictor) are
first corrected based on ground-truth training
label information :math:`\mathbf{Y}` and residual propagation.

.. math::
    \mathbf{e}^{(0)}_i &= \begin{cases}
        \mathbf{y}_i - \mathbf{z}_i, & \text{if }i
        \text{ is training node,}\\
        \mathbf{0}, & \text{else}
    \end{cases}

.. math::
    \mathbf{E}^{(\ell)} &= \alpha_1 \mathbf{D}^{-1/2}\mathbf{A}
    \mathbf{D}^{-1/2} \mathbf{E}^{(\ell - 1)} +
    (1 - \alpha_1) \mathbf{E}^{(\ell - 1)}

    \mathbf{\hat{Z}} &= \mathbf{Z} + \gamma \cdot \mathbf{E}^{(L_1)},

where :math:`\gamma` denotes the scaling factor (either fixed or
automatically determined), and then smoothed over the graph via label
propagation

.. math::
    \mathbf{\hat{z}}^{(0)}_i &= \begin{cases}
        \mathbf{y}_i, & \text{if }i\text{ is training node,}\\
        \mathbf{\hat{z}}_i, & \text{else}
    \end{cases}

.. math::
    \mathbf{\hat{Z}}^{(\ell)} = \alpha_2 \mathbf{D}^{-1/2}\mathbf{A}
    \mathbf{D}^{-1/2} \mathbf{\hat{Z}}^{(\ell - 1)} +
    (1 - \alpha_2) \mathbf{\hat{Z}}^{(\ell - 1)}

to obtain the final prediction :math:`\mathbf{\hat{Z}}^{(L_2)}`.

.. note::

    For an example of using the C&S model, see
    `examples/correct_and_smooth.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    correct_and_smooth.py>`_.

Args:
    num_correction_layers (int): The number of propagations :math:`L_1`.
    correction_alpha (float): The :math:`\alpha_1` coefficient.
    num_smoothing_layers (int): The number of propagations :math:`L_2`.
    smoothing_alpha (float): The :math:`\alpha_2` coefficient.
    autoscale (bool, optional): If set to :obj:`True`, will automatically
        determine the scaling factor :math:`\gamma`. (default: :obj:`True`)
    scale (float, optional): The scaling factor :math:`\gamma`, in case
        :obj:`autoscale = False`. (default: :obj:`1.0`)

#### Methods

- **`forward(self, y_soft: torch.Tensor, *args) -> torch.Tensor`**
  Applies both :meth:`correct` and :meth:`smooth`.

- **`correct(self, y_soft: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

- **`smooth(self, y_soft: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `CuGraphGATConv`

The graph attentional operator from the `"Graph Attention Networks"
<https://arxiv.org/abs/1710.10903>`_ paper.

:class:`CuGraphGATConv` is an optimized version of
:class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`cugraph-ops`
package that fuses message passing computation for accelerated execution
and lower memory footprint.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch_geometric.edge_index.EdgeIndex, max_num_neighbors: Optional[int] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

### `CuGraphRGCNConv`

The relational graph convolutional operator from the `"Modeling
Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

:class:`CuGraphRGCNConv` is an optimized version of
:class:`~torch_geometric.nn.conv.RGCNConv` based on the :obj:`cugraph-ops`
package that fuses message passing computation for accelerated execution
and lower memory footprint.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch_geometric.edge_index.EdgeIndex, edge_type: torch.Tensor, max_num_neighbors: Optional[int] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

### `CuGraphSAGEConv`

The GraphSAGE operator from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

:class:`CuGraphSAGEConv` is an optimized version of
:class:`~torch_geometric.nn.conv.SAGEConv` based on the :obj:`cugraph-ops`
package that fuses message passing computation for accelerated execution
and lower memory footprint.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch_geometric.edge_index.EdgeIndex, max_num_neighbors: Optional[int] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

### `DMoNPooling`

The spectral modularity pooling operator from the `"Graph Clustering
with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper.

.. math::
    \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{X}

    \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
    \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
\times N \times C}`.
Returns the learned cluster assignment matrix, the pooled node feature
matrix, the coarsened symmetrically normalized adjacency matrix, and three
auxiliary objectives: (1) The spectral loss

.. math::
    \mathcal{L}_s = - \frac{1}{2m}
    \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

where :math:`\mathbf{B}` is the modularity matrix, (2) the orthogonality
loss

.. math::
    \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
    {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
    \right\|}_F

where :math:`C` is the number of clusters, and (3) the cluster loss

.. math::
    \mathcal{L}_c = \frac{\sqrt{C}}{n}
    {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

.. note::

    For an example of using :class:`DMoNPooling`, see
    `examples/proteins_dmon_pool.py
    <https://github.com/pyg-team/pytorch_geometric/blob
    /master/examples/proteins_dmon_pool.py>`_.

Args:
    channels (int or List[int]): Size of each input sample. If given as a
        list, will construct an MLP based on the given feature sizes.
    k (int): The number of clusters.
    dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`**
  Forward pass.

### `DNAConv`

The dynamic neighborhood aggregation operator from the `"Just Jump:
Towards Dynamic Neighborhood Aggregation in Graph Neural Networks"
<https://arxiv.org/abs/1904.04849>`_ paper.

.. math::
    \mathbf{x}_v^{(t)} = h_{\mathbf{\Theta}}^{(t)} \left( \mathbf{x}_{v
    \leftarrow v}^{(t)}, \left\{ \mathbf{x}_{v \leftarrow w}^{(t)} : w \in
    \mathcal{N}(v) \right\} \right)

based on (multi-head) dot-product attention

.. math::
    \mathbf{x}_{v \leftarrow w}^{(t)} = \textrm{Attention} \left(
    \mathbf{x}^{(t-1)}_v \, \mathbf{\Theta}_Q^{(t)}, [\mathbf{x}_w^{(1)},
    \ldots, \mathbf{x}_w^{(t-1)}] \, \mathbf{\Theta}_K^{(t)}, \,
    [\mathbf{x}_w^{(1)}, \ldots, \mathbf{x}_w^{(t-1)}] \,
    \mathbf{\Theta}_V^{(t)} \right)

with :math:`\mathbf{\Theta}_Q^{(t)}, \mathbf{\Theta}_K^{(t)},
\mathbf{\Theta}_V^{(t)}` denoting (grouped) projection matrices for query,
key and value information, respectively.
:math:`h^{(t)}_{\mathbf{\Theta}}` is implemented as a non-trainable
version of :class:`torch_geometric.nn.conv.GCNConv`.

.. note::
    In contrast to other layers, this operator expects node features as
    shape :obj:`[num_nodes, num_layers, channels]`.

Args:
    channels (int): Size of each input/output sample.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    groups (int, optional): Number of groups to use for all linear
        projections. (default: :obj:`1`)
    dropout (float, optional): Dropout probability of attention
        coefficients. (default: :obj:`0.`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    normalize (bool, optional): Whether to add self-loops and apply
        symmetric normalization. (default: :obj:`True`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, L, F)` where :math:`L` is the
      number of layers,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F)`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `DataParallel`

Implements data parallelism at the module level.

This container parallelizes the application of the given :attr:`module` by
splitting a list of :class:`torch_geometric.data.Data` objects and copying
them as :class:`torch_geometric.data.Batch` objects to each device.
In the forward pass, the module is replicated on each device, and each
replica handles a portion of the input.
During the backwards pass, gradients from each replica are summed into the
original module.

The batch size should be larger than the number of GPUs used.

The parallelized :attr:`module` must have its parameters and buffers on
:obj:`device_ids[0]`.

.. note::

    You need to use the :class:`torch_geometric.loader.DataListLoader` for
    this module.

.. warning::

    It is recommended to use
    :class:`torch.nn.parallel.DistributedDataParallel` instead of
    :class:`DataParallel` for multi-GPU training.
    :class:`DataParallel` is usually much slower than
    :class:`~torch.nn.parallel.DistributedDataParallel` even on a single
    machine.
    Take a look `here <https://github.com/pyg-team/pytorch_geometric/blob/
    master/examples/multi_gpu/distributed_batching.py>`_ for an example on
    how to use :pyg:`PyG` in combination with
    :class:`~torch.nn.parallel.DistributedDataParallel`.

Args:
    module (Module): Module to be parallelized.
    device_ids (list of int or torch.device): CUDA devices.
        (default: all devices)
    output_device (int or torch.device): Device location of output.
        (default: :obj:`device_ids[0]`)
    follow_batch (list or tuple, optional): Creates assignment batch
        vectors for each key in the list. (default: :obj:`None`)
    exclude_keys (list or tuple, optional): Will exclude each key in the
        list. (default: :obj:`None`)

#### Methods

- **`forward(self, data_list)`**

- **`scatter(self, data_list, device_ids)`**

### `DeepGCNLayer`

The skip connection operations from the
`"DeepGCNs: Can GCNs Go as Deep as CNNs?"
<https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
The implemented skip connections includes the pre-activation residual
connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

* **Res+** (:obj:`"res+"`):

.. math::
    \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
    \text{GraphConv}\to\text{Res}

* **Res** (:obj:`"res"`) / **Dense** (:obj:`"dense"`) / **Plain**
  (:obj:`"plain"`):

.. math::
    \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
    \text{Res/Dense/Plain}\to\text{Dropout}

.. note::

    For an example of using :obj:`GENConv`, see
    `examples/ogbn_proteins_deepgcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    ogbn_proteins_deepgcn.py>`_.

Args:
    conv (torch.nn.Module, optional): the GCN operator.
        (default: :obj:`None`)
    norm (torch.nn.Module): the normalization layer. (default: :obj:`None`)
    act (torch.nn.Module): the activation layer. (default: :obj:`None`)
    block (str, optional): The skip connection operation to use
        (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
        (default: :obj:`"res+"`)
    dropout (float, optional): Whether to apply or dropout.
        (default: :obj:`0.`)
    ckpt_grad (bool, optional): If set to :obj:`True`, will checkpoint this
        part of the model. Checkpointing works by trading compute for
        memory, since intermediate activations do not need to be kept in
        memory. Set this to :obj:`True` in case you encounter out-of-memory
        errors while going deep. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, *args, **kwargs) -> torch.Tensor`**

### `DeepGraphInfomax`

The Deep Graph Infomax model from the
`"Deep Graph Infomax" <https://arxiv.org/abs/1809.10341>`_
paper based on user-defined encoder and summary model :math:`\mathcal{E}`
and :math:`\mathcal{R}` respectively, and a corruption function
:math:`\mathcal{C}`.

Args:
    hidden_channels (int): The latent space dimensionality.
    encoder (torch.nn.Module): The encoder module :math:`\mathcal{E}`.
    summary (callable): The readout function :math:`\mathcal{R}`.
    corruption (callable): The corruption function :math:`\mathcal{C}`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`**
  Returns the latent space for the input arguments, their

- **`discriminate(self, z: torch.Tensor, summary: torch.Tensor, sigmoid: bool = True) -> torch.Tensor`**
  Given the patch-summary pair :obj:`z` and :obj:`summary`, computes

- **`loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor`**
  Computes the mutual information maximization objective.

- **`test(self, train_z: torch.Tensor, train_y: torch.Tensor, test_z: torch.Tensor, test_y: torch.Tensor, solver: str = 'lbfgs', *args, **kwargs) -> float`**
  Evaluates latent space quality via a logistic regression downstream

### `DeepSetsAggregation`

Performs Deep Sets aggregation in which the elements to aggregate are
first transformed by a Multi-Layer Perceptron (MLP)
:math:`\phi_{\mathbf{\Theta}}`, summed, and then transformed by another MLP
:math:`\rho_{\mathbf{\Theta}}`, as suggested in the `"Graph Neural Networks
with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

Args:
    local_nn (torch.nn.Module, optional): The neural network
        :math:`\phi_{\mathbf{\Theta}}`, *e.g.*, defined by
        :class:`torch.nn.Sequential` or
        :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)
    global_nn (torch.nn.Module, optional): The neural network
        :math:`\rho_{\mathbf{\Theta}}`, *e.g.*, defined by
        :class:`torch.nn.Sequential` or
        :class:`torch_geometric.nn.models.MLP`. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `DegreeScalerAggregation`

Combines one or more aggregators and transforms its output with one or
more scalers as introduced in the `"Principal Neighbourhood Aggregation for
Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.
The scalers are normalised by the in-degree of the training set and so must
be provided at time of construction.
See :class:`torch_geometric.nn.conv.PNAConv` for more information.

Args:
    aggr (str or [str] or Aggregation): The aggregation scheme to use.
        See :class:`~torch_geometric.nn.conv.MessagePassing` for more
        information.
    scaler (str or list): Set of scaling function identifiers, namely one
        or more of :obj:`"identity"`, :obj:`"amplification"`,
        :obj:`"attenuation"`, :obj:`"linear"` and :obj:`"inverse_linear"`.
    deg (Tensor): Histogram of in-degrees of nodes in the training set,
        used by scalers to normalize.
    train_norm (bool, optional): Whether normalization parameters
        are trainable. (default: :obj:`False`)
    aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective aggregation function in case it gets automatically
        resolved. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `DenseGATConv`

See :class:`torch_geometric.nn.conv.GATConv`.

#### Methods

- **`reset_parameters(self)`**

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None, add_loop: bool = True)`**
  Forward pass.

### `DenseGCNConv`

See :class:`torch_geometric.nn.conv.GCNConv`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None, add_loop: bool = True) -> torch.Tensor`**
  Forward pass.

### `DenseGINConv`

See :class:`torch_geometric.nn.conv.GINConv`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None, add_loop: bool = True) -> torch.Tensor`**
  Forward pass.

### `DenseGraphConv`

See :class:`torch_geometric.nn.conv.GraphConv`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `DenseSAGEConv`

See :class:`torch_geometric.nn.conv.SAGEConv`.

.. note::

    :class:`~torch_geometric.nn.dense.DenseSAGEConv` expects to work on
    binary adjacency matrices.
    If you want to make use of weighted dense adjacency matrices, please
    use :class:`torch_geometric.nn.dense.DenseGraphConv` instead.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `DiffGroupNorm`

The differentiable group normalization layer from the `"Towards Deeper
Graph Neural Networks with Differentiable Group Normalization"
<https://arxiv.org/abs/2006.06972>`_ paper, which normalizes node features
group-wise via a learnable soft cluster assignment.

.. math::

    \mathbf{S} = \text{softmax} (\mathbf{X} \mathbf{W})

where :math:`\mathbf{W} \in \mathbb{R}^{F \times G}` denotes a trainable
weight matrix mapping each node into one of :math:`G` clusters.
Normalization is then performed group-wise via:

.. math::

    \mathbf{X}^{\prime} = \mathbf{X} + \lambda \sum_{i = 1}^G
    \text{BatchNorm}(\mathbf{S}[:, i] \odot \mathbf{X})

Args:
    in_channels (int): Size of each input sample :math:`F`.
    groups (int): The number of groups :math:`G`.
    lamda (float, optional): The balancing factor :math:`\lambda` between
        input embeddings and normalized embeddings. (default: :obj:`0.01`)
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    momentum (float, optional): The value used for the running mean and
        running variance computation. (default: :obj:`0.1`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`True`)
    track_running_stats (bool, optional): If set to :obj:`True`, this
        module tracks the running mean and variance, and when set to
        :obj:`False`, this module does not track such statistics and always
        uses batch statistics in both training and eval modes.
        (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**
  Forward pass.

- **`group_distance_ratio(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-05) -> float`**
  Measures the ratio of inter-group distance over intra-group

### `DimeNet`

The directional message passing neural network (DimeNet) from the
`"Directional Message Passing for Molecular Graphs"
<https://arxiv.org/abs/2003.03123>`_ paper.
DimeNet transforms messages based on the angle between them in a
rotation-equivariant fashion.

.. note::

    For an example of using a pretrained DimeNet variant, see
    `examples/qm9_pretrained_dimenet.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    qm9_pretrained_dimenet.py>`_.

Args:
    hidden_channels (int): Hidden embedding size.
    out_channels (int): Size of each output sample.
    num_blocks (int): Number of building blocks.
    num_bilinear (int): Size of the bilinear layer tensor.
    num_spherical (int): Number of spherical harmonics.
    num_radial (int): Number of radial basis functions.
    cutoff (float, optional): Cutoff distance for interatomic
        interactions. (default: :obj:`5.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        collect for each node within the :attr:`cutoff` distance.
        (default: :obj:`32`)
    envelope_exponent (int, optional): Shape of the smooth cutoff.
        (default: :obj:`5`)
    num_before_skip (int, optional): Number of residual layers in the
        interaction blocks before the skip connection. (default: :obj:`1`)
    num_after_skip (int, optional): Number of residual layers in the
        interaction blocks after the skip connection. (default: :obj:`2`)
    num_output_layers (int, optional): Number of linear layers for the
        output blocks. (default: :obj:`3`)
    act (str or Callable, optional): The activation function.
        (default: :obj:`"swish"`)
    output_initializer (str, optional): The initialization method for the
        output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
        (default: :obj:`"zeros"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `DimeNetPlusPlus`

The DimeNet++ from the `"Fast and Uncertainty-Aware
Directional Message Passing for Non-Equilibrium Molecules"
<https://arxiv.org/abs/2011.14115>`_ paper.

:class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
8x faster and 10% more accurate than :class:`DimeNet`.

Args:
    hidden_channels (int): Hidden embedding size.
    out_channels (int): Size of each output sample.
    num_blocks (int): Number of building blocks.
    int_emb_size (int): Size of embedding in the interaction block.
    basis_emb_size (int): Size of basis embedding in the interaction block.
    out_emb_channels (int): Size of embedding in the output block.
    num_spherical (int): Number of spherical harmonics.
    num_radial (int): Number of radial basis functions.
    cutoff: (float, optional): Cutoff distance for interatomic
        interactions. (default: :obj:`5.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        collect for each node within the :attr:`cutoff` distance.
        (default: :obj:`32`)
    envelope_exponent (int, optional): Shape of the smooth cutoff.
        (default: :obj:`5`)
    num_before_skip: (int, optional): Number of residual layers in the
        interaction blocks before the skip connection. (default: :obj:`1`)
    num_after_skip: (int, optional): Number of residual layers in the
        interaction blocks after the skip connection. (default: :obj:`2`)
    num_output_layers: (int, optional): Number of linear layers for the
        output blocks. (default: :obj:`3`)
    act: (str or Callable, optional): The activation funtion.
        (default: :obj:`"swish"`)
    output_initializer (str, optional): The initialization method for the
        output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
        (default: :obj:`"zeros"`)

### `DirGNNConv`

A generic wrapper for computing graph convolution on directed
graphs as described in the `"Edge Directionality Improves Learning on
Heterophilic Graphs" <https://arxiv.org/abs/2305.10498>`_ paper.
:class:`DirGNNConv` will pass messages both from source nodes to target
nodes and from target nodes to source nodes.

Args:
    conv (MessagePassing): The underlying
        :class:`~torch_geometric.nn.conv.MessagePassing` layer to use.
    alpha (float, optional): The alpha coefficient used to weight the
        aggregations of in- and out-edges as part of a convex combination.
        (default: :obj:`0.5`)
    root_weight (bool, optional): If set to :obj:`True`, the layer will add
        transformed root node features to the output.
        (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor`**

### `DistMult`

The DistMult model from the `"Embedding Entities and Relations for
Learning and Inference in Knowledge Bases"
<https://arxiv.org/abs/1412.6575>`_ paper.

:class:`DistMult` models relations as diagonal matrices, which simplifies
the bi-linear interaction between the head and tail entities to the score
function:

.. math::
    d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

.. note::

    For an example of using the :class:`DistMult` model, see
    `examples/kge_fb15k_237.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    kge_fb15k_237.py>`_.

Args:
    num_nodes (int): The number of nodes/entities in the graph.
    num_relations (int): The number of relations in the graph.
    hidden_channels (int): The hidden embedding size.
    margin (float, optional): The margin of the ranking loss.
        (default: :obj:`1.0`)
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
        the embedding matrices will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the score for the given triplet.

- **`loss(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the loss value for the given triplet.

### `DynamicEdgeConv`

The dynamic edge convolutional operator from the `"Dynamic Graph CNN
for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
(see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
dynamically constructed using nearest neighbors in the feature space.

Args:
    nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps pair-wise concatenated node features :obj:`x` of shape
        `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
        *e.g.* defined by :class:`torch.nn.Sequential`.
    k (int): Number of nearest neighbors.
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"max"`)
    num_workers (int): Number of workers to use for k-NN computation.
        Has no effect in case :obj:`batch` is not :obj:`None`, or the input
        lies on the GPU. (default: :obj:`1`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
      if bipartite,
      batch vector :math:`(|\mathcal{V}|)` or
      :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
      if bipartite *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch: Union[torch.Tensor, NoneType, Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `EGConv`

The Efficient Graph Convolution from the `"Adaptive Filters and
Aggregator Fusion for Efficient Graph Convolutions"
<https://arxiv.org/abs/2104.01481>`_ paper.

Its node-wise formulation is given by:

.. math::
    \mathbf{x}_i^{\prime} = {\LARGE ||}_{h=1}^H \sum_{\oplus \in
    \mathcal{A}} \sum_{b = 1}^B w_{i, h, \oplus, b} \;
    \underset{j \in \mathcal{N}(i) \cup \{i\}}{\bigoplus}
    \mathbf{W}_b \mathbf{x}_{j}

with :math:`\mathbf{W}_b` denoting a basis weight,
:math:`\oplus` denoting an aggregator, and :math:`w` denoting per-vertex
weighting coefficients across different heads, bases and aggregators.

EGC retains :math:`\mathcal{O}(|\mathcal{V}|)` memory usage, making it a
sensible alternative to :class:`~torch_geometric.nn.conv.GCNConv`,
:class:`~torch_geometric.nn.conv.SAGEConv` or
:class:`~torch_geometric.nn.conv.GINConv`.

.. note::
    For an example of using :obj:`EGConv`, see `examples/egc.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/
    examples/egc.py>`_.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    aggregators (List[str], optional): Aggregators to be used.
        Supported aggregators are :obj:`"sum"`, :obj:`"mean"`,
        :obj:`"symnorm"`, :obj:`"max"`, :obj:`"min"`, :obj:`"std"`,
        :obj:`"var"`.
        Multiple aggregators can be used to improve the performance.
        (default: :obj:`["symnorm"]`)
    num_heads (int, optional): Number of heads :math:`H` to use. Must have
        :obj:`out_channels % num_heads == 0`. It is recommended to set
        :obj:`num_heads >= num_bases`. (default: :obj:`8`)
    num_bases (int, optional): Number of basis weights :math:`B` to use.
        (default: :obj:`4`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of the edge index with added self loops on first
        execution, along with caching the calculation of the symmetric
        normalized edge weights if the :obj:`"symnorm"` aggregator is
        being used. This parameter should only be set to :obj:`True` in
        transductive learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`aggregate(self, inputs: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None, symnorm_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Aggregates messages from neighbors as

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `EdgeCNN`

The Graph Neural Network from the `"Dynamic Graph CNN for Learning on
Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper, using the
:class:`~torch_geometric.nn.conv.EdgeConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.EdgeConv`.

#### Methods

- **`init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `EdgeConv`

The edge convolutional operator from the `"Dynamic Graph CNN for
Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
    h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
    \mathbf{x}_j - \mathbf{x}_i),

where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

Args:
    nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps pair-wise concatenated node features :obj:`x` of shape
        :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
        *e.g.*, defined by :class:`torch.nn.Sequential`.
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"max"`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `EdgePooling`

The edge pooling operator from the `"Towards Graph Pooling by Edge
Contraction" <https://graphreason.github.io/papers/17.pdf>`__ and
`"Edge Contraction Pooling for Graph Neural Networks"
<https://arxiv.org/abs/1905.10990>`__ papers.

In short, a score is computed for each edge.
Edges are contracted iteratively according to that score unless one of
their nodes has already been part of a contracted edge.

To duplicate the configuration from the `"Towards Graph Pooling by Edge
Contraction" <https://graphreason.github.io/papers/17.pdf>`__ paper, use
either :func:`EdgePooling.compute_edge_score_softmax`
or :func:`EdgePooling.compute_edge_score_tanh`, and set
:obj:`add_to_edge_score` to :obj:`0.0`.

To duplicate the configuration from the `"Edge Contraction Pooling for
Graph Neural Networks" <https://arxiv.org/abs/1905.10990>`__ paper,
set :obj:`dropout` to :obj:`0.2`.

Args:
    in_channels (int): Size of each input sample.
    edge_score_method (callable, optional): The function to apply
        to compute the edge score from raw edge scores. By default,
        this is the softmax over all incoming edges for each node.
        This function takes in a :obj:`raw_edge_score` tensor of shape
        :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
        nodes :obj:`num_nodes`, and produces a new tensor of the same size
        as :obj:`raw_edge_score` describing normalized edge scores.
        Included functions are
        :func:`EdgePooling.compute_edge_score_softmax`,
        :func:`EdgePooling.compute_edge_score_tanh`, and
        :func:`EdgePooling.compute_edge_score_sigmoid`.
        (default: :func:`EdgePooling.compute_edge_score_softmax`)
    dropout (float, optional): The probability with
        which to drop edge scores during training. (default: :obj:`0.0`)
    add_to_edge_score (float, optional): A value to be added to each
        computed edge score. Adding this greatly helps with unpooling
        stability. (default: :obj:`0.5`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`compute_edge_score_softmax(raw_edge_score: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor`**
  Normalizes edge scores via softmax application.

- **`compute_edge_score_tanh(raw_edge_score: torch.Tensor, edge_index: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> torch.Tensor`**
  Normalizes edge scores via hyperbolic tangent application.

- **`compute_edge_score_sigmoid(raw_edge_score: torch.Tensor, edge_index: Optional[torch.Tensor] = None, num_nodes: Optional[int] = None) -> torch.Tensor`**
  Normalizes edge scores via sigmoid application.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch_geometric.nn.pool.edge_pool.UnpoolInfo]`**
  Forward pass.

### `EquilibriumAggregation`

The equilibrium aggregation layer from the `"Equilibrium Aggregation:
Encoding Sets via Optimization" <https://arxiv.org/abs/2202.12795>`_ paper.

The output of this layer :math:`\mathbf{y}` is defined implicitly via a
potential function :math:`F(\mathbf{x}, \mathbf{y})`, a regularization term
:math:`R(\mathbf{y})`, and the condition

.. math::
    \mathbf{y} = \min_\mathbf{y} R(\mathbf{y}) + \sum_{i}
    F(\mathbf{x}_i, \mathbf{y}).

The given implementation uses a ResNet-like model for the potential
function and a simple :math:`L_2` norm :math:`R(\mathbf{y}) =
\textrm{softplus}(\lambda) \cdot {\| \mathbf{y} \|}^2_2` for the
regularizer with learnable weight :math:`\lambda`.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    num_layers (List[int): List of hidden channels in the potential
        function.
    grad_iter (int): The number of steps to take in the internal gradient
        descent. (default: :obj:`5`)
    lamb (float): The initial regularization constant.
        (default: :obj:`0.1`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`init_output(self, dim_size: int) -> torch.Tensor`**

- **`reg(self, y: torch.Tensor) -> torch.Tensor`**

- **`energy(self, x: torch.Tensor, y: torch.Tensor, index: Optional[torch.Tensor], dim_size: Optional[int] = None)`**

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `FAConv`

The Frequency Adaptive Graph Convolution operator from the
`"Beyond Low-Frequency Information in Graph Convolutional Networks"
<https://arxiv.org/abs/2101.00797>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i= \epsilon \cdot \mathbf{x}^{(0)}_i +
    \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}}{\sqrt{d_i d_j}}
    \mathbf{x}_{j}

where :math:`\mathbf{x}^{(0)}_i` and :math:`d_i` denote the initial feature
representation and node degree of node :math:`i`, respectively.
The attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \mathbf{\alpha}_{i,j} = \textrm{tanh}(\mathbf{a}^{\top}[\mathbf{x}_i,
    \mathbf{x}_j])

based on the trainable parameter vector :math:`\mathbf{a}`.

Args:
    channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    eps (float, optional): :math:`\epsilon`-value. (default: :obj:`0.1`)
    dropout (float, optional): Dropout probability of the normalized
        coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`).
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\sqrt{d_i d_j}` on first execution, and
        will use the cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    normalize (bool, optional): Whether to add self-loops (if
        :obj:`add_self_loops` is :obj:`True`) and compute
        symmetric normalization coefficients on the fly.
        If set to :obj:`False`, :obj:`edge_weight` needs to be provided in
        the layer's :meth:`forward` method. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      initial node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)` or
      :math:`((|\mathcal{V}|, F), ((2, |\mathcal{E}|),
      (|\mathcal{E}|)))` if :obj:`return_attention_weights=True`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, x_0: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None, return_attention_weights: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch_geometric.typing.SparseTensor]]`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, alpha_j: torch.Tensor, alpha_i: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `FastRGCNConv`

See :class:`RGCNConv`.

#### Methods

- **`forward(self, x: Union[torch.Tensor, NoneType, Tuple[Optional[torch.Tensor], torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_type: Optional[torch.Tensor] = None)`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_type: torch.Tensor, edge_index_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`aggregate(self, inputs: torch.Tensor, edge_type: torch.Tensor, index: torch.Tensor, dim_size: Optional[int] = None) -> torch.Tensor`**
  Aggregates messages from neighbors as

### `FeaStConv`

The (translation-invariant) feature-steered convolutional operator from
the `"FeaStNet: Feature-Steered Graph Convolutions for 3D Shape Analysis"
<https://arxiv.org/abs/1706.05206>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
    \sum_{j \in \mathcal{N}(i)} \sum_{h=1}^H
    q_h(\mathbf{x}_i, \mathbf{x}_j) \mathbf{W}_h \mathbf{x}_j

with :math:`q_h(\mathbf{x}_i, \mathbf{x}_j) = \mathrm{softmax}_j
(\mathbf{u}_h^{\top} (\mathbf{x}_j - \mathbf{x}_i) + c_h)`, where :math:`H`
denotes the number of attention heads, and :math:`\mathbf{W}_h`,
:math:`\mathbf{u}_h` and :math:`c_h` are trainable parameters.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    heads (int, optional): Number of attention heads :math:`H`.
        (default: :obj:`1`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `FiLMConv`

The FiLM graph convolutional operator from the
`"GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation"
<https://arxiv.org/abs/1906.12192>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{r \in \mathcal{R}}
    \sum_{j \in \mathcal{N}(i)} \sigma \left(
    \boldsymbol{\gamma}_{r,i} \odot \mathbf{W}_r \mathbf{x}_j +
    \boldsymbol{\beta}_{r,i} \right)

where :math:`\boldsymbol{\beta}_{r,i}, \boldsymbol{\gamma}_{r,i} =
g(\mathbf{x}_i)` with :math:`g` being a single linear layer by default.
Self-loops are automatically added to the input graph and represented as
its own relation type.

.. note::

    For an example of using FiLM, see `examples/gcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    film.py>`_.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    num_relations (int, optional): Number of relations. (default: :obj:`1`)
    nn (torch.nn.Module, optional): The neural network :math:`g` that
        maps node features :obj:`x_i` of shape
        :obj:`[-1, in_channels]` to shape :obj:`[-1, 2 * out_channels]`.
        If set to :obj:`None`, :math:`g` will be implemented as a single
        linear layer. (default: :obj:`None`)
    act (callable, optional): Activation function :math:`\sigma`.
        (default: :meth:`torch.nn.ReLU()`)
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"mean"`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge types :math:`(|\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_type: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, beta_i: torch.Tensor, gamma_i: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `FusedGATConv`

The fused graph attention operator from the
`"Understanding GNN Computational Graph: A Coordinated Computation, IO, and
Memory Perspective"
<https://proceedings.mlsys.org/paper/2022/file/
9a1158154dfa42caddbd0694a4e9bdc8-Paper.pdf>`_ paper.

:class:`FusedGATConv` is an optimized version of
:class:`~torch_geometric.nn.conv.GATConv` based on the :obj:`dgNN` package
that fuses message passing computation for accelerated execution and lower
memory footprint.

.. note::

    This implementation is based on the :obj:`dgNN` package.
    See `here <https://github.com/dgSPARSE/dgNN>`__ for instructions on how
    to install.

#### Methods

- **`to_graph_format(edge_index: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]`**
  Converts an :obj:`edge_index` representation of a graph to the

- **`forward(self, x: torch.Tensor, csr: Tuple[torch.Tensor, torch.Tensor], csc: Tuple[torch.Tensor, torch.Tensor], perm: torch.Tensor) -> torch.Tensor`**
  Runs the forward pass of the module.

### `GAE`

The Graph Auto-Encoder model from the
`"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
paper based on user-defined encoder and decoder models.

Args:
    encoder (torch.nn.Module): The encoder module.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, *args, **kwargs) -> torch.Tensor`**
  Alias for :meth:`encode`.

- **`encode(self, *args, **kwargs) -> torch.Tensor`**
  Runs the encoder and computes node-wise latent variables.

- **`decode(self, *args, **kwargs) -> torch.Tensor`**
  Runs the decoder and computes edge probabilities.

- **`recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Given latent variables :obj:`z`, computes the binary cross

### `GAT`

The Graph Neural Network from `"Graph Attention Networks"
<https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
:class:`~torch_geometric.nn.GATConv` or
:class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
respectively.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    v2 (bool, optional): If set to :obj:`True`, will make use of
        :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
        :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GATConv` or
        :class:`torch_geometric.nn.conv.GATv2Conv`.

#### Methods

- **`init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `GATConv`

The graph attentional operator from the `"Graph Attention Networks"
<https://arxiv.org/abs/1710.10903>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
    \alpha_{i,j}\mathbf{\Theta}_t\mathbf{x}_{j},

where the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathrm{LeakyReLU}\left(
    \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
    + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathrm{LeakyReLU}\left(
    \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
    + \mathbf{a}^{\top}_{t}\mathbf{\Theta}_{t}\mathbf{x}_k
    \right)\right)}.

If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathrm{LeakyReLU}\left(
    \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
    + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_j
    + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathrm{LeakyReLU}\left(
    \mathbf{a}^{\top}_{s} \mathbf{\Theta}_{s}\mathbf{x}_i
    + \mathbf{a}^{\top}_{t} \mathbf{\Theta}_{t}\mathbf{x}_k
    + \mathbf{a}^{\top}_{e} \mathbf{\Theta}_{e} \mathbf{e}_{i,k}
    \right)\right)}.

If the graph is not bipartite, :math:`\mathbf{\Theta}_{s} =
\mathbf{\Theta}_{t}`.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities in case of a bipartite graph.
    out_channels (int): Size of each output sample.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    edge_dim (int, optional): Edge feature dimensionality (in case
        there are any). (default: :obj:`None`)
    fill_value (float or torch.Tensor or str, optional): The way to
        generate edge features of self-loops (in case
        :obj:`edge_dim != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    residual (bool, optional): If set to :obj:`True`, the layer will add
        a learnable skip-connection. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
      :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
      If :obj:`return_attention_weights=True`, then
      :math:`((|\mathcal{V}|, H * F_{out}),
      ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
      or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
      (|\mathcal{E}|, H)))` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None, return_attention_weights: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch_geometric.typing.SparseTensor]]`**
  Runs the forward pass of the module.

- **`edge_update(self, alpha_j: torch.Tensor, alpha_i: Optional[torch.Tensor], edge_attr: Optional[torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor], dim_size: Optional[int]) -> torch.Tensor`**
  Computes or updates features for each edge in the graph.

- **`message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GATv2Conv`

The GATv2 operator from the `"How Attentive are Graph Attention
Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
static attention problem of the standard
:class:`~torch_geometric.conv.GATConv` layer.
Since the linear layers in the standard GAT are applied right after each
other, the ranking of attended nodes is unconditioned on the query node.
In contrast, in :class:`GATv2`, every node can attend to any other node.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
    \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

where the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
    \right)\right)}.

If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
the attention coefficients :math:`\alpha_{i,j}` are computed as

.. math::
    \alpha_{i,j} =
    \frac{
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i
    + \mathbf{\Theta}_{t} \mathbf{x}_j
    + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
    \mathbf{\Theta}_{s} \mathbf{x}_i
    + \mathbf{\Theta}_{t} \mathbf{x}_k
    + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]
    \right)\right)}.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities in case of a bipartite graph.
    out_channels (int): Size of each output sample.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    edge_dim (int, optional): Edge feature dimensionality (in case
        there are any). (default: :obj:`None`)
    fill_value (float or torch.Tensor or str, optional): The way to
        generate edge features of self-loops
        (in case :obj:`edge_dim != None`).
        If given as :obj:`float` or :class:`torch.Tensor`, edge features of
        self-loops will be directly given by :obj:`fill_value`.
        If given as :obj:`str`, edge features of self-loops are computed by
        aggregating all features of edges that point to the specific node,
        according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    share_weights (bool, optional): If set to :obj:`True`, the same matrix
        will be applied to the source and the target node of every edge,
        *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
        (default: :obj:`False`)
    residual (bool, optional): If set to :obj:`True`, the layer will add
        a learnable skip-connection. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
      :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
      If :obj:`return_attention_weights=True`, then
      :math:`((|\mathcal{V}|, H * F_{out}),
      ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
      or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
      (|\mathcal{E}|, H)))` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, return_attention_weights: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch_geometric.typing.SparseTensor]]`**
  Runs the forward pass of the module.

- **`edge_update(self, x_j: torch.Tensor, x_i: torch.Tensor, edge_attr: Optional[torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor], dim_size: Optional[int]) -> torch.Tensor`**
  Computes or updates features for each edge in the graph.

- **`message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GCN`

The Graph Neural Network from the `"Semi-supervised
Classification with Graph Convolutional Networks"
<https://arxiv.org/abs/1609.02907>`_ paper, using the
:class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GCNConv`.

#### Methods

- **`init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `GCN2Conv`

The graph convolutional operator with initial residual connections and
identity mapping (GCNII) from the `"Simple and Deep Graph Convolutional
Networks" <https://arxiv.org/abs/2007.02133>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \left( (1 - \alpha) \mathbf{\hat{P}}\mathbf{X} +
    \alpha \mathbf{X^{(0)}}\right) \left( (1 - \beta) \mathbf{I} + \beta
    \mathbf{\Theta} \right)

with :math:`\mathbf{\hat{P}} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
\mathbf{\hat{D}}^{-1/2}`, where
:math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the adjacency
matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix,
and :math:`\mathbf{X}^{(0)}` being the initial feature representation.
Here, :math:`\alpha` models the strength of the initial residual
connection, while :math:`\beta` models the strength of the identity
mapping.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Args:
    channels (int): Size of each input and output sample.
    alpha (float): The strength of the initial residual connection
        :math:`\alpha`.
    theta (float, optional): The hyperparameter :math:`\theta` to compute
        the strength of the identity mapping
        :math:`\beta = \log \left( \frac{\theta}{\ell} + 1 \right)`.
        (default: :obj:`None`)
    layer (int, optional): The layer :math:`\ell` in which this module is
        executed. (default: :obj:`None`)
    shared_weights (bool, optional): If set to :obj:`False`, will use
        different weight matrices for the smoothed representation and the
        initial residual ("GCNII*"). (default: :obj:`True`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    normalize (bool, optional): Whether to add self-loops and apply
        symmetric normalization. (default: :obj:`True`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      initial node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, x_0: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `GCNConv`

The graph convolutional operator from the `"Semi-supervised
Classification with Graph Convolutional Networks"
<https://arxiv.org/abs/1609.02907>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Its node-wise formulation is given by:

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{\Theta}^{\top} \sum_{j \in
    \mathcal{N}(i) \cup \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j
    \hat{d}_i}} \mathbf{x}_j

with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
:math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
node :obj:`i` (default: :obj:`1.0`)

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    improved (bool, optional): If set to :obj:`True`, the layer computes
        :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
        (default: :obj:`False`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. By default, self-loops will be added
        in case :obj:`normalize` is set to :obj:`True`, and not added
        otherwise. (default: :obj:`None`)
    normalize (bool, optional): Whether to add self-loops and compute
        symmetric normalization coefficients on-the-fly.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`
      or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `GENConv`

The GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.

:class:`GENConv` supports both :math:`\textrm{softmax}` (see
:class:`~torch_geometric.nn.aggr.SoftmaxAggregation`) and
:math:`\textrm{powermean}` (see
:class:`~torch_geometric.nn.aggr.PowerMeanAggregation`) aggregation.
Its message construction is given by:

.. math::
    \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_i +
    \mathrm{AGG} \left( \left\{
    \mathrm{ReLU} \left( \mathbf{x}_j + \mathbf{e_{ji}} \right) +\epsilon
    : j \in \mathcal{N}(i) \right\} \right)
    \right)

.. note::

    For an example of using :obj:`GENConv`, see
    `examples/ogbn_proteins_deepgcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    ogbn_proteins_deepgcn.py>`_.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    aggr (str or Aggregation, optional): The aggregation scheme to use.
        Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
        (:obj:`"softmax"`, :obj:`"powermean"`, :obj:`"add"`, :obj:`"mean"`,
        :obj:`max`). (default: :obj:`"softmax"`)
    t (float, optional): Initial inverse temperature for softmax
        aggregation. (default: :obj:`1.0`)
    learn_t (bool, optional): If set to :obj:`True`, will learn the value
        :obj:`t` for softmax aggregation dynamically.
        (default: :obj:`False`)
    p (float, optional): Initial power for power mean aggregation.
        (default: :obj:`1.0`)
    learn_p (bool, optional): If set to :obj:`True`, will learn the value
        :obj:`p` for power mean aggregation dynamically.
        (default: :obj:`False`)
    msg_norm (bool, optional): If set to :obj:`True`, will use message
        normalization. (default: :obj:`False`)
    learn_msg_scale (bool, optional): If set to :obj:`True`, will learn the
        scaling factor of message normalization. (default: :obj:`False`)
    norm (str, optional): Norm layer of MLP layers (:obj:`"batch"`,
        :obj:`"layer"`, :obj:`"instance"`) (default: :obj:`batch`)
    num_layers (int, optional): The number of MLP layers.
        (default: :obj:`2`)
    expansion (int, optional): The expansion factor of hidden channels in
        MLP layers. (default: :obj:`2`)
    eps (float, optional): The epsilon value of the message construction
        function. (default: :obj:`1e-7`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    edge_dim (int, optional): Edge feature dimensionality. If set to
        :obj:`None`, Edge feature dimensionality is expected to match
        the `out_channels`. Other-wise, edge features are linearly
        transformed to match `out_channels` of node feature dimensionality.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GenMessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GIN`

The Graph Neural Network from the `"How Powerful are Graph Neural
Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
:class:`~torch_geometric.nn.GINConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.GINConv`.

#### Methods

- **`init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `GINConv`

The graph isomorphism operator from the `"How Powerful are
Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
    \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

or

.. math::
    \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
    (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

Args:
    nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
        shape :obj:`[-1, out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`.
    eps (float, optional): (Initial) :math:`\epsilon`-value.
        (default: :obj:`0.`)
    train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
        will be a trainable parameter. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `GINEConv`

The modified :class:`GINConv` operator from the `"Strategies for
Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
    \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
    ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
the aggregation procedure.

Args:
    nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
        shape :obj:`[-1, out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`.
    eps (float, optional): (Initial) :math:`\epsilon`-value.
        (default: :obj:`0.`)
    train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
        will be a trainable parameter. (default: :obj:`False`)
    edge_dim (int, optional): Edge feature dimensionality. If set to
        :obj:`None`, node and edge feature dimensionality is expected to
        match. Other-wise, edge features are linearly transformed to match
        node feature dimensionality. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GMMConv`

The gaussian mixture model convolutional operator from the `"Geometric
Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
<https://arxiv.org/abs/1611.08402>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
    \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
    \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

where

.. math::
    \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
    \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
    \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

denotes a weighting function based on trainable mean vector
:math:`\mathbf{\mu}_k` and diagonal covariance matrix
:math:`\mathbf{\Sigma}_k`.

.. note::

    The edge attribute :math:`\mathbf{e}_{ij}` is usually given by
    :math:`\mathbf{e}_{ij} = \mathbf{p}_j - \mathbf{p}_i`, where
    :math:`\mathbf{p}_i` denotes the position of node :math:`i` (see
    :class:`torch_geometric.transform.Cartesian`).

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    dim (int): Pseudo-coordinate dimensionality.
    kernel_size (int): Number of kernels :math:`K`.
    separate_gaussians (bool, optional): If set to :obj:`True`, will
        learn separate GMMs for every pair of input and output channel,
        inspired by traditional CNNs. (default: :obj:`False`)
    aggr (str, optional): The aggregation operator to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"mean"`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None)`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`initialize_parameters(self, module, input)`**

### `GNNFF`

The Graph Neural Network Force Field (GNNFF) from the
`"Accurate and scalable graph neural network force field and molecular
dynamics with direct force architecture"
<https://www.nature.com/articles/s41524-021-00543-3>`_ paper.
:class:`GNNFF` directly predicts atomic forces from automatically
extracted features of the local atomic environment that are
translationally-invariant, but rotationally-covariant to the coordinate of
the atoms.

Args:
    hidden_node_channels (int): Hidden node embedding size.
    hidden_edge_channels (int): Hidden edge embedding size.
    num_layers (int): Number of message passing blocks.
    cutoff (float, optional): Cutoff distance for interatomic
        interactions. (default: :obj:`5.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        collect for each node within the :attr:`cutoff` distance.
        (default: :obj:`32`)

#### Methods

- **`reset_parameters(self)`**

- **`forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

### `GPSConv`

The general, powerful, scalable (GPS) graph transformer layer from the
`"Recipe for a General, Powerful, Scalable Graph Transformer"
<https://arxiv.org/abs/2205.12454>`_ paper.

The GPS layer is based on a 3-part recipe:

1. Inclusion of positional (PE) and structural encodings (SE) to the input
   features (done in a pre-processing step via
   :class:`torch_geometric.transforms`).
2. A local message passing layer (MPNN) that operates on the input graph.
3. A global attention layer that operates on the entire graph.

.. note::

    For an example of using :class:`GPSConv`, see
    `examples/graph_gps.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    graph_gps.py>`_.

Args:
    channels (int): Size of each input sample.
    conv (MessagePassing, optional): The local message passing layer.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    dropout (float, optional): Dropout probability of intermediate
        embeddings. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`"batch_norm"`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    attn_type (str): Global attention type, :obj:`multihead` or
        :obj:`performer`. (default: :obj:`multihead`)
    attn_kwargs (Dict[str, Any], optional): Arguments passed to the
        attention layer. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], batch: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor`**
  Runs the forward pass of the module.

### `GRUAggregation`

Performs GRU aggregation in which the elements to aggregate are
interpreted as a sequence, as described in the `"Graph Neural Networks
with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

.. note::

    :class:`GRUAggregation` requires sorted indices :obj:`index` as input.
    Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

.. warning::

    :class:`GRUAggregation` is not a permutation-invariant operator.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    **kwargs (optional): Additional arguments of :class:`torch.nn.GRU`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `GRetriever`

The G-Retriever model from the `"G-Retriever: Retrieval-Augmented
Generation for Textual Graph Understanding and Question Answering"
<https://arxiv.org/abs/2402.07630>`_ paper.

Args:
    llm (LLM): The LLM to use.
    gnn (torch.nn.Module): The GNN to use.
    use_lora (bool, optional): If set to :obj:`True`, will use LORA from
        :obj:`peft` for training the LLM, see
        `here <https://huggingface.co/docs/peft/en/index>`_ for details.
        (default: :obj:`False`)
    mlp_out_channels (int, optional): The size of each graph embedding
        after projection. (default: :obj:`4096`)

.. warning::
    This module has been tested with the following HuggingFace models

    * :obj:`llm_to_use="meta-llama/Llama-2-7b-chat-hf"`
    * :obj:`llm_to_use="google/gemma-7b"`

    and may not work with other models. See other models at `HuggingFace
    Models <https://huggingface.co/models>`_ and let us know if you
    encounter any issues.

.. note::
    For an example of using :class:`GRetriever`, see
    `examples/llm/g_retriever.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/llm/g_retriever.py>`_.

#### Methods

- **`encode(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor`**

- **`forward(self, question: List[str], x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, label: List[str], edge_attr: Optional[torch.Tensor] = None, additional_text_context: Optional[List[str]] = None)`**
  The forward pass.

- **`inference(self, question: List[str], x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, additional_text_context: Optional[List[str]] = None, max_out_tokens: Optional[int] = 32)`**
  The inference pass.

### `GatedGraphConv`

The gated graph convolution operator from the `"Gated Graph Sequence
Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper.

.. math::
    \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

    \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
    \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

    \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
    \mathbf{h}_i^{(l)})

up to representation :math:`\mathbf{h}_i^{(L)}`.
The number of input channels of :math:`\mathbf{x}_i` needs to be less or
equal than :obj:`out_channels`.
:math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
node :obj:`i` (default: :obj:`1`)

Args:
    out_channels (int): Size of each output sample.
    num_layers (int): The sequence length :math:`L`.
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"add"`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor])`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `GeneralConv`

A general GNN layer adapted from the `"Design Space for Graph Neural
Networks" <https://arxiv.org/abs/2011.08843>`_ paper.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    in_edge_channels (int, optional): Size of each input edge.
        (default: :obj:`None`)
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"mean"`)
    skip_linear (bool, optional): Whether apply linear function in skip
        connection. (default: :obj:`False`)
    directed_msg (bool, optional): If message passing is directed;
        otherwise, message passing is bi-directed. (default: :obj:`True`)
    heads (int, optional): Number of message passing ensembles.
        If :obj:`heads > 1`, the GNN layer will output an ensemble of
        multiple messages.
        If attention is used (:obj:`attention=True`), this corresponds to
        multi-head attention. (default: :obj:`1`)
    attention (bool, optional): Whether to add attention to message
        computation. (default: :obj:`False`)
    attention_type (str, optional): Type of attention: :obj:`"additive"`,
        :obj:`"dot_product"`. (default: :obj:`"additive"`)
    l2_normalize (bool, optional): If set to :obj:`True`, output features
        will be :math:`\ell_2`-normalized, *i.e.*,
        :math:`\frac{\mathbf{x}^{\prime}_i}
        {\| \mathbf{x}^{\prime}_i \|_2}`.
        (default: :obj:`False`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge attributes :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message_basic(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor])`**

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index_i: torch.Tensor, size_i: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GlobalAttention`

The soft attention aggregation layer from the `"Graph Matching Networks
for Learning the Similarity of Graph Structured Objects"
<https://arxiv.org/abs/1904.12787>`_ paper.

.. math::
    \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
    h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \cdot
    h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
\mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
MLPs.

Args:
    gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
        that computes attention scores by mapping node features :obj:`x` of
        shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]` (for
        node-level gating) or :obj:`[1, out_channels]` (for feature-level
        gating), *e.g.*, defined by :class:`torch.nn.Sequential`.
    nn (torch.nn.Module, optional): A neural network
        :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
        shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
        before combining them with the attention scores, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)

### `GraphConv`

The graph neural network operator from the `"Weisfeiler and Leman Go
Neural: Higher-order Graph Neural Networks"
<https://arxiv.org/abs/1810.02244>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2
    \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j

where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
target node :obj:`i` (default: :obj:`1`)

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"add"`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]], edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `GraphMultisetTransformer`

The Graph Multiset Transformer pooling operator from the
`"Accurate Learning of Graph Representations
with Graph Multiset Pooling" <https://arxiv.org/abs/2102.11533>`_ paper.

The :class:`GraphMultisetTransformer` aggregates elements into
:math:`k` representative elements via attention-based pooling, computes the
interaction among them via :obj:`num_encoder_blocks` self-attention blocks,
and finally pools the representative elements via attention-based pooling
into a single cluster.

.. note::

    :class:`GraphMultisetTransformer` requires sorted indices :obj:`index`
    as input. Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

Args:
    channels (int): Size of each input sample.
    k (int): Number of :math:`k` representative nodes after pooling.
    num_encoder_blocks (int, optional): Number of Set Attention Blocks
        (SABs) between the two pooling blocks. (default: :obj:`1`)
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    norm (str, optional): If set to :obj:`True`, will apply layer
        normalization. (default: :obj:`False`)
    dropout (float, optional): Dropout probability of attention weights.
        (default: :obj:`0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `GraphNorm`

Applies graph normalization over individual graphs as described in the
`"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
Training" <https://arxiv.org/abs/2009.03294>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
    \textrm{E}[\mathbf{x}]}
    {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
    + \epsilon}} \odot \gamma + \beta

where :math:`\alpha` denotes parameters that learn how much information
to keep in the mean.

Args:
    in_channels (int): Size of each input sample.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `GraphSAGE`

The Graph Neural Network from the `"Inductive Representation Learning
on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
:class:`~torch_geometric.nn.SAGEConv` operator for message passing.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.SAGEConv`.

#### Methods

- **`init_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `GraphSizeNorm`

Applies Graph Size Normalization over each individual graph in a batch
of node features as described in the
`"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{\mathbf{x}_i}{\sqrt{|\mathcal{V}|}}

#### Methods

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `GraphUNet`

The Graph U-Net model from the `"Graph U-Nets"
<https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
architecture with graph pooling and unpooling operations.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output sample.
    depth (int): The depth of the U-Net architecture.
    pool_ratios (float or [float], optional): Graph pooling ratio for each
        depth. (default: :obj:`0.5`)
    sum_res (bool, optional): If set to :obj:`False`, will use
        concatenation for integration of skip connections instead
        summation. (default: :obj:`True`)
    act (torch.nn.functional, optional): The nonlinearity to use.
        (default: :obj:`torch.nn.functional.relu`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

- **`augment_adj(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]`**

### `GravNetConv`

The GravNet operator from the `"Learning Representations of Irregular
Particle-detector Geometry with Distance-weighted Graph
Networks" <https://arxiv.org/abs/1902.07987>`_ paper, where the graph is
dynamically constructed using nearest neighbors.
The neighbors are constructed in a learnable low-dimensional projection of
the feature space.
A second projection of the input feature space is then propagated from the
neighbors to each vertex using distance weights that are derived by
applying a Gaussian function to the distances.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): The number of output channels.
    space_dimensions (int): The dimensionality of the space used to
       construct the neighbors; referred to as :math:`S` in the paper.
    propagate_dimensions (int): The number of features to be propagated
       between the vertices; referred to as :math:`F_{\textrm{LR}}` in the
       paper.
    k (int): The number of nearest neighbors.
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
      if bipartite,
      batch vector :math:`(|\mathcal{V}|)` or
      :math:`((|\mathcal{V}_s|), (|\mathcal{V}_t|))` if bipartite
      *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch: Union[torch.Tensor, NoneType, Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `GroupAddRev`

The Grouped Reversible GNN module from the `"Graph Neural Networks with
1000 Layers" <https://arxiv.org/abs/2106.07476>`_ paper.
This module enables training of arbitary deep GNNs with a memory complexity
independent of the number of layers.

It does so by partitioning input node features :math:`\mathbf{X}` into
:math:`C` groups across the feature dimension. Then, a grouped reversible
GNN block :math:`f_{\theta(i)}` operates on a group of inputs and produces
a group of outputs:

.. math::

    \mathbf{X}^{\prime}_0 &= \sum_{i=2}^C \mathbf{X}_i

    \mathbf{X}^{\prime}_i &= f_{\theta(i)} ( \mathbf{X}^{\prime}_{i - 1},
    \mathbf{A}) + \mathbf{X}_i

for all :math:`i \in \{ 1, \ldots, C \}`.

.. note::

    For an example of using :class:`GroupAddRev`, see `examples/rev_gnn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    rev_gnn.py>`_.

Args:
    conv (torch.nn.Module or torch.nn.ModuleList]): A seed GNN. The input
        and output feature dimensions need to match.
    split_dim (int, optional): The dimension across which to split groups.
        (default: :obj:`-1`)
    num_groups (int, optional): The number of groups :math:`C`.
        (default: :obj:`None`)
    disable (bool, optional): If set to :obj:`True`, will disable the usage
        of :class:`InvertibleFunction` and will execute the module without
        memory savings. (default: :obj:`False`)
    num_bwd_passes (int, optional): Number of backward passes to retain a
        link with the output. After the last backward pass the output is
        discarded and memory is freed. (default: :obj:`1`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

### `HANConv`

The Heterogenous Graph Attention Operator from the
`"Heterogenous Graph Attention Network"
<https://arxiv.org/abs/1903.07293>`_ paper.

.. note::

    For an example of using HANConv, see `examples/hetero/han_imdb.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    hetero/han_imdb.py>`_.

Args:
    in_channels (int or Dict[str, int]): Size of each input sample of every
        node type, or :obj:`-1` to derive the size from the first input(s)
        to the forward method.
    out_channels (int): Size of each output sample.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]], return_semantic_attention_weights: bool = False) -> Union[Dict[str, Optional[torch.Tensor]], Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]]]]`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, alpha_i: torch.Tensor, alpha_j: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `HEATConv`

The heterogeneous edge-enhanced graph attentional operator from the
`"Heterogeneous Edge-Enhanced Graph Attention Network For Multi-Agent
Trajectory Prediction" <https://arxiv.org/abs/2106.07161>`_ paper.

:class:`HEATConv` enhances :class:`~torch_geometric.nn.conv.GATConv` by:

1. type-specific transformations of nodes of different types
2. edge type and edge feature incorporation, in which edges are assumed to
   have different types but contain the same kind of attributes

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    num_node_types (int): The number of node types.
    num_edge_types (int): The number of edge types.
    edge_type_emb_dim (int): The embedding size of edge types.
    edge_dim (int): Edge feature dimensionality.
    edge_attr_emb_dim (int): The embedding size of edge features.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      node types :math:`(|\mathcal{V}|)`,
      edge types :math:`(|\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], node_type: torch.Tensor, edge_type: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_type_emb: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `HGTConv`

The Heterogeneous Graph Transformer (HGT) operator from the
`"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
paper.

.. note::

    For an example of using HGT, see `examples/hetero/hgt_dblp.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    hetero/hgt_dblp.py>`_.

Args:
    in_channels (int or Dict[str, int]): Size of each input sample of every
        node type, or :obj:`-1` to derive the size from the first input(s)
        to the forward method.
    out_channels (int): Size of each output sample.
    metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
        of the heterogeneous graph, *i.e.* its node and edge types given
        by a list of strings and a list of string triplets, respectively.
        See :meth:`torch_geometric.data.HeteroData.metadata` for more
        information.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[Tuple[str, str, str], Union[torch.Tensor, torch_geometric.typing.SparseTensor]]) -> Dict[str, Optional[torch.Tensor]]`**
  Runs the forward pass of the module.

- **`message(self, k_j: torch.Tensor, q_i: torch.Tensor, v_j: torch.Tensor, edge_attr: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `HeteroBatchNorm`

Applies batch normalization over a batch of heterogeneous features as
described in the `"Batch Normalization: Accelerating Deep Network Training
by Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
paper.
Compared to :class:`BatchNorm`, :class:`HeteroBatchNorm` applies
normalization individually for each node or edge type.

Args:
    in_channels (int): Size of each input sample.
    num_types (int): The number of types.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    momentum (float, optional): The value used for the running mean and
        running variance computation. (default: :obj:`0.1`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`True`)
    track_running_stats (bool, optional): If set to :obj:`True`, this
        module tracks the running mean and variance, and when set to
        :obj:`False`, this module does not track such statistics and always
        uses batch statistics in both training and eval modes.
        (default: :obj:`True`)

#### Methods

- **`reset_running_stats(self)`**
  Resets all running statistics of the module.

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, type_vec: torch.Tensor) -> torch.Tensor`**
  Forward pass.

### `HeteroConv`

A generic wrapper for computing graph convolution on heterogeneous
graphs.
This layer will pass messages from source nodes to target nodes based on
the bipartite GNN layer given for a specific edge type.
If multiple relations point to the same destination, their results will be
aggregated according to :attr:`aggr`.
In comparison to :meth:`torch_geometric.nn.to_hetero`, this layer is
especially useful if you want to apply different message passing modules
for different edge types.

.. code-block:: python

    hetero_conv = HeteroConv({
        ('paper', 'cites', 'paper'): GCNConv(-1, 64),
        ('author', 'writes', 'paper'): SAGEConv((-1, -1), 64),
        ('paper', 'written_by', 'author'): GATConv((-1, -1), 64),
    }, aggr='sum')

    out_dict = hetero_conv(x_dict, edge_index_dict)

    print(list(out_dict.keys()))
    >>> ['paper', 'author']

Args:
    convs (Dict[Tuple[str, str, str], MessagePassing]): A dictionary
        holding a bipartite
        :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
        individual edge type.
    aggr (str, optional): The aggregation scheme to use for grouping node
        embeddings generated by different relations
        (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"cat"`, :obj:`None`). (default: :obj:`"sum"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, *args_dict, **kwargs_dict) -> Dict[str, torch.Tensor]`**
  Runs the forward pass of the module.

### `HeteroDictLinear`

Applies separate linear transformations to the incoming data
dictionary.

For key :math:`\kappa`, it computes

.. math::
    \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
    \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}.

It supports lazy initialization and customizable weight and bias
initialization.

Args:
    in_channels (int or Dict[Any, int]): Size of each input sample. If
        passed an integer, :obj:`types` will be a mandatory argument.
        initialized lazily in case it is given as :obj:`-1`.
    out_channels (int): Size of each output sample.
    types (List[Any], optional): The keys of the input dictionary.
        (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.Linear`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`**
  Forward pass.

- **`initialize_parameters(self, module, input)`**

### `HeteroJumpingKnowledge`

A heterogeneous version of the :class:`JumpingKnowledge` module.

Args:
    types (List[str]): The keys of the input dictionary.
    mode (str): The aggregation scheme to use
        (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
    channels (int, optional): The number of channels per representation.
        Needs to be only set for LSTM-style aggregation.
        (default: :obj:`None`)
    num_layers (int, optional): The number of layers to aggregate. Needs to
        be only set for LSTM-style aggregation. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, xs_dict: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]`**
  Forward pass.

### `HeteroLayerNorm`

Applies layer normalization over each individual example in a batch
of heterogeneous features as described in the `"Layer Normalization"
<https://arxiv.org/abs/1607.06450>`_ paper.
Compared to :class:`LayerNorm`, :class:`HeteroLayerNorm` applies
normalization individually for each node or edge type.

Args:
    in_channels (int): Size of each input sample.
    num_types (int): The number of types.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`True`)
    mode (str, optinal): The normalization mode to use for layer
        normalization (:obj:`"node"`). If `"node"` is used, each node will
        be considered as an element to be normalized.
        (default: :obj:`"node"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, type_vec: Optional[torch.Tensor] = None, type_ptr: Union[torch.Tensor, List[int], NoneType] = None) -> torch.Tensor`**
  Forward pass.

### `HeteroLinear`

Applies separate linear transformations to the incoming data according
to types.

For type :math:`\kappa`, it computes

.. math::
    \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
    \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}.

It supports lazy initialization and customizable weight and bias
initialization.

Args:
    in_channels (int): Size of each input sample. Will be initialized
        lazily in case it is given as :obj:`-1`.
    out_channels (int): Size of each output sample.
    num_types (int): The number of types.
    is_sorted (bool, optional): If set to :obj:`True`, assumes that
        :obj:`type_vec` is sorted. This avoids internal re-sorting of the
        data and can improve runtime and memory efficiency.
        (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.Linear`.

Shapes:
    - **input:**
      features :math:`(*, F_{in})`,
      type vector :math:`(*)`
    - **output:** features :math:`(*, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward_naive(self, x: torch.Tensor, type_ptr: torch.Tensor) -> torch.Tensor`**

- **`forward_segmm(self, x: torch.Tensor, type_ptr: torch.Tensor) -> torch.Tensor`**

- **`forward(self, x: torch.Tensor, type_vec: torch.Tensor) -> torch.Tensor`**
  The forward pass.

- **`initialize_parameters(self, module, input)`**

### `HypergraphConv`

The hypergraph convolutional operator from the `"Hypergraph Convolution
and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
    \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
weight matrix, and
:math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
matrices.

For example, in the hypergraph scenario
:math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
:math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
:math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
:obj:`hyperedge_index` is represented as:

.. code-block:: python

    hyperedge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3],
        [0, 0, 0, 1, 1, 1],
    ])

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    use_attention (bool, optional): If set to :obj:`True`, attention
        will be added to this layer. (default: :obj:`False`)
    attention_mode (str, optional): The mode on how to compute attention.
        If set to :obj:`"node"`, will compute attention scores of nodes
        within all nodes belonging to the same hyperedge.
        If set to :obj:`"edge"`, will compute attention scores of nodes
        across all edges holding this node belongs to.
        (default: :obj:`"node"`)
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
      hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
      hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, hyperedge_index: torch.Tensor, hyperedge_weight: Optional[torch.Tensor] = None, hyperedge_attr: Optional[torch.Tensor] = None, num_edges: Optional[int] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, norm_i: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `InnerProductDecoder`

The inner product decoder from the `"Variational Graph Auto-Encoders"
<https://arxiv.org/abs/1611.07308>`_ paper.

.. math::
    \sigma(\mathbf{Z}\mathbf{Z}^{\top})

where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
space produced by the encoder.

#### Methods

- **`forward(self, z: torch.Tensor, edge_index: torch.Tensor, sigmoid: bool = True) -> torch.Tensor`**
  Decodes the latent variables :obj:`z` into edge probabilities for

- **`forward_all(self, z: torch.Tensor, sigmoid: bool = True) -> torch.Tensor`**
  Decodes the latent variables :obj:`z` into a probabilistic dense

### `InstanceNorm`

Applies instance normalization over each individual example in a batch
of node features as described in the `"Instance Normalization: The Missing
Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
    \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
    \odot \gamma + \beta

The mean and standard-deviation are calculated per-dimension separately for
each object in a mini-batch.

Args:
    in_channels (int): Size of each input sample.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    momentum (float, optional): The value used for the running mean and
        running variance computation. (default: :obj:`0.1`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`False`)
    track_running_stats (bool, optional): If set to :obj:`True`, this
        module tracks the running mean and variance, and when set to
        :obj:`False`, this module does not track such statistics and always
        uses instance statistics in both training and eval modes.
        (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `JumpingKnowledge`

The Jumping Knowledge layer aggregation module from the
`"Representation Learning on Graphs with Jumping Knowledge Networks"
<https://arxiv.org/abs/1806.03536>`_ paper.

Jumping knowledge is performed based on either **concatenation**
(:obj:`"cat"`)

.. math::

    \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)},

**max pooling** (:obj:`"max"`)

.. math::

    \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right),

or **weighted summation**

.. math::

    \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
LSTM (:obj:`"lstm"`).

Args:
    mode (str): The aggregation scheme to use
        (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
    channels (int, optional): The number of channels per representation.
        Needs to be only set for LSTM-style aggregation.
        (default: :obj:`None`)
    num_layers (int, optional): The number of layers to aggregate. Needs to
        be only set for LSTM-style aggregation. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, xs: List[torch.Tensor]) -> torch.Tensor`**
  Forward pass.

### `KGEModel`

An abstract base class for implementing custom KGE models.

Args:
    num_nodes (int): The number of nodes/entities in the graph.
    num_relations (int): The number of relations in the graph.
    hidden_channels (int): The hidden embedding size.
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
        embedding matrices will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the score for the given triplet.

- **`loss(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the loss value for the given triplet.

- **`loader(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor, **kwargs) -> torch.Tensor`**
  Returns a mini-batch loader that samples a subset of triplets.

- **`test(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor, batch_size: int, k: int = 10, log: bool = True) -> Tuple[float, float, float]`**
  Evaluates the model quality by computing Mean Rank, MRR and

### `KNNIndex`

A base class to perform fast :math:`k`-nearest neighbor search
(:math:`k`-NN) via the :obj:`faiss` library.

Please ensure that :obj:`faiss` is installed by running

.. code-block:: bash

    pip install faiss-cpu
    # or
    pip install faiss-gpu

depending on whether to plan to use GPU-processing for :math:`k`-NN search.

Args:
    index_factory (str, optional): The name of the index factory to use,
        *e.g.*, :obj:`"IndexFlatL2"` or :obj:`"IndexFlatIP"`. See `here
        <https://github.com/facebookresearch/faiss/wiki/
        The-index-factory>`_ for more information.
    emb (torch.Tensor, optional): The data points to add.
        (default: :obj:`None`)
    reserve (int, optional): The number of elements to reserve memory for
        before re-allocating (GPU-only). (default: :obj:`None`)

#### Methods

- **`add(self, emb: torch.Tensor)`**
  Adds new data points to the :class:`KNNIndex` to search in.

- **`search(self, emb: torch.Tensor, k: int, exclude_links: Optional[torch.Tensor] = None) -> torch_geometric.nn.pool.knn.KNNOutput`**
  Search for the :math:`k` nearest neighbors of the given data

- **`get_emb(self) -> torch.Tensor`**
  Returns the data points stored in the :class:`KNNIndex`.

### `L2KNNIndex`

Performs fast :math:`k`-nearest neighbor search (:math:`k`-NN) based on
the :math:`L_2` metric via the :obj:`faiss` library.

Args:
    emb (torch.Tensor, optional): The data points to add.
        (default: :obj:`None`)

### `LCMAggregation`

The Learnable Commutative Monoid aggregation from the
`"Learnable Commutative Monoids for Graph Neural Networks"
<https://arxiv.org/abs/2212.08541>`_ paper, in which the elements are
aggregated using a binary tree reduction with
:math:`\mathcal{O}(\log |\mathcal{V}|)` depth.

.. note::

    :class:`LCMAggregation` requires sorted indices :obj:`index` as input.
    Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

.. warning::

    :class:`LCMAggregation` is not a permutation-invariant operator.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    project (bool, optional): If set to :obj:`True`, the layer will apply a
        linear transformation followed by an activation function before
        aggregation. (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `LEConv`

The local extremum graph neural network operator from the
`"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph
Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

:class:`LEConv` finds the importance of nodes with respect to their
neighbors using the difference operator:

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{x}_i \cdot \mathbf{\Theta}_1 +
    \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
    (\mathbf{\Theta}_2 \mathbf{x}_i - \mathbf{\Theta}_3 \mathbf{x}_j)

where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
target node :obj:`i` (default: :obj:`1`)

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    bias (bool, optional): If set to :obj:`False`, the layer will
        not learn an additive bias. (default: :obj:`True`).
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, a_j: torch.Tensor, b_i: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `LGConv`

The Light Graph Convolution (LGC) operator from the `"LightGCN:
Simplifying and Powering Graph Convolution Network for Recommendation"
<https://arxiv.org/abs/2002.02126>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
    \frac{e_{j,i}}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j

Args:
    normalize (bool, optional): If set to :obj:`False`, output features
        will not be normalized via symmetric normalization.
        (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)`

#### Methods

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `LINKX`

The LINKX model from the `"Large Scale Learning on Non-Homophilous
Graphs: New Benchmarks and Strong Simple Methods"
<https://arxiv.org/abs/2110.14446>`_ paper.

.. math::
    \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

    \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

    \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
    [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
    \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

.. note::

    For an example of using LINKX, see `examples/linkx.py <https://
    github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

Args:
    num_nodes (int): The number of nodes in the graph.
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output sample.
    num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
    num_edge_layers (int, optional): Number of layers of
        :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
    num_node_layers (int, optional): Number of layers of
        :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
    dropout (float, optional): Dropout probability of each hidden
        embedding. (default: :obj:`0.0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Optional[torch.Tensor], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**

### `LSTMAggregation`

Performs LSTM-style aggregation in which the elements to aggregate are
interpreted as a sequence, as described in the `"Inductive Representation
Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

.. note::

    :class:`LSTMAggregation` requires sorted indices :obj:`index` as input.
    Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

.. warning::

    :class:`LSTMAggregation` is not a permutation-invariant operator.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `LabelPropagation`

The label propagation operator, firstly introduced in the
`"Learning from Labeled and Unlabeled Data with Label Propagation"
<http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper.

.. math::
    \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

where unlabeled data is inferred by labeled data via propagation.
This concrete implementation here is derived from the `"Combining Label
Propagation And Simple Models Out-performs Graph Neural Networks"
<https://arxiv.org/abs/2010.13993>`_ paper.

.. note::

    For an example of using the :class:`LabelPropagation`, see
    `examples/label_prop.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    label_prop.py>`_.

Args:
    num_layers (int): The number of propagations.
    alpha (float): The :math:`\alpha` coefficient.

#### Methods

- **`forward(self, y: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], mask: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None, post_step: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> torch.Tensor`**
  Forward pass.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: torch_geometric.typing.SparseTensor, x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `LayerNorm`

Applies layer normalization over each individual example in a batch
of features as described in the `"Layer Normalization"
<https://arxiv.org/abs/1607.06450>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
    \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
    \odot \gamma + \beta

The mean and standard-deviation are calculated across all nodes and all
node channels separately for each object in a mini-batch.

Args:
    in_channels (int): Size of each input sample.
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)
    affine (bool, optional): If set to :obj:`True`, this module has
        learnable affine parameters :math:`\gamma` and :math:`\beta`.
        (default: :obj:`True`)
    mode (str, optinal): The normalization mode to use for layer
        normalization (:obj:`"graph"` or :obj:`"node"`). If :obj:`"graph"`
        is used, each graph will be considered as an element to be
        normalized. If `"node"` is used, each node will be considered as
        an element to be normalized. (default: :obj:`"graph"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `LightGCN`

The LightGCN model from the `"LightGCN: Simplifying and Powering
Graph Convolution Network for Recommendation"
<https://arxiv.org/abs/2002.02126>`_ paper.

:class:`~torch_geometric.nn.models.LightGCN` learns embeddings by linearly
propagating them on the underlying graph, and uses the weighted sum of the
embeddings learned at all layers as the final embedding

.. math::
    \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{(l)}_i,

where each layer's embedding is computed as

.. math::
    \mathbf{x}^{(l+1)}_i = \sum_{j \in \mathcal{N}(i)}
    \frac{1}{\sqrt{\deg(i)\deg(j)}}\mathbf{x}^{(l)}_j.

Two prediction heads and training objectives are provided:
**link prediction** (via
:meth:`~torch_geometric.nn.models.LightGCN.link_pred_loss` and
:meth:`~torch_geometric.nn.models.LightGCN.predict_link`) and
**recommendation** (via
:meth:`~torch_geometric.nn.models.LightGCN.recommendation_loss` and
:meth:`~torch_geometric.nn.models.LightGCN.recommend`).

.. note::

    Embeddings are propagated according to the graph connectivity specified
    by :obj:`edge_index` while rankings or link probabilities are computed
    according to the edges specified by :obj:`edge_label_index`.

.. note::

    For an example of using :class:`LightGCN`, see `examples/lightgcn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    lightgcn.py>`_.

Args:
    num_nodes (int): The number of nodes in the graph.
    embedding_dim (int): The dimensionality of node embeddings.
    num_layers (int): The number of
        :class:`~torch_geometric.nn.conv.LGConv` layers.
    alpha (float or torch.Tensor, optional): The scalar or vector
        specifying the re-weighting coefficients for aggregating the final
        embedding. If set to :obj:`None`, the uniform initialization of
        :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
    **kwargs (optional): Additional arguments of the underlying
        :class:`~torch_geometric.nn.conv.LGConv` layers.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`get_embedding(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Returns the embedding of nodes in the graph.

- **`forward(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_label_index: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Computes rankings for pairs of nodes.

- **`predict_link(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_label_index: Optional[torch.Tensor] = None, edge_weight: Optional[torch.Tensor] = None, prob: bool = False) -> torch.Tensor`**
  Predict links between nodes specified in :obj:`edge_label_index`.

- **`recommend(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None, src_index: Optional[torch.Tensor] = None, dst_index: Optional[torch.Tensor] = None, k: int = 1, sorted: bool = True) -> torch.Tensor`**
  Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

### `Linear`

Applies a linear transformation to the incoming data.

.. math::
    \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

In contrast to :class:`torch.nn.Linear`, it supports lazy initialization
and customizable weight and bias initialization.

Args:
    in_channels (int): Size of each input sample. Will be initialized
        lazily in case it is given as :obj:`-1`.
    out_channels (int): Size of each output sample.
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    weight_initializer (str, optional): The initializer for the weight
        matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
        or :obj:`None`).
        If set to :obj:`None`, will match default weight initialization of
        :class:`torch.nn.Linear`. (default: :obj:`None`)
    bias_initializer (str, optional): The initializer for the bias vector
        (:obj:`"zeros"` or :obj:`None`).
        If set to :obj:`None`, will match default bias initialization of
        :class:`torch.nn.Linear`. (default: :obj:`None`)

Shapes:
    - **input:** features :math:`(*, F_{in})`
    - **output:** features :math:`(*, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**
  Forward pass.

- **`initialize_parameters(self, module, input)`**

### `MFConv`

The graph neural network operator from the
`"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
<https://arxiv.org/abs/1509.09292>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}^{(\deg(i))}_1 \mathbf{x}_i +
    \mathbf{W}^{(\deg(i))}_2 \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j

which trains a distinct weight matrix for each possible vertex degree.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    max_degree (int, optional): The maximum node degree to consider when
        updating weights (default: :obj:`10`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **inputs:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `MIPSKNNIndex`

Performs fast :math:`k`-nearest neighbor search (:math:`k`-NN) based on
the maximum inner product via the :obj:`faiss` library.

Args:
    emb (torch.Tensor, optional): The data points to add.
        (default: :obj:`None`)

### `MLP`

A Multi-Layer Perception (MLP) model.

There exists two ways to instantiate an :class:`MLP`:

1. By specifying explicit channel sizes, *e.g.*,

   .. code-block:: python

      mlp = MLP([16, 32, 64, 128])

   creates a three-layer MLP with **differently** sized hidden layers.

1. By specifying fixed hidden channel sizes over a number of layers,
   *e.g.*,

   .. code-block:: python

      mlp = MLP(in_channels=16, hidden_channels=32,
                out_channels=128, num_layers=3)

   creates a three-layer MLP with **equally** sized hidden layers.

Args:
    channel_list (List[int] or int, optional): List of input, intermediate
        and output channels such that :obj:`len(channel_list) - 1` denotes
        the number of layers of the MLP (default: :obj:`None`)
    in_channels (int, optional): Size of each input sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    hidden_channels (int, optional): Size of each hidden sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    out_channels (int, optional): Size of each output sample.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    num_layers (int, optional): The number of layers.
        Will override :attr:`channel_list`. (default: :obj:`None`)
    dropout (float or List[float], optional): Dropout probability of each
        hidden embedding. If a list is provided, sets the dropout value per
        layer. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`"batch_norm"`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    plain_last (bool, optional): If set to :obj:`False`, will apply
        non-linearity, batch normalization and dropout to the last layer as
        well. (default: :obj:`True`)
    bias (bool or List[bool], optional): If set to :obj:`False`, the module
        will not learn additive biases. If a list is provided, sets the
        bias per layer. (default: :obj:`True`)
    **kwargs (optional): Additional deprecated arguments of the MLP layer.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None, return_emb: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `MLPAggregation`

Performs MLP aggregation in which the elements to aggregate are
flattened into a single vectorial representation, and are then processed by
a Multi-Layer Perceptron (MLP), as described in the `"Graph Neural Networks
with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.

.. note::

    :class:`MLPAggregation` requires sorted indices :obj:`index` as input.
    Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

.. warning::

    :class:`MLPAggregation` is not a permutation-invariant operator.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    max_num_elements (int): The maximum number of elements to aggregate per
        group.
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.models.MLP`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `MaskLabel`

The label embedding and masking layer from the `"Masked Label
Prediction: Unified Message Passing Model for Semi-Supervised
Classification" <https://arxiv.org/abs/2009.03509>`_ paper.

Here, node labels :obj:`y` are merged to the initial node features :obj:`x`
for a subset of their nodes according to :obj:`mask`.

.. note::

    For an example of using :class:`MaskLabel`, see
    `examples/unimp_arxiv.py <https://github.com/pyg-team/
    pytorch_geometric/blob/master/examples/unimp_arxiv.py>`_.


Args:
    num_classes (int): The number of classes.
    out_channels (int): Size of each output sample.
    method (str, optional): If set to :obj:`"add"`, label embeddings are
        added to the input. If set to :obj:`"concat"`, label embeddings are
        concatenated. In case :obj:`method="add"`, then :obj:`out_channels`
        needs to be identical to the input dimensionality of node features.
        (default: :obj:`"add"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`**

- **`ratio_mask(mask: torch.Tensor, ratio: float)`**
  Modifies :obj:`mask` by setting :obj:`ratio` of :obj:`True`

### `MaxAggregation`

An aggregation operator that takes the feature-wise maximum across a
set of elements.

.. math::
    \mathrm{max}(\mathcal{X}) = \max_{\mathbf{x}_i \in \mathcal{X}}
    \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `MeanAggregation`

An aggregation operator that averages features across a set of
elements.

.. math::
    \mathrm{mean}(\mathcal{X}) = \frac{1}{|\mathcal{X}|}
    \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `MeanSubtractionNorm`

Applies layer normalization by subtracting the mean from the inputs
as described in the  `"Revisiting 'Over-smoothing' in Deep GCNs"
<https://arxiv.org/abs/2003.13663>`_ paper.

.. math::
    \mathbf{x}_i = \mathbf{x}_i - \frac{1}{|\mathcal{V}|}
    \sum_{j \in \mathcal{V}} \mathbf{x}_j

#### Methods

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `MedianAggregation`

An aggregation operator that returns the feature-wise median of a set.

That is, for every feature :math:`d`, it computes

.. math::
    {\mathrm{median}(\mathcal{X})}_d = x_{\pi_i,d}

where :math:`x_{\pi_1,d} \le x_{\pi_2,d} \le \dots \le
x_{\pi_n,d}` and :math:`i = \lfloor \frac{n}{2} \rfloor`.

.. note::
    If the median lies between two values, the lowest one is returned.
    To compute the midpoint (or other kind of interpolation) of the two
    values, use :class:`QuantileAggregation` instead.

Args:
    fill_value (float, optional): The default value in the case no entry is
        found for a given index (default: :obj:`0.0`).

### `MemPooling`

Memory based pooling layer from `"Memory-Based Graph Networks"
<https://arxiv.org/abs/2002.09518>`_ paper, which learns a coarsened graph
representation based on soft cluster assignments.

.. math::
    S_{i,j}^{(h)} &= \frac{
    (1+{\| \mathbf{x}_i-\mathbf{k}^{(h)}_j \|}^2 / \tau)^{
    -\frac{1+\tau}{2}}}{
    \sum_{k=1}^K (1 + {\| \mathbf{x}_i-\mathbf{k}^{(h)}_k \|}^2 / \tau)^{
    -\frac{1+\tau}{2}}}

    \mathbf{S} &= \textrm{softmax}(\textrm{Conv2d}
    (\Vert_{h=1}^H \mathbf{S}^{(h)})) \in \mathbb{R}^{N \times K}

    \mathbf{X}^{\prime} &= \mathbf{S}^{\top} \mathbf{X} \mathbf{W} \in
    \mathbb{R}^{K \times F^{\prime}}

where :math:`H` denotes the number of heads, and :math:`K` denotes the
number of clusters.

Args:
    in_channels (int): Size of each input sample :math:`F`.
    out_channels (int): Size of each output sample :math:`F^{\prime}`.
    heads (int): The number of heads :math:`H`.
    num_clusters (int): number of clusters :math:`K` per head.
    tau (int, optional): The temperature :math:`\tau`. (default: :obj:`1.`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`kl_loss(S: torch.Tensor) -> torch.Tensor`**
  The additional KL divergence-based loss.

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, max_num_nodes: Optional[int] = None, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`**
  Forward pass.

### `MessageNorm`

Applies message normalization over the aggregated messages as described
in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
<https://arxiv.org/abs/2006.07739>`_ paper.

.. math::

    \mathbf{x}_i^{\prime} = \mathrm{MLP} \left( \mathbf{x}_{i} + s \cdot
    {\| \mathbf{x}_i \|}_2 \cdot
    \frac{\mathbf{m}_{i}}{{\|\mathbf{m}_i\|}_2} \right)

Args:
    learn_scale (bool, optional): If set to :obj:`True`, will learn the
        scaling factor :math:`s` of message normalization.
        (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, msg: torch.Tensor, p: float = 2.0) -> torch.Tensor`**
  Forward pass.

### `MessagePassing`

Base class for creating message passing layers.

Message passing layers follow the form

.. math::
    \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
    \bigoplus_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
    \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{j,i}\right) \right),

where :math:`\bigoplus` denotes a differentiable, permutation invariant
function, *e.g.*, sum, mean, min, max or mul, and
:math:`\gamma_{\mathbf{\Theta}}` and :math:`\phi_{\mathbf{\Theta}}` denote
differentiable functions such as MLPs.
See `here <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/
create_gnn.html>`__ for the accompanying tutorial.

Args:
    aggr (str or [str] or Aggregation, optional): The aggregation scheme
        to use, *e.g.*, :obj:`"sum"` :obj:`"mean"`, :obj:`"min"`,
        :obj:`"max"` or :obj:`"mul"`.
        In addition, can be any
        :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
        that automatically resolves to it).
        If given as a list, will make use of multiple aggregations in which
        different outputs will get concatenated in the last dimension.
        If set to :obj:`None`, the :class:`MessagePassing` instantiation is
        expected to implement its own aggregation logic via
        :meth:`aggregate`. (default: :obj:`"add"`)
    aggr_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective aggregation function in case it gets automatically
        resolved. (default: :obj:`None`)
    flow (str, optional): The flow direction of message passing
        (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
        (default: :obj:`"source_to_target"`)
    node_dim (int, optional): The axis along which to propagate.
        (default: :obj:`-2`)
    decomposed_layers (int, optional): The number of feature decomposition
        layers, as introduced in the `"Optimizing Memory Efficiency of
        Graph Neural Networks on Edge Computing Platforms"
        <https://arxiv.org/abs/2104.03058>`_ paper.
        Feature decomposition reduces the peak memory usage by slicing
        the feature dimensions into separated feature decomposition layers
        during GNN aggregation.
        This method can accelerate GNN execution on CPU-based platforms
        (*e.g.*, 2-3x speedup on the
        :class:`~torch_geometric.datasets.Reddit` dataset) for common GNN
        models such as :class:`~torch_geometric.nn.models.GCN`,
        :class:`~torch_geometric.nn.models.GraphSAGE`,
        :class:`~torch_geometric.nn.models.GIN`, etc.
        However, this method is not applicable to all GNN operators
        available, in particular for operators in which message computation
        can not easily be decomposed, *e.g.* in attention-based GNNs.
        The selection of the optimal value of :obj:`decomposed_layers`
        depends both on the specific graph dataset and available hardware
        resources.
        A value of :obj:`2` is suitable in most cases.
        Although the peak memory usage is directly associated with the
        granularity of feature decomposition, the same is not necessarily
        true for execution speedups. (default: :obj:`1`)

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, *args: Any, **kwargs: Any) -> Any`**
  Runs the forward pass of the module.

- **`propagate(self, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], size: Optional[Tuple[int, int]] = None, **kwargs: Any) -> torch.Tensor`**
  The initial call to start propagating messages.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`aggregate(self, inputs: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None) -> torch.Tensor`**
  Aggregates messages from neighbors as

### `MetaLayer`

A meta layer for building any kind of graph network, inspired by the
`"Relational Inductive Biases, Deep Learning, and Graph Networks"
<https://arxiv.org/abs/1806.01261>`_ paper.

A graph network takes a graph as input and returns an updated graph as
output (with same connectivity).
The input graph has node features :obj:`x`, edge features :obj:`edge_attr`
as well as graph-level features :obj:`u`.
The output graph has the same structure, but updated features.

Edge features, node features as well as global features are updated by
calling the modules :obj:`edge_model`, :obj:`node_model` and
:obj:`global_model`, respectively.

To allow for batch-wise graph processing, all callable functions take an
additional argument :obj:`batch`, which determines the assignment of
edges or nodes to their specific graphs.

Args:
    edge_model (torch.nn.Module, optional): A callable which updates a
        graph's edge features based on its source and target node features,
        its current edge features and its global features.
        (default: :obj:`None`)
    node_model (torch.nn.Module, optional): A callable which updates a
        graph's node features based on its current node features, its graph
        connectivity, its edge features and its global features.
        (default: :obj:`None`)
    global_model (torch.nn.Module, optional): A callable which updates a
        graph's global features based on its node features, its graph
        connectivity, its edge features and its current global features.
        (default: :obj:`None`)

.. code-block:: python

    from torch.nn import Sequential as Seq, Linear as Lin, ReLU
    from torch_geometric.utils import scatter
    from torch_geometric.nn import MetaLayer

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

        def forward(self, src, dst, edge_attr, u, batch):
            # src, dst: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = torch.cat([src, dst, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
            self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

        def forward(self, x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter(out, col, dim=0, dim_size=x.size(0),
                          reduce='mean')
            out = torch.cat([x, out, u[batch]], dim=1)
            return self.node_mlp_2(out)

    class GlobalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

        def forward(self, x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            out = torch.cat([
                u,
                scatter(x, batch, dim=0, reduce='mean'),
            ], dim=1)
            return self.global_mlp(out)

    op = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())
    x, edge_attr, u = op(x, edge_index, edge_attr, u, batch)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, u: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]`**
  Forward pass.

### `MetaPath2Vec`

The MetaPath2Vec model from the `"metapath2vec: Scalable Representation
Learning for Heterogeneous Networks"
<https://ericdongyx.github.io/papers/
KDD17-dong-chawla-swami-metapath2vec.pdf>`_ paper where random walks based
on a given :obj:`metapath` are sampled in a heterogeneous graph, and node
embeddings are learned via negative sampling optimization.

.. note::

    For an example of using MetaPath2Vec, see
    `examples/hetero/metapath2vec.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    hetero/metapath2vec.py>`_.

Args:
    edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): Dictionary
        holding edge indices for each
        :obj:`(src_node_type, rel_type, dst_node_type)` edge type present
        in the heterogeneous graph.
    embedding_dim (int): The size of each embedding vector.
    metapath (List[Tuple[str, str, str]]): The metapath described as a list
        of :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
    walk_length (int): The walk length.
    context_size (int): The actual context size which is considered for
        positive samples. This parameter increases the effective sampling
        rate by reusing samples across different source nodes.
    walks_per_node (int, optional): The number of walks to sample for each
        node. (default: :obj:`1`)
    num_negative_samples (int, optional): The number of negative samples to
        use for each positive sample. (default: :obj:`1`)
    num_nodes_dict (Dict[str, int], optional): Dictionary holding the
        number of nodes for each node type. (default: :obj:`None`)
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
        weight matrix will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, node_type: str, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Returns the embeddings for the nodes in :obj:`batch` of type

- **`loader(self, **kwargs)`**
  Returns the data loader that creates both positive and negative

- **`loss(self, pos_rw: torch.Tensor, neg_rw: torch.Tensor) -> torch.Tensor`**
  Computes the loss given positive and negative random walks.

- **`test(self, train_z: torch.Tensor, train_y: torch.Tensor, test_z: torch.Tensor, test_y: torch.Tensor, solver: str = 'lbfgs', *args, **kwargs) -> float`**
  Evaluates latent space quality via a logistic regression downstream

### `MinAggregation`

An aggregation operator that takes the feature-wise minimum across a
set of elements.

.. math::
    \mathrm{min}(\mathcal{X}) = \min_{\mathbf{x}_i \in \mathcal{X}}
    \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `MixHopConv`

The Mix-Hop graph convolutional operator from the
`"MixHop: Higher-Order Graph Convolutional Architectures via Sparsified
Neighborhood Mixing" <https://arxiv.org/abs/1905.00067>`_ paper.

.. math::
    \mathbf{X}^{\prime}={\Bigg\Vert}_{p\in P}
    {\left( \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2} \right)}^p \mathbf{X} \mathbf{\Theta},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    powers (List[int], optional): The powers of the adjacency matrix to
        use. (default: :obj:`[0, 1, 2]`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:**
      node features :math:`(|\mathcal{V}|, |P| \cdot F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `MulAggregation`

An aggregation operator that multiples features across a set of
elements.

.. math::
    \mathrm{mul}(\mathcal{X}) = \prod_{\mathbf{x}_i \in \mathcal{X}}
    \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `MultiAggregation`

Performs aggregations with one or more aggregators and combines
aggregated results, as described in the `"Principal Neighbourhood
Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ and
`"Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions"
<https://arxiv.org/abs/2104.01481>`_ papers.

Args:
    aggrs (list): The list of aggregation schemes to use.
    aggrs_kwargs (dict, optional): Arguments passed to the
        respective aggregation function in case it gets automatically
        resolved. (default: :obj:`None`)
    mode (str, optional): The combine mode to use for combining
        aggregated results from multiple aggregations (:obj:`"cat"`,
        :obj:`"proj"`, :obj:`"sum"`, :obj:`"mean"`, :obj:`"max"`,
        :obj:`"min"`, :obj:`"logsumexp"`, :obj:`"std"`, :obj:`"var"`,
        :obj:`"attn"`). (default: :obj:`"cat"`)
    mode_kwargs (dict, optional): Arguments passed for the combine
        :obj:`mode`. When :obj:`"proj"` or :obj:`"attn"` is used as the
        combine :obj:`mode`, :obj:`in_channels` (int or tuple) and
        :obj:`out_channels` (int) are needed to be specified respectively
        for the size of each input sample to combine from the respective
        aggregation outputs and the size of each output sample after
        combination. When :obj:`"attn"` mode is used, :obj:`num_heads`
        (int) is needed to be specified for the number of parallel
        attention heads. (default: :obj:`None`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`get_out_channels(self, in_channels: int) -> int`**

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

- **`combine(self, inputs: List[torch.Tensor]) -> torch.Tensor`**

### `NNConv`

The continuous kernel-based convolutional operator from the
`"Neural Message Passing for Quantum Chemistry"
<https://arxiv.org/abs/1704.01212>`_ paper.

This convolution is also known as the edge-conditioned convolution from the
`"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
:class:`torch_geometric.nn.conv.ECConv` for an alias):

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
    \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
    h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
a MLP.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
        maps edge features :obj:`edge_attr` of shape :obj:`[-1,
        num_edge_features]` to shape
        :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`.
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"add"`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add the transformed root node features to the output.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `NeuralFingerprint`

The Neural Fingerprint model from the
`"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
<https://arxiv.org/abs/1509.09292>`__ paper to generate fingerprints
of molecules.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output fingerprint.
    num_layers (int): Number of layers.
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MFConv`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**

### `Node2Vec`

The Node2Vec model from the
`"node2vec: Scalable Feature Learning for Networks"
<https://arxiv.org/abs/1607.00653>`_ paper where random walks of
length :obj:`walk_length` are sampled in a given graph, and node embeddings
are learned via negative sampling optimization.

.. note::

    For an example of using Node2Vec, see `examples/node2vec.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    node2vec.py>`_.

Args:
    edge_index (torch.Tensor): The edge indices.
    embedding_dim (int): The size of each embedding vector.
    walk_length (int): The walk length.
    context_size (int): The actual context size which is considered for
        positive samples. This parameter increases the effective sampling
        rate by reusing samples across different source nodes.
    walks_per_node (int, optional): The number of walks to sample for each
        node. (default: :obj:`1`)
    p (float, optional): Likelihood of immediately revisiting a node in the
        walk. (default: :obj:`1`)
    q (float, optional): Control parameter to interpolate between
        breadth-first strategy and depth-first strategy (default: :obj:`1`)
    num_negative_samples (int, optional): The number of negative samples to
        use for each positive sample. (default: :obj:`1`)
    num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
        weight matrix will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Returns the embeddings for the nodes in :obj:`batch`.

- **`loader(self, **kwargs) -> torch.utils.data.dataloader.DataLoader`**

- **`pos_sample(self, batch: torch.Tensor) -> torch.Tensor`**

- **`neg_sample(self, batch: torch.Tensor) -> torch.Tensor`**

### `PANConv`

The path integral based convolutional operator from the
`"Path Integral Based Convolution and Pooling for Graph Neural Networks"
<https://arxiv.org/abs/2006.16811>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \mathbf{M} \mathbf{X} \mathbf{W}

where :math:`\mathbf{M}` denotes the normalized and learned maximal entropy
transition (MET) matrix that includes neighbors up to :obj:`filter_size`
hops:

.. math::

    \mathbf{M} = \mathbf{Z}^{-1/2} \sum_{n=0}^L e^{-\frac{E(n)}{T}}
    \mathbf{A}^n \mathbf{Z}^{-1/2}

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    filter_size (int): The filter size :math:`L`.
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> Tuple[torch.Tensor, torch_geometric.typing.SparseTensor]`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

- **`panentropy(self, adj_t: torch_geometric.typing.SparseTensor, dtype: Optional[int] = None) -> torch_geometric.typing.SparseTensor`**

### `PANPooling`

The path integral based pooling operator from the
`"Path Integral Based Convolution and Pooling for Graph Neural Networks"
<https://arxiv.org/abs/2006.16811>`_ paper.

PAN pooling performs top-:math:`k` pooling where global node importance is
measured based on node features and the MET matrix:

.. math::
    {\rm score} = \beta_1 \mathbf{X} \cdot \mathbf{p} + \beta_2
    {\rm deg}(\mathbf{M})

Args:
    in_channels (int): Size of each input sample.
    ratio (float): Graph pooling ratio, which is used to compute
        :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
        This value is ignored if min_score is not None.
        (default: :obj:`0.5`)
    min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
        which is used to compute indices of pooled nodes
        :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
        When this value is not :obj:`None`, the :obj:`ratio` argument is
        ignored. (default: :obj:`None`)
    multiplier (float, optional): Coefficient by which features gets
        multiplied after pooling. This can be useful for large graphs and
        when :obj:`min_score` is used. (default: :obj:`1.0`)
    nonlinearity (str or callable, optional): The non-linearity to use.
        (default: :obj:`"tanh"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, M: torch_geometric.typing.SparseTensor, batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]`**
  Forward pass.

### `PDNConv`

The pathfinder discovery network convolutional operator from the
`"Pathfinder Discovery Networks for Neural Message Passing"
<https://arxiv.org/abs/2010.12878>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup
    \{i\}}f_{\Theta}(\textbf{e}_{(j,i)}) \cdot f_{\Omega}(\mathbf{x}_{j})

where :math:`z_{i,j}` denotes the edge feature vector from source node
:math:`j` to target node :math:`i`, and :math:`\mathbf{x}_{j}` denotes the
node feature vector of node :math:`j`.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    edge_dim (int): Edge feature dimensionality.
    hidden_channels (int): Hidden edge feature dimensionality.
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    normalize (bool, optional): Whether to add self-loops and compute
        symmetric normalization coefficients on the fly.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `PMLP`

The P(ropagational)MLP model from the `"Graph Neural Networks are
Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
<https://arxiv.org/abs/2212.09034>`_ paper.
:class:`PMLP` is identical to a standard MLP during training, but then
adopts a GNN architecture during testing.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    out_channels (int): Size of each output sample.
    num_layers (int): The number of layers.
    dropout (float, optional): Dropout probability of each hidden
        embedding. (default: :obj:`0.`)
    norm (bool, optional): If set to :obj:`False`, will not apply batch
        normalization. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the module
        will not learn additive biases. (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor`**

### `PNA`

The Graph Neural Network from the `"Principal Neighbourhood Aggregation
for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper, using the
:class:`~torch_geometric.nn.conv.PNAConv` operator for message passing.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of message passing layers.
    out_channels (int, optional): If not set to :obj:`None`, will apply a
        final linear transformation to convert hidden node embeddings to
        output size :obj:`out_channels`. (default: :obj:`None`)
    dropout (float, optional): Dropout probability. (default: :obj:`0.`)
    act (str or Callable, optional): The non-linear activation function to
        use. (default: :obj:`"relu"`)
    act_first (bool, optional): If set to :obj:`True`, activation is
        applied before normalization. (default: :obj:`False`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    norm (str or Callable, optional): The normalization function to
        use. (default: :obj:`None`)
    norm_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective normalization function defined by :obj:`norm`.
        (default: :obj:`None`)
    jk (str, optional): The Jumping Knowledge mode. If specified, the model
        will additionally apply a final linear transformation to transform
        node embeddings to the expected output feature dimensionality.
        (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
        :obj:`"lstm"`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.PNAConv`.

#### Methods

- **`init_conv(self, in_channels: int, out_channels: int, **kwargs) -> torch_geometric.nn.conv.message_passing.MessagePassing`**

### `PNAConv`

The Principal Neighbourhood Aggregation graph convolution operator
from the `"Principal Neighbourhood Aggregation for Graph Nets"
<https://arxiv.org/abs/2004.05718>`_ paper.

.. math::
    \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left(
    \mathbf{x}_i, \underset{j \in \mathcal{N}(i)}{\bigoplus}
    h_{\mathbf{\Theta}} \left( \mathbf{x}_i, \mathbf{x}_j \right)
    \right)

with

.. math::
    \bigoplus = \underbrace{\begin{bmatrix}
        1 \\
        S(\mathbf{D}, \alpha=1) \\
        S(\mathbf{D}, \alpha=-1)
    \end{bmatrix} }_{\text{scalers}}
    \otimes \underbrace{\begin{bmatrix}
        \mu \\
        \sigma \\
        \max \\
        \min
    \end{bmatrix}}_{\text{aggregators}},

where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
denote MLPs.

.. note::

    For an example of using :obj:`PNAConv`, see `examples/pna.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/
    examples/pna.py>`_.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    aggregators (List[str]): Set of aggregation function identifiers,
        namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"var"` and :obj:`"std"`.
    scalers (List[str]): Set of scaling function identifiers, namely
        :obj:`"identity"`, :obj:`"amplification"`,
        :obj:`"attenuation"`, :obj:`"linear"` and
        :obj:`"inverse_linear"`.
    deg (torch.Tensor): Histogram of in-degrees of nodes in the training
        set, used by scalers to normalize.
    edge_dim (int, optional): Edge feature dimensionality (in case
        there are any). (default :obj:`None`)
    towers (int, optional): Number of towers (default: :obj:`1`).
    pre_layers (int, optional): Number of transformation layers before
        aggregation (default: :obj:`1`).
    post_layers (int, optional): Number of transformation layers after
        aggregation (default: :obj:`1`).
    divide_input (bool, optional): Whether the input features should
        be split between towers or not (default: :obj:`False`).
    act (str or callable, optional): Pre- and post-layer activation
        function to use. (default: :obj:`"relu"`)
    act_kwargs (Dict[str, Any], optional): Arguments passed to the
        respective activation function defined by :obj:`act`.
        (default: :obj:`None`)
    train_norm (bool, optional): Whether normalization parameters
        are trainable. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge features :math:`(|\mathcal{E}|, D)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`get_degree_histogram(loader: torch.utils.data.dataloader.DataLoader) -> torch.Tensor`**
  Returns the degree histogram to be used as input for the :obj:`deg`

### `PPFConv`

The PPFNet operator from the `"PPFNet: Global Context Aware Local
Features for Robust 3D Point Matching" <https://arxiv.org/abs/1802.02669>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
    \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j, \|
    \mathbf{d_{j,i}} \|, \angle(\mathbf{n}_i, \mathbf{d_{j,i}}),
    \angle(\mathbf{n}_j, \mathbf{d_{j,i}}), \angle(\mathbf{n}_i,
    \mathbf{n}_j) \right)

where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
denote neural networks, *.i.e.* MLPs, which takes in node features and
:class:`torch_geometric.transforms.PointPairFeatures`.

Args:
    local_nn (torch.nn.Module, optional): A neural network
        :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
        relative spatial coordinates :obj:`pos_j - pos_i` of shape
        :obj:`[-1, in_channels + num_dimensions]` to shape
        :obj:`[-1, out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)
    global_nn (torch.nn.Module, optional): A neural network
        :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
        of shape :obj:`[-1, out_channels]` to shape :obj:`[-1,
        final_out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      positions :math:`(|\mathcal{V}|, 3)` or
      :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
      point normals :math:`(|\mathcal{V}, 3)` or
      :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, NoneType, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]], pos: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], normal: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: Optional[torch.Tensor], pos_i: torch.Tensor, pos_j: torch.Tensor, normal_i: torch.Tensor, normal_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `PairNorm`

Applies pair normalization over node features as described in the
`"PairNorm: Tackling Oversmoothing in GNNs"
<https://arxiv.org/abs/1909.12223>`_ paper.

.. math::
    \mathbf{x}_i^c &= \mathbf{x}_i - \frac{1}{n}
    \sum_{i=1}^n \mathbf{x}_i \\

    \mathbf{x}_i^{\prime} &= s \cdot
    \frac{\mathbf{x}_i^c}{\sqrt{\frac{1}{n} \sum_{i=1}^n
    {\| \mathbf{x}_i^c \|}^2_2}}

Args:
    scale (float, optional): Scaling factor :math:`s` of normalization.
        (default, :obj:`1.`)
    scale_individually (bool, optional): If set to :obj:`True`, will
        compute the scaling step as :math:`\mathbf{x}^{\prime}_i = s \cdot
        \frac{\mathbf{x}_i^c}{{\| \mathbf{x}_i^c \|}_2}`.
        (default: :obj:`False`)
    eps (float, optional): A value added to the denominator for numerical
        stability. (default: :obj:`1e-5`)

#### Methods

- **`forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `PatchTransformerAggregation`

Performs patch transformer aggregation in which the elements to
aggregate are processed by multi-head attention blocks across patches, as
described in the `"Simplifying Temporal Heterogeneous Network for
Continuous-Time Link Prediction"
<https://dl.acm.org/doi/pdf/10.1145/3583780.3615059>`_ paper.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    patch_size (int): Number of elements in a patch.
    hidden_channels (int): Intermediate size of each sample.
    num_transformer_blocks (int, optional): Number of transformer blocks
        (default: :obj:`1`).
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    dropout (float, optional): Dropout probability of attention weights.
        (default: :obj:`0.0`)
    aggr (str or list[str], optional): The aggregation module, *e.g.*,
        :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
        :obj:`"var"`, :obj:`"std"`. (default: :obj:`"mean"`)

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `PointGNNConv`

The PointGNN operator from the `"Point-GNN: Graph Neural Network for
3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
paper.

.. math::

    \Delta \textrm{pos}_i &= h_{\mathbf{\Theta}}(\mathbf{x}_i)

    \mathbf{e}_{j,i} &= f_{\mathbf{\Theta}}(\textrm{pos}_j -
    \textrm{pos}_i + \Delta \textrm{pos}_i, \mathbf{x}_j)

    \mathbf{x}^{\prime}_i &= g_{\mathbf{\Theta}}(\max_{j \in
    \mathcal{N}(i)} \mathbf{e}_{j,i}) + \mathbf{x}_i

The relative position is used in the message passing step to introduce
global translation invariance.
To also counter shifts in the local neighborhood of the center node, the
authors propose to utilize an alignment offset.
The graph should be statically constructed using radius-based cutoff.

Args:
    mlp_h (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}`
        that maps node features of size :math:`F_{in}` to three-dimensional
        coordination offsets :math:`\Delta \textrm{pos}_i`.
    mlp_f (torch.nn.Module): A neural network :math:`f_{\mathbf{\Theta}}`
        that computes :math:`\mathbf{e}_{j,i}` from the features of
        neighbors of size :math:`F_{in}` and the three-dimensional vector
        :math:`\textrm{pos_j} - \textrm{pos_i} + \Delta \textrm{pos}_i`.
    mlp_g (torch.nn.Module): A neural network :math:`g_{\mathbf{\Theta}}`
        that maps the aggregated edge features back to :math:`F_{in}`.
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      positions :math:`(|\mathcal{V}|, 3)`,
      edge indices :math:`(2, |\mathcal{E}|)`,
    - **output:** node features :math:`(|\mathcal{V}|, F_{in})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, pos_j: torch.Tensor, pos_i: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `PointNetConv`

The PointNet set layer from the `"PointNet: Deep Learning on Point Sets
for 3D Classification and Segmentation"
<https://arxiv.org/abs/1612.00593>`_ and `"PointNet++: Deep Hierarchical
Feature Learning on Point Sets in a Metric Space"
<https://arxiv.org/abs/1706.02413>`_ papers.

.. math::
    \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \max_{j \in
    \mathcal{N}(i) \cup \{ i \}} h_{\mathbf{\Theta}} ( \mathbf{x}_j,
    \mathbf{p}_j - \mathbf{p}_i) \right),

where :math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}`
denote neural networks, *i.e.* MLPs, and
:math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
each point.

Args:
    local_nn (torch.nn.Module, optional): A neural network
        :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` and
        relative spatial coordinates :obj:`pos_j - pos_i` of shape
        :obj:`[-1, in_channels + num_dimensions]` to shape
        :obj:`[-1, out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)
    global_nn (torch.nn.Module, optional): A neural network
        :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
        of shape :obj:`[-1, out_channels]` to shape :obj:`[-1,
        final_out_channels]`, *e.g.*, defined by
        :class:`torch.nn.Sequential`. (default: :obj:`None`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      positions :math:`(|\mathcal{V}|, 3)` or
      :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, NoneType, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]], pos: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: Optional[torch.Tensor], pos_i: torch.Tensor, pos_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `PointTransformerConv`

The Point Transformer layer from the `"Point Transformer"
<https://arxiv.org/abs/2012.09164>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i =  \sum_{j \in
    \mathcal{N}(i) \cup \{ i \}} \alpha_{i,j} \left(\mathbf{W}_3
    \mathbf{x}_j + \delta_{ij} \right),

where the attention coefficients :math:`\alpha_{i,j}` and
positional embedding :math:`\delta_{ij}` are computed as

.. math::
    \alpha_{i,j}= \textrm{softmax} \left( \gamma_\mathbf{\Theta}
    (\mathbf{W}_1 \mathbf{x}_i - \mathbf{W}_2 \mathbf{x}_j +
    \delta_{i,j}) \right)

and

.. math::
    \delta_{i,j}= h_{\mathbf{\Theta}}(\mathbf{p}_i - \mathbf{p}_j),

with :math:`\gamma_\mathbf{\Theta}` and :math:`h_\mathbf{\Theta}`
denoting neural networks, *i.e.* MLPs, and
:math:`\mathbf{P} \in \mathbb{R}^{N \times D}` defines the position of
each point.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    pos_nn (torch.nn.Module, optional): A neural network
        :math:`h_\mathbf{\Theta}` which maps relative spatial coordinates
        :obj:`pos_j - pos_i` of shape :obj:`[-1, 3]` to shape
        :obj:`[-1, out_channels]`.
        Will default to a :class:`torch.nn.Linear` transformation if not
        further specified. (default: :obj:`None`)
    attn_nn (torch.nn.Module, optional): A neural network
        :math:`\gamma_\mathbf{\Theta}` which maps transformed
        node features of shape :obj:`[-1, out_channels]`
        to shape :obj:`[-1, out_channels]`. (default: :obj:`None`)
    add_self_loops (bool, optional) : If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      positions :math:`(|\mathcal{V}|, 3)` or
      :math:`((|\mathcal{V_s}|, 3), (|\mathcal{V_t}|, 3))` if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V}_t|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], pos: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, pos_i: torch.Tensor, pos_j: torch.Tensor, alpha_i: torch.Tensor, alpha_j: torch.Tensor, index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `PositionalEncoding`

The positional encoding scheme from the `"Attention Is All You Need"
<https://arxiv.org/abs/1706.03762>`_ paper.

.. math::

    PE(x)_{2 \cdot i} &= \sin(x / 10000^{2 \cdot i / d})

    PE(x)_{2 \cdot i + 1} &= \cos(x / 10000^{2 \cdot i / d})

where :math:`x` is the position and :math:`i` is the dimension.

Args:
    out_channels (int): Size :math:`d` of each output sample.
    base_freq (float, optional): The base frequency of sinusoidal
        functions. (default: :obj:`1e-4`)
    granularity (float, optional): The granularity of the positions. If
        set to smaller value, the encoder will capture more fine-grained
        changes in positions. (default: :obj:`1.0`)

#### Methods

- **`reset_parameters(self)`**

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `PowerMeanAggregation`

The powermean aggregation operator based on a power term, as
described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
<https://arxiv.org/abs/2006.07739>`_ paper.

.. math::
    \mathrm{powermean}(\mathcal{X}|p) = \left(\frac{1}{|\mathcal{X}|}
    \sum_{\mathbf{x}_i\in\mathcal{X}}\mathbf{x}_i^{p}\right)^{1/p},

where :math:`p` controls the power of the powermean when aggregating over
a set of features :math:`\mathcal{X}`.

Args:
    p (float, optional): Initial power for powermean aggregation.
        (default: :obj:`1.0`)
    learn (bool, optional): If set to :obj:`True`, will learn the value
        :obj:`p` for powermean aggregation dynamically.
        (default: :obj:`False`)
    channels (int, optional): Number of channels to learn from :math:`p`.
        If set to a value greater than :obj:`1`, :math:`p` will be learned
        per input feature channel. This requires compatible shapes for the
        input to the forward calculation. (default: :obj:`1`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `QuantileAggregation`

An aggregation operator that returns the feature-wise :math:`q`-th
quantile of a set :math:`\mathcal{X}`.

That is, for every feature :math:`d`, it computes

.. math::
    {\mathrm{Q}_q(\mathcal{X})}_d = \begin{cases}
        x_{\pi_i,d} & i = q \cdot n, \\
        f(x_{\pi_i,d}, x_{\pi_{i+1},d}) & i < q \cdot n < i + 1,\\
    \end{cases}

where :math:`x_{\pi_1,d} \le \dots \le x_{\pi_i,d} \le \dots \le
x_{\pi_n,d}` and :math:`f(a, b)` is an interpolation
function defined by :obj:`interpolation`.

Args:
    q (float or list): The quantile value(s) :math:`q`. Can be a scalar or
        a list of scalars in the range :math:`[0, 1]`. If more than a
        quantile is passed, the results are concatenated.
    interpolation (str): Interpolation method applied if the quantile point
        :math:`q\cdot n` lies between two values
        :math:`a \le b`. Can be one of the following:

        * :obj:`"lower"`: Returns the one with lowest value.

        * :obj:`"higher"`: Returns the one with highest value.

        * :obj:`"midpoint"`: Returns the average of the two values.

        * :obj:`"nearest"`: Returns the one whose index is nearest to the
          quantile point.

        * :obj:`"linear"`: Returns a linear combination of the two
          elements, defined as
          :math:`f(a, b) = a + (b - a)\cdot(q\cdot n - i)`.

        (default: :obj:`"linear"`)
    fill_value (float, optional): The default value in the case no entry is
        found for a given index (default: :obj:`0.0`).

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `RECT_L`

The RECT model, *i.e.* its supervised RECT-L part, from the
`"Network Embedding with Completely-imbalanced Labels"
<https://arxiv.org/abs/2007.03545>`_ paper.
In particular, a GCN model is trained that reconstructs semantic class
knowledge.

.. note::

    For an example of using RECT, see `examples/rect.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    rect.py>`_.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Intermediate size of each sample.
    normalize (bool, optional): Whether to add self-loops and compute
        symmetric normalization coefficients on-the-fly.
        (default: :obj:`True`)
    dropout (float, optional): The dropout probability.
        (default: :obj:`0.0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**

- **`embed(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**

- **`get_semantic_labels(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`**
  Replaces the original labels by their class-centers.

### `RENet`

The Recurrent Event Network model from the `"Recurrent Event Network
for Reasoning over Temporal Knowledge Graphs"
<https://arxiv.org/abs/1904.05530>`_ paper.

.. math::
    f_{\mathbf{\Theta}}(\mathbf{e}_s, \mathbf{e}_r,
    \mathbf{h}^{(t-1)}(s, r))

based on a RNN encoder

.. math::
    \mathbf{h}^{(t)}(s, r) = \textrm{RNN}(\mathbf{e}_s, \mathbf{e}_r,
    g(\mathcal{O}^{(t)}_r(s)), \mathbf{h}^{(t-1)}(s, r))

where :math:`\mathbf{e}_s` and :math:`\mathbf{e}_r` denote entity and
relation embeddings, and :math:`\mathcal{O}^{(t)}_r(s)` represents the set
of objects interacted with subject :math:`s` under relation :math:`r` at
timestamp :math:`t`.
This model implements :math:`g` as the **Mean Aggregator** and
:math:`f_{\mathbf{\Theta}}` as a linear projection.

Args:
    num_nodes (int): The number of nodes in the knowledge graph.
    num_rels (int): The number of relations in the knowledge graph.
    hidden_channels (int): Hidden size of node and relation embeddings.
    seq_len (int): The sequence length of past events.
    num_layers (int, optional): The number of recurrent layers.
        (default: :obj:`1`)
    dropout (float): If non-zero, introduces a dropout layer before the
        final prediction. (default: :obj:`0.`)
    bias (bool, optional): If set to :obj:`False`, all layers will not
        learn an additive bias. (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**

- **`pre_transform(seq_len: int) -> Callable`**
  Precomputes history objects.

- **`forward(self, data: torch_geometric.data.data.Data) -> Tuple[torch.Tensor, torch.Tensor]`**
  Given a :obj:`data` batch, computes the forward pass.

- **`test(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor`**
  Given ground-truth :obj:`y`, computes Mean Reciprocal Rank (MRR)

### `RGATConv`

The relational graph attentional operator from the `"Relational Graph
Attention Networks" <https://arxiv.org/abs/1904.05811>`_ paper.

Here, attention logits :math:`\mathbf{a}^{(r)}_{i,j}` are computed for each
relation type :math:`r` with the help of both query and key kernels, *i.e.*

.. math::
    \mathbf{q}^{(r)}_i = \mathbf{W}_1^{(r)}\mathbf{x}_{i} \cdot
    \mathbf{Q}^{(r)}
    \quad \textrm{and} \quad
    \mathbf{k}^{(r)}_i = \mathbf{W}_1^{(r)}\mathbf{x}_{i} \cdot
    \mathbf{K}^{(r)}.

Two schemes have been proposed to compute attention logits
:math:`\mathbf{a}^{(r)}_{i,j}` for each relation type :math:`r`:

**Additive attention**

.. math::
    \mathbf{a}^{(r)}_{i,j} = \mathrm{LeakyReLU}(\mathbf{q}^{(r)}_i +
    \mathbf{k}^{(r)}_j)

or **multiplicative attention**

.. math::
    \mathbf{a}^{(r)}_{i,j} = \mathbf{q}^{(r)}_i \cdot \mathbf{k}^{(r)}_j.

If the graph has multi-dimensional edge features
:math:`\mathbf{e}^{(r)}_{i,j}`, the attention logits
:math:`\mathbf{a}^{(r)}_{i,j}` for each relation type :math:`r` are
computed as

.. math::
    \mathbf{a}^{(r)}_{i,j} = \mathrm{LeakyReLU}(\mathbf{q}^{(r)}_i +
    \mathbf{k}^{(r)}_j + \mathbf{W}_2^{(r)}\mathbf{e}^{(r)}_{i,j})

or

.. math::
    \mathbf{a}^{(r)}_{i,j} = \mathbf{q}^{(r)}_i \cdot \mathbf{k}^{(r)}_j
    \cdot \mathbf{W}_2^{(r)} \mathbf{e}^{(r)}_{i,j},

respectively.
The attention coefficients :math:`\alpha^{(r)}_{i,j}` for each relation
type :math:`r` are then obtained via two different attention mechanisms:
The **within-relation** attention mechanism

.. math::
    \alpha^{(r)}_{i,j} =
    \frac{\exp(\mathbf{a}^{(r)}_{i,j})}
    {\sum_{k \in \mathcal{N}_r(i)} \exp(\mathbf{a}^{(r)}_{i,k})}

or the **across-relation** attention mechanism

.. math::
    \alpha^{(r)}_{i,j} =
    \frac{\exp(\mathbf{a}^{(r)}_{i,j})}
    {\sum_{r^{\prime} \in \mathcal{R}}
    \sum_{k \in \mathcal{N}_{r^{\prime}}(i)}
    \exp(\mathbf{a}^{(r^{\prime})}_{i,k})}

where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
stores a relation identifier :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}`
for each edge.

To enhance the discriminative power of attention-based GNNs, this layer
further implements four different cardinality preservation options as
proposed in the `"Improving Attention Mechanism in Graph Neural Networks
via Cardinality Preservation" <https://arxiv.org/abs/1907.02204>`_ paper:

.. math::
    \text{additive:}~~~\mathbf{x}^{{\prime}(r)}_i &=
    \sum_{j \in \mathcal{N}_r(i)}
    \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j + \mathcal{W} \odot
    \sum_{j \in \mathcal{N}_r(i)} \mathbf{x}^{(r)}_j

    \text{scaled:}~~~\mathbf{x}^{{\prime}(r)}_i &=
    \psi(|\mathcal{N}_r(i)|) \odot
    \sum_{j \in \mathcal{N}_r(i)} \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j

    \text{f-additive:}~~~\mathbf{x}^{{\prime}(r)}_i &=
    \sum_{j \in \mathcal{N}_r(i)}
    (\alpha^{(r)}_{i,j} + 1) \cdot \mathbf{x}^{(r)}_j

    \text{f-scaled:}~~~\mathbf{x}^{{\prime}(r)}_i &=
    |\mathcal{N}_r(i)| \odot \sum_{j \in \mathcal{N}_r(i)}
    \alpha^{(r)}_{i,j} \mathbf{x}^{(r)}_j

* If :obj:`attention_mode="additive-self-attention"` and
  :obj:`concat=True`, the layer outputs :obj:`heads * out_channels`
  features for each node.

* If :obj:`attention_mode="multiplicative-self-attention"` and
  :obj:`concat=True`, the layer outputs :obj:`heads * dim * out_channels`
  features for each node.

* If :obj:`attention_mode="additive-self-attention"` and
  :obj:`concat=False`, the layer outputs :obj:`out_channels` features for
  each node.

* If :obj:`attention_mode="multiplicative-self-attention"` and
  :obj:`concat=False`, the layer outputs :obj:`dim * out_channels` features
  for each node.

Please make sure to set the :obj:`in_channels` argument of the next
layer accordingly if more than one instance of this layer is used.

.. note::

    For an example of using :class:`RGATConv`, see
    `examples/rgat.py <https://github.com/pyg-team/pytorch_geometric/blob
    /master/examples/rgat.py>`_.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    num_relations (int): Number of relations.
    num_bases (int, optional): If set, this layer will use the
        basis-decomposition regularization scheme where :obj:`num_bases`
        denotes the number of bases to use. (default: :obj:`None`)
    num_blocks (int, optional): If set, this layer will use the
        block-diagonal-decomposition regularization scheme where
        :obj:`num_blocks` denotes the number of blocks to use.
        (default: :obj:`None`)
    mod (str, optional): The cardinality preservation option to use.
        (:obj:`"additive"`, :obj:`"scaled"`, :obj:`"f-additive"`,
        :obj:`"f-scaled"`, :obj:`None`). (default: :obj:`None`)
    attention_mechanism (str, optional): The attention mechanism to use
        (:obj:`"within-relation"`, :obj:`"across-relation"`).
        (default: :obj:`"across-relation"`)
    attention_mode (str, optional): The mode to calculate attention logits.
        (:obj:`"additive-self-attention"`,
        :obj:`"multiplicative-self-attention"`).
        (default: :obj:`"additive-self-attention"`)
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    dim (int): Number of dimensions for query and key kernels.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    edge_dim (int, optional): Edge feature dimensionality (in case there
        are any). (default: :obj:`None`)
    bias (bool, optional): If set to :obj:`False`, the layer will not
        learn an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_type: Optional[torch.Tensor] = None, edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None, return_attention_weights=None)`**
  Runs the forward pass of the module.

- **`message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_type: torch.Tensor, edge_attr: Optional[torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`update(self, aggr_out: torch.Tensor) -> torch.Tensor`**
  Updates node embeddings in analogy to

### `RGCNConv`

The relational graph convolutional operator from the `"Modeling
Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
    \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
    \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
stores a relation identifier
:math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

.. note::
    This implementation is as memory-efficient as possible by iterating
    over each individual relation type.
    Therefore, it may result in low GPU utilization in case the graph has a
    large number of relations.
    As an alternative approach, :class:`FastRGCNConv` does not iterate over
    each individual type, but may consume a large amount of memory to
    compensate.
    We advise to check out both implementations to see which one fits your
    needs.

.. note::
    :class:`RGCNConv` can use `dynamic shapes
    <https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index
    .html#work_dynamic_shapes>`_, which means that the shape of the interim
    tensors can be determined at runtime.
    If your device doesn't support dynamic shapes, use
    :class:`FastRGCNConv` instead.

Args:
    in_channels (int or tuple): Size of each input sample. A tuple
        corresponds to the sizes of source and target dimensionalities.
        In case no input features are given, this argument should
        correspond to the number of nodes in your graph.
    out_channels (int): Size of each output sample.
    num_relations (int): Number of relations.
    num_bases (int, optional): If set, this layer will use the
        basis-decomposition regularization scheme where :obj:`num_bases`
        denotes the number of bases to use. (default: :obj:`None`)
    num_blocks (int, optional): If set, this layer will use the
        block-diagonal-decomposition regularization scheme where
        :obj:`num_blocks` denotes the number of blocks to use.
        (default: :obj:`None`)
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"mean"`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    is_sorted (bool, optional): If set to :obj:`True`, assumes that
        :obj:`edge_index` is sorted by :obj:`edge_type`. This avoids
        internal re-sorting of the data and can improve runtime and memory
        efficiency. (default: :obj:`False`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, NoneType, Tuple[Optional[torch.Tensor], torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_type: Optional[torch.Tensor] = None)`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_type_ptr: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `ResGatedGraphConv`

The residual gated graph convolutional operator from the
`"Residual Gated Graph ConvNets" <https://arxiv.org/abs/1711.07553>`_
paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
    \sum_{j \in \mathcal{N}(i)} \eta_{i,j} \odot \mathbf{W}_2 \mathbf{x}_j

where the gate :math:`\eta_{i,j}` is defined as

.. math::
    \eta_{i,j} = \sigma(\mathbf{W}_3 \mathbf{x}_i + \mathbf{W}_4
    \mathbf{x}_j)

with :math:`\sigma` denoting the sigmoid function.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    act (callable, optional): Gating function :math:`\sigma`.
        (default: :meth:`torch.nn.Sigmoid()`)
    edge_dim (int, optional): Edge feature dimensionality (in case
        there are any). (default: :obj:`None`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **inputs:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, k_i: torch.Tensor, q_j: torch.Tensor, v_j: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `Reshape`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

#### Methods

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `RotatE`

The RotatE model from the `"RotatE: Knowledge Graph Embedding by
Relational Rotation in Complex Space" <https://arxiv.org/abs/
1902.10197>`_ paper.

:class:`RotatE` models relations as a rotation in complex space
from head to tail such that

.. math::
    \mathbf{e}_t = \mathbf{e}_h \circ \mathbf{e}_r,

resulting in the scoring function

.. math::
    d(h, r, t) = - {\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|}_p

.. note::

    For an example of using the :class:`RotatE` model, see
    `examples/kge_fb15k_237.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    kge_fb15k_237.py>`_.

Args:
    num_nodes (int): The number of nodes/entities in the graph.
    num_relations (int): The number of relations in the graph.
    hidden_channels (int): The hidden embedding size.
    margin (float, optional): The margin of the ranking loss.
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
        the embedding matrices will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the score for the given triplet.

- **`loss(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the loss value for the given triplet.

### `SAGEConv`

The GraphSAGE operator from the `"Inductive Representation Learning on
Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
    \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
projected via

.. math::
    \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
    \mathbf{b})

as described in Eq. (3) of the paper.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    aggr (str or Aggregation, optional): The aggregation scheme to use.
        Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
        *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
        (default: :obj:`"mean"`)
    normalize (bool, optional): If set to :obj:`True`, output features
        will be :math:`\ell_2`-normalized, *i.e.*,
        :math:`\frac{\mathbf{x}^{\prime}_i}
        {\| \mathbf{x}^{\prime}_i \|_2}`.
        (default: :obj:`False`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    project (bool, optional): If set to :obj:`True`, the layer will apply a
        linear transformation followed by an activation function before
        aggregation (as described in Eq. (3) of the paper).
        (default: :obj:`False`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **inputs:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `SAGPooling`

The self-attention pooling operator from the `"Self-Attention Graph
Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
Attention and Generalization in Graph Neural Networks"
<https://arxiv.org/abs/1905.02850>`_ papers.

If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

    .. math::
        \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
computes:

    .. math::
        \mathbf{y} &= \mathrm{softmax}(\textrm{GNN}(\mathbf{X},\mathbf{A}))

        \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}.

Projections scores are learned based on a graph neural network layer.

Args:
    in_channels (int): Size of each input sample.
    ratio (float or int): Graph pooling ratio, which is used to compute
        :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
        of :math:`k` itself, depending on whether the type of :obj:`ratio`
        is :obj:`float` or :obj:`int`.
        This value is ignored if :obj:`min_score` is not :obj:`None`.
        (default: :obj:`0.5`)
    GNN (torch.nn.Module, optional): A graph neural network layer for
        calculating projection scores (one of
        :class:`torch_geometric.nn.conv.GraphConv`,
        :class:`torch_geometric.nn.conv.GCNConv`,
        :class:`torch_geometric.nn.conv.GATConv` or
        :class:`torch_geometric.nn.conv.SAGEConv`). (default:
        :class:`torch_geometric.nn.conv.GraphConv`)
    min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
        which is used to compute indices of pooled nodes
        :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
        When this value is not :obj:`None`, the :obj:`ratio` argument is
        ignored. (default: :obj:`None`)
    multiplier (float, optional): Coefficient by which features gets
        multiplied after pooling. This can be useful for large graphs and
        when :obj:`min_score` is used. (default: :obj:`1`)
    nonlinearity (str or callable, optional): The non-linearity to use.
        (default: :obj:`"tanh"`)
    **kwargs (optional): Additional parameters for initializing the graph
        neural network layer.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None, attn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]`**
  Forward pass.

### `SGConv`

The simple graph convolutional operator from the `"Simplifying Graph
Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

.. math::
    \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
        first execution, and will use the cached version for further
        executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:**
      node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `SSGConv`

The simple spectral graph convolutional operator from the
`"Simple Spectral Graph Convolution"
<https://openreview.net/forum?id=CYO5T-YjWZV>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \frac{1}{K} \sum_{k=1}^K\left((1-\alpha)
    {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-1/2} \right)}^k
    \mathbf{X}+\alpha \mathbf{X}\right) \mathbf{\Theta},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
adjacency matrix with inserted self-loops and
:math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.
:class:`~torch_geometric.nn.conv.SSGConv` is an improved operator of
:class:`~torch_geometric.nn.conv.SGConv` by introducing the :obj:`alpha`
parameter to address the oversmoothing issue.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    alpha (float): Teleport probability :math:`\alpha \in [0, 1]`.
    K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
    cached (bool, optional): If set to :obj:`True`, the layer will cache
        the computation of :math:`\frac{1}{K} \sum_{k=1}^K\left((1-\alpha)
        {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^k \mathbf{X}+
        \alpha \mathbf{X}\right)` on first execution, and will use the
        cached version for further executions.
        This parameter should only be set to :obj:`True` in transductive
        learning scenarios. (default: :obj:`False`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:**
      node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `SchNet`

The continuous-filter convolutional neural network SchNet from the
`"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
the interactions blocks of the form.

.. math::
    \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
    h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
:math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

.. note::

    For an example of using a pretrained SchNet variant, see
    `examples/qm9_pretrained_schnet.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    qm9_pretrained_schnet.py>`_.

Args:
    hidden_channels (int, optional): Hidden embedding size.
        (default: :obj:`128`)
    num_filters (int, optional): The number of filters to use.
        (default: :obj:`128`)
    num_interactions (int, optional): The number of interaction blocks.
        (default: :obj:`6`)
    num_gaussians (int, optional): The number of gaussians :math:`\mu`.
        (default: :obj:`50`)
    interaction_graph (callable, optional): The function used to compute
        the pairwise interaction graph and interatomic distances. If set to
        :obj:`None`, will construct a graph based on :obj:`cutoff` and
        :obj:`max_num_neighbors` properties.
        If provided, this method takes in :obj:`pos` and :obj:`batch`
        tensors and should return :obj:`(edge_index, edge_weight)` tensors.
        (default :obj:`None`)
    cutoff (float, optional): Cutoff distance for interatomic interactions.
        (default: :obj:`10.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors to
        collect for each node within the :attr:`cutoff` distance.
        (default: :obj:`32`)
    readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
        global aggregation. (default: :obj:`"add"`)
    dipole (bool, optional): If set to :obj:`True`, will use the magnitude
        of the dipole moment to make the final prediction, *e.g.*, for
        target 0 of :class:`torch_geometric.datasets.QM9`.
        (default: :obj:`False`)
    mean (float, optional): The mean of the property to predict.
        (default: :obj:`None`)
    std (float, optional): The standard deviation of the property to
        predict. (default: :obj:`None`)
    atomref (torch.Tensor, optional): The reference of single-atom
        properties.
        Expects a vector of shape :obj:`(max_atomic_number, )`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`from_qm9_pretrained(root: str, dataset: torch_geometric.data.dataset.Dataset, target: int) -> Tuple[ForwardRef('SchNet'), torch_geometric.data.dataset.Dataset, torch_geometric.data.dataset.Dataset, torch_geometric.data.dataset.Dataset]`**
  Returns a pre-trained :class:`SchNet` model on the

- **`forward(self, z: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Forward pass.

### `Sequential`

An extension of the :class:`torch.nn.Sequential` container in order to
define a sequential GNN model.

Since GNN operators take in multiple input arguments,
:class:`torch_geometric.nn.Sequential` additionally expects both global
input arguments, and function header definitions of individual operators.
If omitted, an intermediate module will operate on the *output* of its
preceding module:

.. code-block:: python

    from torch.nn import Linear, ReLU
    from torch_geometric.nn import Sequential, GCNConv

    model = Sequential('x, edge_index', [
        (GCNConv(in_channels, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        Linear(64, out_channels),
    ])

Here, :obj:`'x, edge_index'` defines the input arguments of :obj:`model`,
and :obj:`'x, edge_index -> x'` defines the function header, *i.e.* input
arguments *and* return types of :class:`~torch_geometric.nn.conv.GCNConv`.

In particular, this also allows to create more sophisticated models,
such as utilizing :class:`~torch_geometric.nn.models.JumpingKnowledge`:

.. code-block:: python

    from torch.nn import Linear, ReLU, Dropout
    from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
    from torch_geometric.nn import global_mean_pool

    model = Sequential('x, edge_index, batch', [
        (Dropout(p=0.5), 'x -> x'),
        (GCNConv(dataset.num_features, 64), 'x, edge_index -> x1'),
        ReLU(inplace=True),
        (GCNConv(64, 64), 'x1, edge_index -> x2'),
        ReLU(inplace=True),
        (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
        (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
        (global_mean_pool, 'x, batch -> x'),
        Linear(2 * 64, dataset.num_classes),
    ])

Args:
    input_args (str): The input arguments of the model.
    modules ([(Callable, str) or Callable]): A list of modules (with
        optional function header definitions). Alternatively, an
        :obj:`OrderedDict` of modules (and function header definitions) can
        be passed.

#### Methods

- **`reset_parameters(self) -> None`**
  Resets all learnable parameters of the module.

- **`forward(self, *args: Any, **kwargs: Any) -> Any`**

### `Set2Set`

The Set2Set aggregation operator based on iterative content-based
attention, as described in the `"Order Matters: Sequence to sequence for
Sets" <https://arxiv.org/abs/1511.06391>`_ paper.

.. math::
    \mathbf{q}_t &= \mathrm{LSTM}(\mathbf{q}^{*}_{t-1})

    \alpha_{i,t} &= \mathrm{softmax}(\mathbf{x}_i \cdot \mathbf{q}_t)

    \mathbf{r}_t &= \sum_{i=1}^N \alpha_{i,t} \mathbf{x}_i

    \mathbf{q}^{*}_t &= \mathbf{q}_t \, \Vert \, \mathbf{r}_t,

where :math:`\mathbf{q}^{*}_T` defines the output of the layer with twice
the dimensionality as the input.

Args:
    in_channels (int): Size of each input sample.
    processing_steps (int): Number of iterations :math:`T`.
    **kwargs (optional): Additional arguments of :class:`torch.nn.LSTM`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `SetTransformerAggregation`

Performs "Set Transformer" aggregation in which the elements to
aggregate are processed by multi-head attention blocks, as described in
the `"Graph Neural Networks with Adaptive Readouts"
<https://arxiv.org/abs/2211.04952>`_ paper.

.. note::

    :class:`SetTransformerAggregation` requires sorted indices :obj:`index`
    as input. Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

Args:
    channels (int): Size of each input sample.
    num_seed_points (int, optional): Number of seed points.
        (default: :obj:`1`)
    num_encoder_blocks (int, optional): Number of Set Attention Blocks
        (SABs) in the encoder. (default: :obj:`1`).
    num_decoder_blocks (int, optional): Number of Set Attention Blocks
        (SABs) in the decoder. (default: :obj:`1`).
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the seed embeddings
        are averaged instead of concatenated. (default: :obj:`True`)
    norm (str, optional): If set to :obj:`True`, will apply layer
        normalization. (default: :obj:`False`)
    dropout (float, optional): Dropout probability of attention weights.
        (default: :obj:`0`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `SignedConv`

The signed graph convolutional operator from the `"Signed Graph
Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.

.. math::
    \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
    \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
    \mathbf{x}_w , \mathbf{x}_v \right]

    \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
    \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
    \mathbf{x}_w , \mathbf{x}_v \right]

if :obj:`first_aggr` is set to :obj:`True`, and

.. math::
    \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
    \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
    \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
    \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})},
    \mathbf{x}_v^{(\textrm{pos})} \right]

    \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
    \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
    \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
    \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})},
    \mathbf{x}_v^{(\textrm{neg})} \right]

otherwise.
In case :obj:`first_aggr` is :obj:`False`, the layer expects :obj:`x` to be
a tensor where :obj:`x[:, :in_channels]` denotes the positive node features
:math:`\mathbf{X}^{(\textrm{pos})}` and :obj:`x[:, in_channels:]` denotes
the negative node features :math:`\mathbf{X}^{(\textrm{neg})}`.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    first_aggr (bool): Denotes which aggregation formula to use.
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})` or
      :math:`((|\mathcal{V_s}|, F_{in}), (|\mathcal{V_t}|, F_{in}))`
      if bipartite,
      positive edge indices :math:`(2, |\mathcal{E}^{(+)}|)`,
      negative edge indices :math:`(2, |\mathcal{E}^{(-)}|)`
    - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
      :math:`(|\mathcal{V_t}|, F_{out})` if bipartite

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], pos_edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], neg_edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor])`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `SignedGCN`

The signed graph convolutional network model from the `"Signed Graph
Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
Internally, this module uses the
:class:`torch_geometric.nn.conv.SignedConv` operator.

Args:
    in_channels (int): Size of each input sample.
    hidden_channels (int): Size of each hidden sample.
    num_layers (int): Number of layers.
    lamb (float, optional): Balances the contributions of the overall
        objective. (default: :obj:`5`)
    bias (bool, optional): If set to :obj:`False`, all layers will not
        learn an additive bias. (default: :obj:`True`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`split_edges(self, edge_index: torch.Tensor, test_ratio: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]`**
  Splits the edges :obj:`edge_index` into train and test edges.

- **`create_spectral_features(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor`**
  Creates :obj:`in_channels` spectral node features based on

- **`forward(self, x: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> torch.Tensor`**
  Computes node embeddings :obj:`z` based on positive edges

- **`discriminate(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor`**
  Given node embeddings :obj:`z`, classifies the link relation

### `SimpleConv`

A simple message passing operator that performs (non-trainable)
propagation.

.. math::
    \mathbf{x}^{\prime}_i = \bigoplus_{j \in \mathcal{N(i)}} e_{ji} \cdot
    \mathbf{x}_j

where :math:`\bigoplus` defines a custom aggregation scheme.

Args:
    aggr (str or [str] or Aggregation, optional): The aggregation scheme
        to use, *e.g.*, :obj:`"add"`, :obj:`"sum"` :obj:`"mean"`,
        :obj:`"min"`, :obj:`"max"` or :obj:`"mul"`.
        In addition, can be any
        :class:`~torch_geometric.nn.aggr.Aggregation` module (or any string
        that automatically resolves to it). (default: :obj:`"sum"`)
    combine_root (str, optional): Specifies whether or how to combine the
        central node representation (one of :obj:`"sum"`, :obj:`"cat"`,
        :obj:`"self_loop"`, :obj:`None`). (default: :obj:`None`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **inputs:**
      node features :math:`(|\mathcal{V}|, F)` or
      :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, *))`
      if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **outputs:** node features :math:`(|\mathcal{V}|, F)` or
      :math:`(|\mathcal{V_t}|, F)` if bipartite

#### Methods

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `SoftmaxAggregation`

The softmax aggregation operator based on a temperature term, as
described in the `"DeeperGCN: All You Need to Train Deeper GCNs"
<https://arxiv.org/abs/2006.07739>`_ paper.

.. math::
    \mathrm{softmax}(\mathcal{X}|t) = \sum_{\mathbf{x}_i\in\mathcal{X}}
    \frac{\exp(t\cdot\mathbf{x}_i)}{\sum_{\mathbf{x}_j\in\mathcal{X}}
    \exp(t\cdot\mathbf{x}_j)}\cdot\mathbf{x}_{i},

where :math:`t` controls the softness of the softmax when aggregating over
a set of features :math:`\mathcal{X}`.

Args:
    t (float, optional): Initial inverse temperature for softmax
        aggregation. (default: :obj:`1.0`)
    learn (bool, optional): If set to :obj:`True`, will learn the value
        :obj:`t` for softmax aggregation dynamically.
        (default: :obj:`False`)
    semi_grad (bool, optional): If set to :obj:`True`, will turn off
        gradient calculation during softmax computation. Therefore, only
        semi-gradients are used during backpropagation. Useful for saving
        memory and accelerating backward computation when :obj:`t` is not
        learnable. (default: :obj:`False`)
    channels (int, optional): Number of channels to learn from :math:`t`.
        If set to a value greater than :obj:`1`, :math:`t` will be learned
        per input feature channel. This requires compatible shapes for the
        input to the forward calculation. (default: :obj:`1`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `SortAggregation`

The pooling operator from the `"An End-to-End Deep Learning
Architecture for Graph Classification"
<https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
where node features are sorted in descending order based on their last
feature channel. The first :math:`k` nodes form the output of the layer.

.. note::

    :class:`SortAggregation` requires sorted indices :obj:`index` as input.
    Specifically, if you use this aggregation as part of
    :class:`~torch_geometric.nn.conv.MessagePassing`, ensure that
    :obj:`edge_index` is sorted by destination nodes, either by manually
    sorting edge indices via :meth:`~torch_geometric.utils.sort_edge_index`
    or by calling :meth:`torch_geometric.data.Data.sort`.

Args:
    k (int): The number of nodes to hold for each graph.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2, max_num_elements: Optional[int] = None) -> torch.Tensor`**
  Forward pass.

### `SplineConv`

The spline-based convolutional operator from the `"SplineCNN: Fast
Geometric Deep Learning with Continuous B-Spline Kernels"
<https://arxiv.org/abs/1711.08920>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in
    \mathcal{N}(i)} \mathbf{x}_j \cdot
    h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

where :math:`h_{\mathbf{\Theta}}` denotes a kernel function defined
over the weighted B-Spline tensor product basis.

.. note::

    Pseudo-coordinates must lay in the fixed interval :math:`[0, 1]` for
    this method to work as intended.

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    dim (int): Pseudo-coordinate dimensionality.
    kernel_size (int or [int]): Size of the convolving kernel.
    is_open_spline (bool or [bool], optional): If set to :obj:`False`, the
        operator will use a closed B-spline basis in this dimension.
        (default :obj:`True`)
    degree (int, optional): B-spline basis degrees. (default: :obj:`1`)
    aggr (str, optional): The aggregation scheme to use
        (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
        (default: :obj:`"mean"`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add transformed root node features to the output.
        (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`initialize_parameters(self, module, input)`**

### `StdAggregation`

An aggregation operator that takes the feature-wise standard deviation
across a set of elements.

.. math::
    \mathrm{std}(\mathcal{X}) = \sqrt{\mathrm{var}(\mathcal{X})}.

Args:
    semi_grad (bool, optional): If set to :obj:`True`, will turn off
        gradient calculation during :math:`E[X^2]` computation. Therefore,
        only semi-gradients are used during backpropagation. Useful for
        saving memory and accelerating backward computation.
        (default: :obj:`False`)

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `SumAggregation`

An aggregation operator that sums up features across a set of elements.

.. math::
    \mathrm{sum}(\mathcal{X}) = \sum_{\mathbf{x}_i \in \mathcal{X}}
    \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `SuperGATConv`

The self-supervised graph attentional operator from the `"How to Find
Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
<https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper.

.. math::

    \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
    \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

where the two types of attention :math:`\alpha_{i,j}^{\mathrm{MX\ or\ SD}}`
are computed as:

.. math::

    \alpha_{i,j}^{\mathrm{MX\ or\ SD}} &=
    \frac{
    \exp\left(\mathrm{LeakyReLU}\left(
        e_{i,j}^{\mathrm{MX\ or\ SD}}
    \right)\right)}
    {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
    \exp\left(\mathrm{LeakyReLU}\left(
        e_{i,k}^{\mathrm{MX\ or\ SD}}
    \right)\right)}

    e_{i,j}^{\mathrm{MX}} &= \mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \,
         \mathbf{\Theta}\mathbf{x}_j]
        \cdot \sigma \left(
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        \right)

    e_{i,j}^{\mathrm{SD}} &= \frac{
        \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
        \mathbf{\Theta}\mathbf{x}_j
    }{ \sqrt{d} }

The self-supervised task is a link prediction using the attention values
as input to predict the likelihood :math:`\phi_{i,j}^{\mathrm{MX\ or\ SD}}`
that an edge exists between nodes:

.. math::

    \phi_{i,j}^{\mathrm{MX}} &= \sigma \left(
        \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
        \mathbf{\Theta}\mathbf{x}_j
    \right)

    \phi_{i,j}^{\mathrm{SD}} &= \sigma \left(
        \frac{
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        }{ \sqrt{d} }
    \right)

.. note::

    For an example of using SuperGAT, see `examples/super_gat.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    super_gat.py>`_.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    add_self_loops (bool, optional): If set to :obj:`False`, will not add
        self-loops to the input graph. (default: :obj:`True`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    attention_type (str, optional): Type of attention to use
        (:obj:`'MX'`, :obj:`'SD'`). (default: :obj:`'MX'`)
    neg_sample_ratio (float, optional): The ratio of the number of sampled
        negative edges to the number of positive edges.
        (default: :obj:`0.5`)
    edge_sample_ratio (float, optional): The ratio of samples to use for
        training among the number of training edges. (default: :obj:`1.0`)
    is_undirected (bool, optional): Whether the input graph is undirected.
        If not given, will be automatically computed with the input graph
        when negative sampling is performed. (default: :obj:`False`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      edge indices :math:`(2, |\mathcal{E}|)`,
      negative edge indices :math:`(2, |\mathcal{E}^{(-)}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], neg_edge_index: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, edge_index_i: torch.Tensor, x_i: torch.Tensor, x_j: torch.Tensor, size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`negative_sampling(self, edge_index: torch.Tensor, num_nodes: int, batch: Optional[torch.Tensor] = None) -> torch.Tensor`**

- **`positive_sampling(self, edge_index: torch.Tensor) -> torch.Tensor`**

### `TAGConv`

The topology adaptive graph convolutional networks operator from the
`"Topology Adaptive Graph Convolutional Networks"
<https://arxiv.org/abs/1710.10370>`_ paper.

.. math::
    \mathbf{X}^{\prime} = \sum_{k=0}^K \left( \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2} \right)^k \mathbf{X} \mathbf{W}_{k},

where :math:`\mathbf{A}` denotes the adjacency matrix and
:math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix.
The adjacency matrix can include other values than :obj:`1` representing
edge weights via the optional :obj:`edge_weight` tensor.

Args:
    in_channels (int): Size of each input sample, or :obj:`-1` to derive
        the size from the first input(s) to the forward method.
    out_channels (int): Size of each output sample.
    K (int, optional): Number of hops :math:`K`. (default: :obj:`3`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    normalize (bool, optional): Whether to apply symmetric normalization.
        (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node_features :math:`(|\mathcal{V}|, F_{in})`,
      edge_index :math:`(2, |\mathcal{E}|)`,
      edge_weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: torch.Tensor) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `TGNMemory`

The Temporal Graph Network (TGN) memory model from the
`"Temporal Graph Networks for Deep Learning on Dynamic Graphs"
<https://arxiv.org/abs/2006.10637>`_ paper.

.. note::

    For an example of using TGN, see `examples/tgn.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    tgn.py>`_.

Args:
    num_nodes (int): The number of nodes to save memories for.
    raw_msg_dim (int): The raw message dimensionality.
    memory_dim (int): The hidden memory dimensionality.
    time_dim (int): The time encoding dimensionality.
    message_module (torch.nn.Module): The message function which
        combines source and destination node memory embeddings, the raw
        message and the time encoding.
    aggregator_module (torch.nn.Module): The message aggregator function
        which aggregates messages to the same destination into a single
        representation.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`reset_state(self)`**
  Resets the memory to its initial state.

- **`detach(self)`**
  Detaches the memory from gradient computation.

- **`forward(self, n_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]`**
  Returns, for all nodes :obj:`n_id`, their current memory and their

- **`update_state(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, raw_msg: torch.Tensor)`**
  Updates the memory with newly encountered interactions

### `TemporalEncoding`

The time-encoding function from the `"Do We Really Need Complicated
Model Architectures for Temporal Networks?"
<https://openreview.net/forum?id=ayPPc0SyLv1>`_ paper.

It first maps each entry to a vector with exponentially decreasing values,
and then uses the cosine function to project all values to range
:math:`[-1, 1]`.

.. math::
    y_{i} = \cos \left(x \cdot \sqrt{d}^{-(i - 1)/\sqrt{d}} \right)

where :math:`d` defines the output feature dimension, and
:math:`1 \leq i \leq d`.

Args:
    out_channels (int): Size :math:`d` of each output sample.

#### Methods

- **`reset_parameters(self)`**

- **`forward(self, x: torch.Tensor) -> torch.Tensor`**

### `TopKPooling`

:math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
<https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
and `"Understanding Attention and Generalization in Graph Neural
Networks" <https://arxiv.org/abs/1905.02850>`_ papers.

If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:

    .. math::
        \mathbf{y} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
        \mathbf{p} \|} \right)

        \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot
        \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}

If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
computes:

    .. math::
        \mathbf{y} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

        \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

        \mathbf{X}^{\prime} &= (\mathbf{X} \odot \mathbf{y})_{\mathbf{i}}

        \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}},

where nodes are dropped based on a learnable projection score
:math:`\mathbf{p}`.

Args:
    in_channels (int): Size of each input sample.
    ratio (float or int): The graph pooling ratio, which is used to compute
        :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
        of :math:`k` itself, depending on whether the type of :obj:`ratio`
        is :obj:`float` or :obj:`int`.
        This value is ignored if :obj:`min_score` is not :obj:`None`.
        (default: :obj:`0.5`)
    min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
        which is used to compute indices of pooled nodes
        :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
        When this value is not :obj:`None`, the :obj:`ratio` argument is
        ignored. (default: :obj:`None`)
    multiplier (float, optional): Coefficient by which features gets
        multiplied after pooling. This can be useful for large graphs and
        when :obj:`min_score` is used. (default: :obj:`1`)
    nonlinearity (str or callable, optional): The non-linearity
        :math:`\sigma`. (default: :obj:`"tanh"`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, batch: Optional[torch.Tensor] = None, attn: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]`**
  Forward pass.

### `TransE`

The TransE model from the `"Translating Embeddings for Modeling
Multi-Relational Data" <https://proceedings.neurips.cc/paper/2013/file/
1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf>`_ paper.

:class:`TransE` models relations as a translation from head to tail
entities such that

.. math::
    \mathbf{e}_h + \mathbf{e}_r \approx \mathbf{e}_t,

resulting in the scoring function:

.. math::
    d(h, r, t) = - {\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|}_p

.. note::

    For an example of using the :class:`TransE` model, see
    `examples/kge_fb15k_237.py
    <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
    kge_fb15k_237.py>`_.

Args:
    num_nodes (int): The number of nodes/entities in the graph.
    num_relations (int): The number of relations in the graph.
    hidden_channels (int): The hidden embedding size.
    margin (int, optional): The margin of the ranking loss.
        (default: :obj:`1.0`)
    p_norm (int, optional): The order embedding and distance normalization.
        (default: :obj:`1.0`)
    sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to the
        embedding matrices will be sparse. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the score for the given triplet.

- **`loss(self, head_index: torch.Tensor, rel_type: torch.Tensor, tail_index: torch.Tensor) -> torch.Tensor`**
  Returns the loss value for the given triplet.

### `TransformerConv`

The graph transformer operator from the `"Masked Label Prediction:
Unified Message Passing Model for Semi-Supervised Classification"
<https://arxiv.org/abs/2009.03509>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
    \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \mathbf{W}_2 \mathbf{x}_{j},

where the attention coefficients :math:`\alpha_{i,j}` are computed via
multi-head dot product attention:

.. math::
    \alpha_{i,j} = \textrm{softmax} \left(
    \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)}
    {\sqrt{d}} \right)

Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    beta (bool, optional): If set, will combine aggregation and
        skip information via

        .. math::
            \mathbf{x}^{\prime}_i = \beta_i \mathbf{W}_1 \mathbf{x}_i +
            (1 - \beta_i) \underbrace{\left(\sum_{j \in \mathcal{N}(i)}
            \alpha_{i,j} \mathbf{W}_2 \vec{x}_j \right)}_{=\mathbf{m}_i}

        with :math:`\beta_i = \textrm{sigmoid}(\mathbf{w}_5^{\top}
        [ \mathbf{W}_1 \mathbf{x}_i, \mathbf{m}_i, \mathbf{W}_1
        \mathbf{x}_i - \mathbf{m}_i ])` (default: :obj:`False`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    edge_dim (int, optional): Edge feature dimensionality (in case
        there are any). Edge features are added to the keys after
        linear transformation, that is, prior to computing the
        attention dot product. They are also added to final values
        after the same linear transformation. The model is:

        .. math::
            \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i +
            \sum_{j \in \mathcal{N}(i)} \alpha_{i,j} \left(
            \mathbf{W}_2 \mathbf{x}_{j} + \mathbf{W}_6 \mathbf{e}_{ij}
            \right),

        where the attention coefficients :math:`\alpha_{i,j}` are now
        computed via:

        .. math::
            \alpha_{i,j} = \textrm{softmax} \left(
            \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top}
            (\mathbf{W}_4\mathbf{x}_j + \mathbf{W}_6 \mathbf{e}_{ij})}
            {\sqrt{d}} \right)

        (default :obj:`None`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    root_weight (bool, optional): If set to :obj:`False`, the layer will
        not add the transformed root node features to the output and the
        option  :attr:`beta` is set to :obj:`False`. (default: :obj:`True`)
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_attr: Optional[torch.Tensor] = None, return_attention_weights: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch_geometric.typing.SparseTensor]]`**
  Runs the forward pass of the module.

- **`message(self, query_i: torch.Tensor, key_j: torch.Tensor, value_j: torch.Tensor, edge_attr: Optional[torch.Tensor], index: torch.Tensor, ptr: Optional[torch.Tensor], size_i: Optional[int]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

### `VGAE`

The Variational Graph Auto-Encoder model from the
`"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
paper.

Args:
    encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
        and :math:`\log\sigma^2`.
    decoder (torch.nn.Module, optional): The decoder module. If set to
        :obj:`None`, will default to the
        :class:`torch_geometric.nn.models.InnerProductDecoder`.
        (default: :obj:`None`)

#### Methods

- **`reparametrize(self, mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor`**

- **`encode(self, *args, **kwargs) -> torch.Tensor`**

- **`kl_loss(self, mu: Optional[torch.Tensor] = None, logstd: Optional[torch.Tensor] = None) -> torch.Tensor`**
  Computes the KL loss, either for the passed arguments :obj:`mu`

### `VarAggregation`

An aggregation operator that takes the feature-wise variance across a
set of elements.

.. math::
    \mathrm{var}(\mathcal{X}) = \mathrm{mean}(\{ \mathbf{x}_i^2 : x \in
    \mathcal{X} \}) - \mathrm{mean}(\mathcal{X})^2.

Args:
    semi_grad (bool, optional): If set to :obj:`True`, will turn off
        gradient calculation during :math:`E[X^2]` computation. Therefore,
        only semi-gradients are used during backpropagation. Useful for
        saving memory and accelerating backward computation.
        (default: :obj:`False`)

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `VariancePreservingAggregation`

Performs the Variance Preserving Aggregation (VPA) from the `"GNN-VPA:
A Variance-Preserving Aggregation Strategy for Graph Neural Networks"
<https://arxiv.org/abs/2403.04747>`_ paper.

.. math::
    \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
    \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

### `ViSNet`

A :pytorch:`PyTorch` module that implements the equivariant
vector-scalar interactive graph neural network (ViSNet) from the
`"Enhancing Geometric Representations for Molecules with Equivariant
Vector-Scalar Interactive Message Passing"
<https://arxiv.org/abs/2210.16518>`_ paper.

Args:
    lmax (int, optional): The maximum degree of the spherical harmonics.
        (default: :obj:`1`)
    vecnorm_type (str, optional): The type of normalization to apply to the
        vectors. (default: :obj:`None`)
    trainable_vecnorm (bool, optional):  Whether the normalization weights
        are trainable. (default: :obj:`False`)
    num_heads (int, optional): The number of attention heads.
        (default: :obj:`8`)
    num_layers (int, optional): The number of layers in the network.
        (default: :obj:`6`)
    hidden_channels (int, optional): The number of hidden channels in the
        node embeddings. (default: :obj:`128`)
    num_rbf (int, optional): The number of radial basis functions.
        (default: :obj:`32`)
    trainable_rbf (bool, optional): Whether the radial basis function
        parameters are trainable. (default: :obj:`False`)
    max_z (int, optional): The maximum atomic numbers.
        (default: :obj:`100`)
    cutoff (float, optional): The cutoff distance. (default: :obj:`5.0`)
    max_num_neighbors (int, optional): The maximum number of neighbors
        considered for each atom. (default: :obj:`32`)
    vertex (bool, optional): Whether to use vertex geometric features.
        (default: :obj:`False`)
    atomref (torch.Tensor, optional): A tensor of atom reference values,
        or :obj:`None` if not provided. (default: :obj:`None`)
    reduce_op (str, optional): The type of reduction operation to apply
        (:obj:`"sum"`, :obj:`"mean"`). (default: :obj:`"sum"`)
    mean (float, optional): The mean of the output distribution.
        (default: :obj:`0.0`)
    std (float, optional): The standard deviation of the output
        distribution. (default: :obj:`1.0`)
    derivative (bool, optional): Whether to compute the derivative of the
        output with respect to the positions. (default: :obj:`False`)

#### Methods

- **`reset_parameters(self)`**
  Resets the parameters of the module.

- **`forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]`**
  Computes the energies or properties (forces) for a batch of

### `WLConv`

The Weisfeiler Lehman (WL) operator from the `"A Reduction of a Graph
to a Canonical Form and an Algebra Arising During this Reduction"
<https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper.

:class:`WLConv` iteratively refines node colorings according to:

.. math::
    \mathbf{x}^{\prime}_i = \textrm{hash} \left( \mathbf{x}_i, \{
    \mathbf{x}_j \colon j \in \mathcal{N}(i) \} \right)

Shapes:
    - **input:**
      node coloring :math:`(|\mathcal{V}|, F_{in})` *(one-hot encodings)*
      or :math:`(|\mathcal{V}|)` *(integer-based)*,
      edge indices :math:`(2, |\mathcal{E}|)`
    - **output:** node coloring :math:`(|\mathcal{V}|)` *(integer-based)*

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor]) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`histogram(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None, norm: bool = False) -> torch.Tensor`**
  Given a node coloring :obj:`x`, computes the color histograms of

### `WLConvContinuous`

The Weisfeiler Lehman operator from the `"Wasserstein
Weisfeiler-Lehman Graph Kernels" <https://arxiv.org/abs/1906.01277>`_
paper.

Refinement is done though a degree-scaled mean aggregation and works on
nodes with continuous attributes:

.. math::
    \mathbf{x}^{\prime}_i = \frac{1}{2}\big(\mathbf{x}_i +
    \frac{1}{\textrm{deg}(i)}
    \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot \mathbf{x}_j \big)

where :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to
target node :obj:`i` (default: :obj:`1`)

Args:
    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F)` or
      :math:`((|\mathcal{V_s}|, F), (|\mathcal{V_t}|, F))` if bipartite,
      edge indices :math:`(2, |\mathcal{E}|)`,
      edge weights :math:`(|\mathcal{E}|)` *(optional)*
    - **output:** node features :math:`(|\mathcal{V}|, F)` or
      :math:`(|\mathcal{V}_t|, F)` if bipartite

#### Methods

- **`forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]], edge_index: Union[torch.Tensor, torch_geometric.typing.SparseTensor], edge_weight: Optional[torch.Tensor] = None, size: Optional[Tuple[int, int]] = None) -> torch.Tensor`**
  Runs the forward pass of the module.

- **`message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor]) -> torch.Tensor`**
  Constructs messages from node :math:`j` to node :math:`i`

- **`message_and_aggregate(self, adj_t: Union[torch.Tensor, torch_geometric.typing.SparseTensor], x: Tuple[torch.Tensor, Optional[torch.Tensor]]) -> torch.Tensor`**
  Fuses computations of :func:`message` and :func:`aggregate` into a

### `XConv`

The convolutional operator on :math:`\mathcal{X}`-transformed points
from the `"PointCNN: Convolution On X-Transformed Points"
<https://arxiv.org/abs/1801.07791>`_ paper.

.. math::
    \mathbf{x}^{\prime}_i = \mathrm{Conv}\left(\mathbf{K},
    \gamma_{\mathbf{\Theta}}(\mathbf{P}_i - \mathbf{p}_i) \times
    \left( h_\mathbf{\Theta}(\mathbf{P}_i - \mathbf{p}_i) \, \Vert \,
    \mathbf{x}_i \right) \right),

where :math:`\mathbf{K}` and :math:`\mathbf{P}_i` denote the trainable
filter and neighboring point positions of :math:`\mathbf{x}_i`,
respectively.
:math:`\gamma_{\mathbf{\Theta}}` and :math:`h_{\mathbf{\Theta}}` describe
neural networks, *i.e.* MLPs, where :math:`h_{\mathbf{\Theta}}`
individually lifts each point into a higher-dimensional space, and
:math:`\gamma_{\mathbf{\Theta}}` computes the :math:`\mathcal{X}`-
transformation matrix based on *all* points in a neighborhood.

Args:
    in_channels (int): Size of each input sample.
    out_channels (int): Size of each output sample.
    dim (int): Point cloud dimensionality.
    kernel_size (int): Size of the convolving kernel, *i.e.* number of
        neighbors including self-loops.
    hidden_channels (int, optional): Output size of
        :math:`h_{\mathbf{\Theta}}`, *i.e.* dimensionality of lifted
        points. If set to :obj:`None`, will be automatically set to
        :obj:`in_channels / 4`. (default: :obj:`None`)
    dilation (int, optional): The factor by which the neighborhood is
        extended, from which :obj:`kernel_size` neighbors are then
        uniformly sampled. Can be interpreted as the dilation rate of
        classical convolutional operators. (default: :obj:`1`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    num_workers (int): Number of workers to use for k-NN computation.
        Has no effect in case :obj:`batch` is not :obj:`None`, or the input
        lies on the GPU. (default: :obj:`1`)

Shapes:
    - **input:**
      node features :math:`(|\mathcal{V}|, F_{in})`,
      positions :math:`(|\mathcal{V}|, D)`,
      batch vector :math:`(|\mathcal{V}|)` *(optional)*
    - **output:**
      node features :math:`(|\mathcal{V}|, F_{out})`

#### Methods

- **`reset_parameters(self)`**
  Resets all learnable parameters of the module.

- **`forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None)`**
  Runs the forward pass of the module.

## Nested Submodules (24)

Each nested submodule is documented in a separate file:

### [aggr](./nn/aggr.md)
Module: `torch_geometric.nn.aggr`

*Contains: 27 classes*

### [attention](./nn/attention.md)
Module: `torch_geometric.nn.attention`

*Contains: 1 classes*

### [conv](./nn/conv.md)
Module: `torch_geometric.nn.conv`

*Contains: 68 classes*

### [data_parallel](./nn/data_parallel.md)
Module: `torch_geometric.nn.data_parallel`

*Contains: 1 functions, 3 classes*

### [dense](./nn/dense.md)
Module: `torch_geometric.nn.dense`

*Contains: 2 functions, 9 classes*

### [encoding](./nn/encoding.md)
Module: `torch_geometric.nn.encoding`

*Contains: 3 classes*

### [functional](./nn/functional.md)
Module: `torch_geometric.nn.functional`

*Contains: 2 functions*

### [fx](./nn/fx.md)
Module: `torch_geometric.nn.fx`

*Contains: 4 functions, 9 classes*

### [glob](./nn/glob.md)
Module: `torch_geometric.nn.glob`

*Contains: 5 functions, 3 classes*

### [inits](./nn/inits.md)
Module: `torch_geometric.nn.inits`

*Contains: 9 functions, 2 classes*

### [kge](./nn/kge.md)
Module: `torch_geometric.nn.kge`

*Contains: 5 classes*

### [lr_scheduler](./nn/lr_scheduler.md)
Module: `torch_geometric.nn.lr_scheduler`

*Contains: 7 classes*

### [models](./nn/models.md)
Module: `torch_geometric.nn.models`

*Contains: 3 functions, 39 classes*

### [module_dict](./nn/module_dict.md)
Module: `torch_geometric.nn.module_dict`

*Contains: 2 classes*

### [nlp](./nn/nlp.md)
Module: `torch_geometric.nn.nlp`

*Contains: 2 classes*

### [norm](./nn/norm.md)
Module: `torch_geometric.nn.norm`

*Contains: 11 classes*

### [parameter_dict](./nn/parameter_dict.md)
Module: `torch_geometric.nn.parameter_dict`

*Contains: 2 classes*

### [reshape](./nn/reshape.md)
Module: `torch_geometric.nn.reshape`

*Contains: 2 classes*

### [resolver](./nn/resolver.md)
Module: `torch_geometric.nn.resolver`

*Contains: 8 functions, 10 classes*

### [sequential](./nn/sequential.md)
Module: `torch_geometric.nn.sequential`

*Contains: 4 functions, 6 classes*

### [to_fixed_size_transformer](./nn/to_fixed_size_transformer.md)
Module: `torch_geometric.nn.to_fixed_size_transformer`

*Contains: 1 functions, 7 classes*

### [to_hetero_transformer](./nn/to_hetero_transformer.md)
Module: `torch_geometric.nn.to_hetero_transformer`

*Contains: 7 functions, 10 classes*

### [to_hetero_with_bases_transformer](./nn/to_hetero_with_bases_transformer.md)
Module: `torch_geometric.nn.to_hetero_with_bases_transformer`

*Contains: 10 functions, 15 classes*

### [unpool](./nn/unpool.md)
Module: `torch_geometric.nn.unpool`

*Contains: 1 functions*
