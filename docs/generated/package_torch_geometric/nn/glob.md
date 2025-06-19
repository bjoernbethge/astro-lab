# glob

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.glob`

## Functions (5)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

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

## Classes (3)

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
