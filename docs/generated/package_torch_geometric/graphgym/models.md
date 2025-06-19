# models

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.models`

## Functions (5)

### `GNNLayer(dim_in: int, dim_out: int, has_act: bool = True) -> torch_geometric.graphgym.models.layer.GeneralLayer`

Creates a GNN layer, given the specified input and output dimensions
and the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    has_act (bool, optional): Whether to apply an activation function
        after the layer. (default: :obj:`True`)

### `GNNPreMP(dim_in: int, dim_out: int, num_layers: int) -> torch_geometric.graphgym.models.layer.GeneralMultiLayer`

Creates a NN layer used before message passing, given the specified
input and output dimensions and the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    num_layers (int): The number of layers.

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

## Classes (23)

### `AtomEncoder`

The atom encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = AtomEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BatchNorm1dEdge`

A batch normalization layer for edge-level features.

Args:
    layer_config (LayerConfig): The configuration of the layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BatchNorm1dNode`

A batch normalization layer for node-level features.

Args:
    layer_config (LayerConfig): The configuration of the layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `BondEncoder`

The bond encoder used in OGB molecule dataset.

Args:
    emb_dim (int): The output embedding dimension.

Example:
    >>> encoder = BondEncoder(emb_dim=16)
    >>> batch = torch.randint(0, 10, (10, 3))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `FeatureEncoder`

Encodes node and edge features, given the specified input dimension and
the underlying configuration in :obj:`cfg`.

Args:
    dim_in (int): The input feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GATConv`

A Graph Attention Network (GAT) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GCNConv`

A Graph Convolutional Network (GCN) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GINConv`

A Graph Isomorphism Network (GIN) layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNN`

A general Graph Neural Network (GNN) model.

The GNN model consists of three main components:

1. An encoder to transform input features into a fixed-size embedding
   space.
2. A processing or message passing stage for information exchange between
   nodes.
3. A head to produce the final output features/predictions.

The configuration of each component is determined by the underlying
configuration in :obj:`cfg`.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNEdgeHead`

A GNN prediction head for edge-level/link-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNGraphHead`

A GNN prediction head for graph-level prediction tasks.
A post message passing layer (as specified by :obj:`cfg.gnn.post_mp`) is
used to transform the pooled graph-level embeddings using an MLP.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNNodeHead`

A GNN prediction head for node-level prediction tasks.

Args:
    dim_in (int): The input feature dimension.
    dim_out (int): The output feature dimension.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GNNStackStage`

Stacks a number of GNN layers.

Args:
    dim_in (int): The input dimension
    dim_out (int): The output dimension.
    num_layers (int): The number of layers.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralConv`

A general GNN layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralEdgeConv`

A general GNN layer with edge feature support.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralLayer`

A general wrapper for layers.

Args:
    name (str): The registered name of the layer.
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralMultiLayer`

A general wrapper class for a stacking multiple NN layers.

Args:
    name (str): The registered name of the layer.
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `GeneralSampleEdgeConv`

A general GNN layer that supports edge features and edge sampling.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `IntegerFeatureEncoder`

Provides an encoder for integer node features.

Args:
    emb_dim (int): The output embedding dimension.
    num_classes (int): The number of classes/integers.

Example:
    >>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
    >>> batch = torch.randint(0, 10, (10, 2))
    >>> encoder(batch).size()
    torch.Size([10, 16])

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `Linear`

A basic Linear layer.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `MLP`

A basic MLP model.

Args:
    layer_config (LayerConfig): The configuration of the layer.
    **kwargs (optional): Additional keyword arguments.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `SAGEConv`

A GraphSAGE layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.

### `SplineConv`

A SplineCNN layer.

#### Methods

- **`forward(self, batch)`**
  Define the computation performed at every call.
