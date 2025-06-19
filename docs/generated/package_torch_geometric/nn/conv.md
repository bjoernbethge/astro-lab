# conv

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.conv`

## Classes (68)

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

### `ECConv`

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

### `PointConv`

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
