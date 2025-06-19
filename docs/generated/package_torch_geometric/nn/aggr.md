# aggr

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.aggr`

## Classes (27)

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

### `MinAggregation`

An aggregation operator that takes the feature-wise minimum across a
set of elements.

.. math::
    \mathrm{min}(\mathcal{X}) = \min_{\mathbf{x}_i \in \mathcal{X}}
    \mathbf{x}_i.

#### Methods

- **`forward(self, x: torch.Tensor, index: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> torch.Tensor`**
  Forward pass.

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
