# norm

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.norm`

## Description

Normalization package.

## Classes (11)

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
