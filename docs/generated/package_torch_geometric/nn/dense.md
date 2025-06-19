# dense

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.dense`

## Description

Dense neural network module package.

This package provides modules applicable for operating on dense tensor
representations.

## Functions (2)

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

## Classes (9)

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
