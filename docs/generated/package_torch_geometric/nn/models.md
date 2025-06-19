# models

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.models`

## Description

Model package.

## Functions (3)

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

## Classes (39)

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
