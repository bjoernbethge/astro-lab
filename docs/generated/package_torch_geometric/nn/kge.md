# kge

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.kge`

## Description

Knowledge Graph Embedding (KGE) package.

## Classes (5)

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
