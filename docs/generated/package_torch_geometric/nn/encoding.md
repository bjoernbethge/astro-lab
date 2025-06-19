# encoding

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.encoding`

## Classes (3)

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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
