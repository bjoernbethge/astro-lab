# nested

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.nested`

## Functions (3)

### `from_nested_tensor(x: torch.Tensor, return_batch: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`

Given a `nested PyTorch tensor
<https://pytorch.org/docs/stable/nested.html>`__, creates a contiguous
batch of tensors
:math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`, and
optionally a batch vector which assigns each element to a specific example.
Reverse operation of :meth:`to_nested_tensor`.

Args:
    x (torch.Tensor): The nested input tensor. The size of nested tensors
        need to match except for the first dimension.
    return_batch (bool, optional): If set to :obj:`True`, will also return
        the batch vector :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`.
        (default: :obj:`False`)

### `scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None, reduce: str = 'sum') -> torch.Tensor`

Reduces all values from the :obj:`src` tensor at the indices
specified in the :obj:`index` tensor along a given dimension
:obj:`dim`. See the `documentation
<https://pytorch-scatter.readthedocs.io/en/latest/functions/
scatter.html>`__ of the :obj:`torch_scatter` package for more
information.

Args:
    src (torch.Tensor): The source tensor.
    index (torch.Tensor): The index tensor.
    dim (int, optional): The dimension along which to index.
        (default: :obj:`0`)
    dim_size (int, optional): The size of the output tensor at
        dimension :obj:`dim`. If set to :obj:`None`, will create a
        minimal-sized output tensor according to
        :obj:`index.max() + 1`. (default: :obj:`None`)
    reduce (str, optional): The reduce operation (:obj:`"sum"`,
        :obj:`"mean"`, :obj:`"mul"`, :obj:`"min"` or :obj:`"max"`,
        :obj:`"any"`). (default: :obj:`"sum"`)

### `to_nested_tensor(x: torch.Tensor, batch: Optional[torch.Tensor] = None, ptr: Optional[torch.Tensor] = None, batch_size: Optional[int] = None) -> torch.Tensor`

Given a contiguous batch of tensors
:math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`
(with :math:`N_i` indicating the number of elements in example :math:`i`),
creates a `nested PyTorch tensor
<https://pytorch.org/docs/stable/nested.html>`__.
Reverse operation of :meth:`from_nested_tensor`.

Args:
    x (torch.Tensor): The input tensor
        :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`.
    batch (torch.Tensor, optional): The batch vector
        :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
        element to a specific example. Must be ordered.
        (default: :obj:`None`)
    ptr (torch.Tensor, optional): Alternative representation of
        :obj:`batch` in compressed format. (default: :obj:`None`)
    batch_size (int, optional): The batch size :math:`B`.
        (default: :obj:`None`)

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
