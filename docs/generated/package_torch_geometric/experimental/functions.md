# functions

Part of `torch_geometric.experimental`
Module: `torch_geometric.utils.functions`

## Functions (1)

### `cumsum(x: torch.Tensor, dim: int = 0) -> torch.Tensor`

Returns the cumulative sum of elements of :obj:`x`.
In contrast to :meth:`torch.cumsum`, prepends the output with zero.

Args:
    x (torch.Tensor): The input tensor.
    dim (int, optional): The dimension to do the operation over.
        (default: :obj:`0`)

Example:
    >>> x = torch.tensor([2, 4, 1])
    >>> cumsum(x)
    tensor([0, 2, 6, 7])

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
