# mask

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.mask`

## Functions (3)

### `index_to_mask(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

Converts indices to a mask representation.

Args:
    index (Tensor): The indices.
    size (int, optional): The size of the mask. If set to :obj:`None`, a
        minimal sized output mask is returned.

Example:
    >>> index = torch.tensor([1, 3, 5])
    >>> index_to_mask(index)
    tensor([False,  True, False,  True, False,  True])

    >>> index_to_mask(index, size=7)
    tensor([False,  True, False,  True, False,  True, False])

### `mask_select(src: torch.Tensor, dim: int, mask: torch.Tensor) -> torch.Tensor`

Returns a new tensor which masks the :obj:`src` tensor along the
dimension :obj:`dim` according to the boolean mask :obj:`mask`.

Args:
    src (torch.Tensor): The input tensor.
    dim (int): The dimension in which to mask.
    mask (torch.BoolTensor): The 1-D tensor containing the binary mask to
        index with.

### `mask_to_index(mask: torch.Tensor) -> torch.Tensor`

Converts a mask to an index representation.

Args:
    mask (Tensor): The mask.

Example:
    >>> mask = torch.tensor([False, True, False])
    >>> mask_to_index(mask)
    tensor([1])

## Classes (2)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `TensorFrame`
