# txt_array

Part of `torch_geometric.io`
Module: `torch_geometric.io.txt_array`

## Functions (2)

### `parse_txt_array(src: List[str], sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

### `read_txt_array(path: str, sep: Optional[str] = None, start: int = 0, end: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor`

## Classes (1)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
