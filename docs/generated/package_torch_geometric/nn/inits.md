# inits

Part of `torch_geometric.nn`
Module: `torch_geometric.nn.inits`

## Functions (9)

### `constant(value: Any, fill_value: float)`

### `glorot(value: Any)`

### `glorot_orthogonal(tensor, scale)`

### `kaiming_uniform(value: Any, fan: int, a: float)`

### `normal(value: Any, mean: float, std: float)`

### `ones(tensor: Any)`

### `reset(value: Any)`

### `uniform(size: int, value: Any)`

### `zeros(value: Any)`

## Classes (2)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
