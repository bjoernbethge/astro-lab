# embedding

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.embedding`

## Functions (1)

### `get_embeddings(model: torch.nn.modules.module.Module, *args: Any, **kwargs: Any) -> List[torch.Tensor]`

Returns the output embeddings of all
:class:`~torch_geometric.nn.conv.MessagePassing` layers in
:obj:`model`.

Internally, this method registers forward hooks on all
:class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
and runs the forward pass of the :obj:`model` by calling
:obj:`model(*args, **kwargs)`.

Args:
    model (torch.nn.Module): The message passing model.
    *args: Arguments passed to the model.
    **kwargs (optional): Additional keyword arguments passed to the model.

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
