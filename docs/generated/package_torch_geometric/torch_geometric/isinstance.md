# isinstance

Part of `torch_geometric.torch_geometric`
Module: `torch_geometric.isinstance`

## Functions (1)

### `is_torch_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool`

Checks if the :obj:`obj` is an instance of a :obj:`cls`.

This function extends :meth:`isinstance` to be applicable during
:meth:`torch.compile` usage by checking against the original class of
compiled models.

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
