# warnings Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.warnings`

## Functions (2)

### `filterwarnings(action: Literal['default', 'error', 'ignore', 'always', 'module', 'once'], message: str) -> None`

### `warn(message: str) -> None`

## Important Data Types (1)

### `Literal`
**Type**: `<class 'typing._LiteralSpecialForm'>`

Special typing form to define literal types (a.k.a. value types).

This form can be used to indicate to type checkers that the corresponding
variable or function parameter has a value equivalent to the provided
literal (or one of several literals)::

    def validate_simple(data: Any) -> Literal[True]:  # always returns True
        ...

    MODE = Literal['r', 'rb', 'w', 'wb']
    def open_helper(file: str, mode: MODE) -> str:
        ...

    open_helper('/some/path', 'r')  # Passes type check
    open_helper('/other/path', 'typo')  # Error in type checker

Literal[...] cannot be subclassed. At runtime, an arbitrary value
is allowed as type argument to Literal[...], but type checkers may
impose restrictions.

*(has methods, callable)*

## Nested Submodules (1)

Each nested submodule is documented in a separate file:

### [torch_geometric](./warnings/torch_geometric.md)
Module: `torch_geometric`

*Contains: 12 functions, 8 classes*
