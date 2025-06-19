# deprecation Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.deprecation`

## Functions (1)

### `deprecated(details: Optional[str] = None, func_name: Optional[str] = None) -> Callable`

## Important Data Types (3)

### `Any`
**Type**: `<class 'typing._AnyMeta'>`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

*(has methods, callable)*

### `Callable`
**Type**: `<class 'typing._CallableType'>`

Deprecated alias to collections.abc.Callable.

Callable[[int], str] signifies a function that takes a single
parameter of type int and returns a str.

The subscription syntax must always be used with exactly two
values: the argument list and the return type.
The argument list must be a list of types, a ParamSpec,
Concatenate or ellipsis. The return type must be a single type.

There is no syntax to indicate optional or keyword arguments;
such function types are rarely used as callback types.

*(has methods, callable)*

### `Optional`
**Type**: `<class 'typing._SpecialForm'>`

Optional[X] is equivalent to Union[X, None].

*(callable)*

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
