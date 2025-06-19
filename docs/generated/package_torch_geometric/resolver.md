# resolver Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.resolver`

## Functions (2)

### `normalize_string(s: str) -> str`

### `resolver(classes: List[Any], class_dict: Dict[str, Any], query: Union[Any, str], base_cls: Optional[Any], base_cls_repr: Optional[str], *args: Any, **kwargs: Any) -> Any`

## Important Data Types (5)

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

### `Dict`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of dict.

*(has methods, callable)*

### `List`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of list.

*(has methods, callable)*

### `Union`
**Type**: `<class 'typing._SpecialForm'>`

Union type; Union[X, Y] means either X or Y.

On Python 3.10 and higher, the | operator
can also be used to denote unions;
X | Y means the same thing to the type checker as Union[X, Y].

To define a union, use e.g. Union[int, str]. Details:
- The arguments must be types and there must be at least one.
- None as an argument is a special case and is replaced by
  type(None).
- Unions of unions are flattened, e.g.::

    assert Union[Union[int, str], float] == Union[int, str, float]

- Unions of a single argument vanish, e.g.::

    assert Union[int] == int  # The constructor actually returns int

- Redundant arguments are skipped, e.g.::

    assert Union[int, str, int] == Union[int, str]

- When comparing unions, the argument order is ignored, e.g.::

    assert Union[int, str] == Union[str, int]

- You cannot subclass or instantiate a union.
- You can use Optional[X] as a shorthand for Union[X, None].

*(callable)*

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
