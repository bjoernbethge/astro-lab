# isinstance Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.isinstance`

## Functions (1)

### `is_torch_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool`

Checks if the :obj:`obj` is an instance of a :obj:`cls`.

This function extends :meth:`isinstance` to be applicable during
:meth:`torch.compile` usage by checking against the original class of
compiled models.

## Important Data Types (4)

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

### `Type`
**Type**: `<class 'typing._SpecialGenericAlias'>`

Deprecated alias to builtins.type.

builtins.type or typing.Type can be used to annotate class objects.
For example, suppose we have the following classes::

    class User: ...  # Abstract base for User classes
    class BasicUser(User): ...
    class ProUser(User): ...
    class TeamUser(User): ...

And a function that takes a class argument that's a subclass of
User and returns an instance of the corresponding class::

    U = TypeVar('U', bound=User)
    def new_user(user_class: Type[U]) -> U:
        user = user_class()
        # (Here we could write the user object to a database)
        return user

    joe = new_user(BasicUser)

At this point the type checker knows that joe has type BasicUser.

*(has methods, callable)*

### `Tuple`
**Type**: `<class 'typing._TupleType'>`

Deprecated alias to builtins.tuple.

Tuple[X, Y] is the cross-product type of X and Y.

Example: Tuple[T1, T2] is a tuple of two elements corresponding
to type variables T1 and T2.  Tuple[int, float, str] is a tuple
of an int, a float and a string.

To specify a variable-length tuple of homogeneous type, use Tuple[T, ...].

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

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

## Nested Submodules (1)

Each nested submodule is documented in a separate file:

### [torch_geometric](./isinstance/torch_geometric.md)
Module: `torch_geometric`

*Contains: 12 functions, 8 classes*
