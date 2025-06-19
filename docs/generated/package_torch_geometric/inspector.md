# inspector Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.inspector`

## Functions (6)

### `NamedTuple(typename, fields=None, /, **kwargs)`

Typed version of namedtuple.

Usage::

    class Employee(NamedTuple):
        name: str
        id: int

This is equivalent to::

    Employee = collections.namedtuple('Employee', ['name', 'id'])

The resulting class has an extra __annotations__ attribute, giving a
dict that maps field names to types.  (The field names are also in
the _fields attribute, which is part of the namedtuple API.)
An alternative equivalent functional syntax is also accepted::

    Employee = NamedTuple('Employee', [('name', str), ('id', int)])

### `eval_type(value: Any, _globals: Dict[str, Any]) -> Type`

Returns the type hint of a string.

### `find_parenthesis_content(source: str, prefix: str) -> Optional[str]`

Returns the content of :obj:`{prefix}.*(...)` within :obj:`source`.

### `remove_comments(content: str) -> str`

### `split(content: str, sep: str) -> List[str]`

Splits :obj:`content` based on :obj:`sep`.
:obj:`sep` inside parentheses or square brackets are ignored.

### `type_repr(obj: Any, _globals: Dict[str, Any]) -> str`

Returns the type hint representation of an object.

## Important Data Types (11)

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

### `Tensor`
**Type**: `<class 'torch._C._TensorMeta'>`

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

### `Inspector`
**Type**: `<class 'type'>`

Inspects a given class and collects information about its instance
methods.

Args:
    cls (Type): The class to inspect.

*(has methods, callable)*

### `Parameter`
**Type**: `<class 'type'>`

Parameter(name, type, type_repr, default)

*(has methods, callable)*

### `Signature`
**Type**: `<class 'type'>`

Signature(param_dict, return_type, return_type_repr)

*(has methods, callable)*

## Classes (5)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Inspector`

Inspects a given class and collects information about its instance
methods.

Args:
    cls (Type): The class to inspect.

#### Methods

- **`eval_type(self, value: Any) -> Type`**
  Returns the type hint of a string.

- **`type_repr(self, obj: Any) -> str`**
  Returns the type hint representation of an object.

- **`implements(self, func_name: str) -> bool`**
  Returns :obj:`True` in case the inspected class implements the

- **`inspect_signature(self, func: Union[Callable, str], exclude: Optional[List[Union[str, int]]] = None) -> torch_geometric.inspector.Signature`**
  Inspects the function signature of :obj:`func` and returns a tuple

- **`get_signature(self, func: Union[Callable, str], exclude: Optional[List[str]] = None) -> torch_geometric.inspector.Signature`**
  Returns the function signature of the inspected function

### `Parameter`

Parameter(name, type, type_repr, default)

### `Signature`

Signature(param_dict, return_type, return_type_repr)

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

- **`register_post_accumulate_grad_hook(self, hook)`**
  Registers a backward hook that runs after grad accumulation.

- **`reinforce(self, reward)`**
