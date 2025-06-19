# inspector

Part of `torch_geometric.torch_geometric`
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
