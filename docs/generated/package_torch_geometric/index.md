# index Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.index`

## Functions (11)

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

### `add_(input: torch_geometric.index.Index, other: Union[int, torch.Tensor, torch_geometric.index.Index], *, alpha: int = 1) -> torch_geometric.index.Index`

### `apply_(tensor: torch_geometric.index.Index, fn: Callable, *args: Any, **kwargs: Any) -> Union[torch_geometric.index.Index, torch.Tensor]`

### `assert_contiguous(tensor: torch.Tensor) -> None`

### `assert_one_dimensional(tensor: torch.Tensor) -> None`

### `assert_sorted(func: Callable) -> Callable`

### `assert_valid_dtype(tensor: torch.Tensor) -> None`

### `implements(torch_function: Callable) -> Callable`

Registers a :pytorch:`PyTorch` function override.

### `index2ptr(index: torch.Tensor, size: Optional[int] = None) -> torch.Tensor`

### `ptr2index(ptr: torch.Tensor, output_size: Optional[int] = None) -> torch.Tensor`

### `sub_(input: torch_geometric.index.Index, other: Union[int, torch.Tensor, torch_geometric.index.Index], *, alpha: int = 1) -> torch_geometric.index.Index`

## Important Data Types (12)

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

### `Index`
**Type**: `<class 'torch._C._TensorMeta'>`

A one-dimensional :obj:`index` tensor with additional (meta)data
attached.

:class:`Index` is a :pytorch:`null` :class:`torch.Tensor` that holds
indices of shape :obj:`[num_indices]`.

While :class:`Index` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`dim_size`: The size of the underlying sparse vector size, *i.e.*,
  the size of a dimension that can be indexed via :obj:`index`.
  By default, it is inferred as :obj:`dim_size=index.max() + 1`.
* :obj:`is_sorted`: Whether indices are sorted in ascending order.

Additionally, :class:`Index` caches data via :obj:`indptr` for fast CSR
conversion in case its representation is sorted.
Caches are filled based on demand (*e.g.*, when calling
:meth:`Index.get_indptr`), or when explicitly requested via
:meth:`Index.fill_cache_`, and are maintaned and adjusted over its
lifespan.

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import Index

    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    >>> Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    assert index.dim_size == 3
    assert index.is_sorted

    # Flipping order:
    edge_index.flip(0)
    >>> Index([[2, 1, 1, 0], dim_size=3)
    assert not index.is_sorted

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    index[:, mask]
    >>> Index([[0, 1, 1], dim_size=3, is_sorted=True)
    assert index.is_sorted

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

### `Iterable`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of collections.abc.Iterable.

*(has methods, callable)*

### `Optional`
**Type**: `<class 'typing._SpecialForm'>`

Optional[X] is equivalent to Union[X, None].

*(callable)*

### `CatMetadata`
**Type**: `<class 'type'>`

CatMetadata(nnz, dim_size, is_sorted)

*(has methods, callable)*

## Classes (4)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `CatMetadata`

CatMetadata(nnz, dim_size, is_sorted)

### `Index`

A one-dimensional :obj:`index` tensor with additional (meta)data
attached.

:class:`Index` is a :pytorch:`null` :class:`torch.Tensor` that holds
indices of shape :obj:`[num_indices]`.

While :class:`Index` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`dim_size`: The size of the underlying sparse vector size, *i.e.*,
  the size of a dimension that can be indexed via :obj:`index`.
  By default, it is inferred as :obj:`dim_size=index.max() + 1`.
* :obj:`is_sorted`: Whether indices are sorted in ascending order.

Additionally, :class:`Index` caches data via :obj:`indptr` for fast CSR
conversion in case its representation is sorted.
Caches are filled based on demand (*e.g.*, when calling
:meth:`Index.get_indptr`), or when explicitly requested via
:meth:`Index.fill_cache_`, and are maintaned and adjusted over its
lifespan.

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import Index

    index = Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    >>> Index([0, 1, 1, 2], dim_size=3, is_sorted=True)
    assert index.dim_size == 3
    assert index.is_sorted

    # Flipping order:
    edge_index.flip(0)
    >>> Index([[2, 1, 1, 0], dim_size=3)
    assert not index.is_sorted

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    index[:, mask]
    >>> Index([[0, 1, 1], dim_size=3, is_sorted=True)
    assert index.is_sorted

#### Methods

- **`validate(self) -> 'Index'`**
  Validates the :class:`Index` representation.

- **`get_dim_size(self) -> int`**
  The size of the underlying sparse vector.

- **`dim_resize_(self, dim_size: Optional[int]) -> 'Index'`**
  Assigns or re-assigns the size of the underlying sparse vector.

- **`get_indptr(self) -> torch.Tensor`**
  Returns the compressed index representation in case :class:`Index`

- **`fill_cache_(self) -> 'Index'`**
  Fills the cache with (meta)data information.

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
