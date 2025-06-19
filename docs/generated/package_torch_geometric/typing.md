# typing Submodule

Part of the `torch_geometric` package
Module: `torch_geometric.typing`

## Important Data Types (15)

### `Adj`
**Type**: `<class 'typing._UnionGenericAlias'>`

*(has methods, callable)*

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

### `Set`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of set.

*(has methods, callable)*

### `Dict`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of dict.

*(has methods, callable)*

### `List`
**Type**: `<class 'typing._SpecialGenericAlias'>`

A generic version of list.

*(has methods, callable)*

### `Size`
**Type**: `<class 'typing._UnionGenericAlias'>`

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

### `pyg_lib`
**Type**: `<class 'type'>`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.

*(has methods, callable)*

### `EdgeType`
**Type**: `<class 'typing._GenericAlias'>`

*(has methods, callable)*

### `Metadata`
**Type**: `<class 'typing._GenericAlias'>`

*(has methods, callable)*

### `NodeType`
**Type**: `<class 'type'>`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

*(has methods, callable)*

### `NoneType`
**Type**: `<class 'typing._UnionGenericAlias'>`

*(has methods, callable)*

### `Optional`
**Type**: `<class 'typing._SpecialForm'>`

Optional[X] is equivalent to Union[X, None].

*(callable)*

## Classes (13)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `EdgeTypeStr`

A helper class to construct serializable edge types by merging an edge
type tuple into a single string.

#### Methods

- **`to_tuple(self) -> Tuple[str, str, str]`**
  Returns the original edge type.

### `MockTorchCSCTensor`

#### Methods

- **`t(self) -> torch.Tensor`**

### `NodeType`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

#### Methods

- **`encode(self, /, encoding='utf-8', errors='strict')`**
  Encode the string using the codec registered for encoding.

- **`replace(self, old, new, count=-1, /)`**
  Return a copy with all occurrences of substring old replaced by new.

- **`split(self, /, sep=None, maxsplit=-1)`**
  Return a list of the substrings in the string, using sep as the separator string.

- **`rsplit(self, /, sep=None, maxsplit=-1)`**
  Return a list of the substrings in the string, using sep as the separator string.

- **`join(self, iterable, /)`**
  Concatenate any number of strings.

### `SparseStorage`

#### Methods

- **`value(self) -> Optional[torch.Tensor]`**

- **`rowcount(self) -> torch.Tensor`**

### `SparseTensor`

#### Methods

- **`size(self, dim: int) -> int`**

- **`nnz(self) -> int`**

- **`is_cuda(self) -> bool`**

- **`has_value(self) -> bool`**

- **`set_value(self, value: Optional[torch.Tensor], layout: Optional[str] = None) -> 'SparseTensor'`**

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

### `TensorFrame`

### `TorchCluster`

### `pyg_lib`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.

### `torch_frame`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.

### `torch_scatter`

The base class of the class hierarchy.

When called, it accepts no arguments and returns a new featureless
instance that has no instance attributes and cannot be given any.

### `torch_sparse`

#### Methods

- **`matmul(src: torch_geometric.typing.SparseTensor, other: torch.Tensor, reduce: str = 'sum') -> torch.Tensor`**

- **`sum(src: torch_geometric.typing.SparseTensor, dim: Optional[int] = None) -> torch.Tensor`**

- **`mul(src: torch_geometric.typing.SparseTensor, other: torch.Tensor) -> torch_geometric.typing.SparseTensor`**

- **`set_diag(src: torch_geometric.typing.SparseTensor, values: Optional[torch.Tensor] = None, k: int = 0) -> torch_geometric.typing.SparseTensor`**

- **`fill_diag(src: torch_geometric.typing.SparseTensor, fill_value: float, k: int = 0) -> torch_geometric.typing.SparseTensor`**
