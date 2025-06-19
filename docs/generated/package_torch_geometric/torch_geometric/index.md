# index

Part of `torch_geometric.torch_geometric`
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

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.
