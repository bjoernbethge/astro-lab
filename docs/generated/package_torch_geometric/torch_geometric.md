# torch_geometric Submodule

Part of the `torch_geometric` package
Module: `torch_geometric`

## Functions (12)

### `compile(model: Optional[torch.nn.modules.module.Module] = None, *args: Any, **kwargs: Any) -> Union[torch.nn.modules.module.Module, Callable[[torch.nn.modules.module.Module], torch.nn.modules.module.Module]]`

Optimizes the given :pyg:`PyG` model/function via
:meth:`torch.compile`.
This function has the same signature as :meth:`torch.compile` (see
`here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__).

.. note::
    :meth:`torch_geometric.compile` is deprecated in favor of
    :meth:`torch.compile`.

### `device(device: Any) -> torch.device`

Returns a :class:`torch.device`.

If :obj:`"auto"` is specified, returns the optimal device depending on
available hardware.

### `get_home_dir() -> str`

Get the cache directory used for storing all :pyg:`PyG`-related data.

If :meth:`set_home_dir` is not called, the path is given by the environment
variable :obj:`$PYG_HOME` which defaults to :obj:`"~/.cache/pyg"`.

### `is_compiling() -> bool`

Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
:meth:`torch.compile`.

### `is_debug_enabled() -> bool`

Returns :obj:`True` if the debug mode is enabled.

### `is_experimental_mode_enabled(options: Union[str, List[str], NoneType] = None) -> bool`

Returns :obj:`True` if the experimental mode is enabled. See
:class:`torch_geometric.experimental_mode` for a list of (optional)
options.

### `is_in_onnx_export() -> bool`

Returns :obj:`True` in case :pytorch:`PyTorch` is exporting to ONNX via
:meth:`torch.onnx.export`.

### `is_mps_available() -> bool`

Returns a bool indicating if MPS is currently available.

### `is_torch_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool`

Checks if the :obj:`obj` is an instance of a :obj:`cls`.

This function extends :meth:`isinstance` to be applicable during
:meth:`torch.compile` usage by checking against the original class of
compiled models.

### `is_xpu_available() -> bool`

Returns a bool indicating if XPU is currently available.

### `seed_everything(seed: int) -> None`

Sets the seed for generating random numbers in :pytorch:`PyTorch`,
:obj:`numpy` and :python:`Python`.

Args:
    seed (int): The desired seed.

### `set_home_dir(path: str) -> None`

Set the cache directory used for storing all :pyg:`PyG`-related data.

Args:
    path (str): The path to a local folder.

## Important Data Types (8)

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

### `debug`
**Type**: `<class 'type'>`

Context-manager that enables the debug mode to help track down errors
and separate usage errors from real bugs.

.. code-block:: python

    with torch_geometric.debug():
        out = model(data.x, data.edge_index)

*(has methods, callable)*

### `EdgeIndex`
**Type**: `<class 'torch._C._TensorMeta'>`

A COO :obj:`edge_index` tensor with additional (meta)data attached.

:class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
Edges are given as pairwise source and destination node indices in sparse
COO format.

While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`sparse_size`: The underlying sparse matrix size
* :obj:`sort_order`: The sort order (if present), either by row or column.
* :obj:`is_undirected`: Whether edges are bidirectional.

Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
in case its representation is sorted, such as its :obj:`rowptr` or
:obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
versa.
Caches are filled based on demand (*e.g.*, when calling
:meth:`EdgeIndex.sort_by`), or when explicitly requested via
:meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import EdgeIndex

    edge_index = EdgeIndex(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]]
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
        device='cpu',
    )
    >>> EdgeIndex([[0, 1, 1, 2],
    ...            [1, 0, 2, 1]])
    assert edge_index.is_sorted_by_row
    assert edge_index.is_undirected

    # Flipping order:
    edge_index = edge_index.flip(0)
    >>> EdgeIndex([[1, 0, 2, 1],
    ...            [0, 1, 1, 2]])
    assert edge_index.is_sorted_by_col
    assert edge_index.is_undirected

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    edge_index = edge_index[:, mask]
    >>> EdgeIndex([[1, 0, 2],
    ...            [0, 1, 1]])
    assert edge_index.is_sorted_by_col
    assert not edge_index.is_undirected

    # Sparse-Dense Matrix Multiplication:
    out = edge_index.flip(0) @ torch.randn(3, 16)
    assert out.size() == (3, 16)

*(has methods, callable)*

### `set_debug`
**Type**: `<class 'type'>`

Context-manager that sets the debug mode on or off.

:class:`set_debug` will enable or disable the debug mode based on its
argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`debug` above for more details.

*(has methods, callable)*

### `LazyLoader`
**Type**: `<class 'type'>`

Create a module object.

The name must be a string; the optional doc argument can have any type.

*(has methods, callable)*

### `defaultdict`
**Type**: `<class 'type'>`

defaultdict(default_factory=None, /, [...]) --> dict with default factory

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.

*(has methods, callable)*

### `experimental_mode`
**Type**: `<class 'type'>`

Context-manager that enables the experimental mode to test new but
potentially unstable features.

.. code-block:: python

    with torch_geometric.experimental_mode():
        out = model(data.x, data.edge_index)

Args:
    options (str or list, optional): Currently there are no experimental
        features.

*(has methods, callable)*

### `set_experimental_mode`
**Type**: `<class 'type'>`

Context-manager that sets the experimental mode on or off.

:class:`set_experimental_mode` will enable or disable the experimental mode
based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`experimental_mode` above for more details.

*(has methods, callable)*

## Classes (8)

### `EdgeIndex`

A COO :obj:`edge_index` tensor with additional (meta)data attached.

:class:`EdgeIndex` is a :pytorch:`null` :class:`torch.Tensor`, that holds
an :obj:`edge_index` representation of shape :obj:`[2, num_edges]`.
Edges are given as pairwise source and destination node indices in sparse
COO format.

While :class:`EdgeIndex` sub-classes a general :pytorch:`null`
:class:`torch.Tensor`, it can hold additional (meta)data, *i.e.*:

* :obj:`sparse_size`: The underlying sparse matrix size
* :obj:`sort_order`: The sort order (if present), either by row or column.
* :obj:`is_undirected`: Whether edges are bidirectional.

Additionally, :class:`EdgeIndex` caches data for fast CSR or CSC conversion
in case its representation is sorted, such as its :obj:`rowptr` or
:obj:`colptr`, or the permutation vector for going from CSR to CSC or vice
versa.
Caches are filled based on demand (*e.g.*, when calling
:meth:`EdgeIndex.sort_by`), or when explicitly requested via
:meth:`EdgeIndex.fill_cache_`, and are maintained and adjusted over its
lifespan (*e.g.*, when calling :meth:`EdgeIndex.flip`).

This representation ensures for optimal computation in GNN message passing
schemes, while preserving the ease-of-use of regular COO-based :pyg:`PyG`
workflows.

.. code-block:: python

    from torch_geometric import EdgeIndex

    edge_index = EdgeIndex(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]]
        sparse_size=(3, 3),
        sort_order='row',
        is_undirected=True,
        device='cpu',
    )
    >>> EdgeIndex([[0, 1, 1, 2],
    ...            [1, 0, 2, 1]])
    assert edge_index.is_sorted_by_row
    assert edge_index.is_undirected

    # Flipping order:
    edge_index = edge_index.flip(0)
    >>> EdgeIndex([[1, 0, 2, 1],
    ...            [0, 1, 1, 2]])
    assert edge_index.is_sorted_by_col
    assert edge_index.is_undirected

    # Filtering:
    mask = torch.tensor([True, True, True, False])
    edge_index = edge_index[:, mask]
    >>> EdgeIndex([[1, 0, 2],
    ...            [0, 1, 1]])
    assert edge_index.is_sorted_by_col
    assert not edge_index.is_undirected

    # Sparse-Dense Matrix Multiplication:
    out = edge_index.flip(0) @ torch.randn(3, 16)
    assert out.size() == (3, 16)

#### Methods

- **`validate(self) -> 'EdgeIndex'`**
  Validates the :class:`EdgeIndex` representation.

- **`sparse_size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**
  The size of the underlying sparse matrix.

- **`get_sparse_size(self, dim: Optional[int] = None) -> Union[torch.Size, int]`**
  The size of the underlying sparse matrix.

- **`sparse_resize_(self, num_rows: Optional[int], num_cols: Optional[int]) -> 'EdgeIndex'`**
  Assigns or re-assigns the size of the underlying sparse matrix.

- **`get_num_rows(self) -> int`**
  The number of rows of the underlying sparse matrix.

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

### `LazyLoader`

Create a module object.

The name must be a string; the optional doc argument can have any type.

### `debug`

Context-manager that enables the debug mode to help track down errors
and separate usage errors from real bugs.

.. code-block:: python

    with torch_geometric.debug():
        out = model(data.x, data.edge_index)

### `defaultdict`

defaultdict(default_factory=None, /, [...]) --> dict with default factory

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.

#### Methods

- **`copy(...)`**
  D.copy() -> a shallow copy of D.

### `experimental_mode`

Context-manager that enables the experimental mode to test new but
potentially unstable features.

.. code-block:: python

    with torch_geometric.experimental_mode():
        out = model(data.x, data.edge_index)

Args:
    options (str or list, optional): Currently there are no experimental
        features.

### `set_debug`

Context-manager that sets the debug mode on or off.

:class:`set_debug` will enable or disable the debug mode based on its
argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`debug` above for more details.

### `set_experimental_mode`

Context-manager that sets the experimental mode on or off.

:class:`set_experimental_mode` will enable or disable the experimental mode
based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`experimental_mode` above for more details.

## Nested Submodules (26)

Each nested submodule is documented in a separate file:

### [backend](./torch_geometric/backend.md)
Module: `torch_geometric.backend`

*Contains: 1 functions*

### [data](./torch_geometric/data.md)
Module: `torch_geometric.data`

*Contains: 8 functions, 29 classes*

### [datasets](./torch_geometric/datasets.md)
Module: `torch_geometric.datasets`

*Contains: 107 classes*

### [deprecation](./torch_geometric/deprecation.md)
Module: `torch_geometric.deprecation`

*Contains: 1 functions, 1 classes*

### [edge_index](./torch_geometric/edge_index.md)
Module: `torch_geometric.edge_index`

*Contains: 19 functions, 9 classes*

### [experimental](./torch_geometric/experimental.md)
Module: `torch_geometric.experimental`

*Contains: 43 functions, 3 classes*

### [explain](./torch_geometric/explain.md)
Module: `torch_geometric.explain`

*Contains: 5 functions, 13 classes*

### [graphgym](./torch_geometric/graphgym.md)
Module: `torch_geometric.graphgym`

*Contains: 54 functions, 24 classes*

### [home](./torch_geometric/home.md)
Module: `torch_geometric.home`

*Contains: 2 functions*

### [index](./torch_geometric/index.md)
Module: `torch_geometric.index`

*Contains: 11 functions, 4 classes*

### [inspector](./torch_geometric/inspector.md)
Module: `torch_geometric.inspector`

*Contains: 6 functions, 5 classes*

### [io](./torch_geometric/io.md)
Module: `torch_geometric.io`

*Contains: 12 functions*

### [isinstance](./torch_geometric/isinstance.md)
Module: `torch_geometric.isinstance`

*Contains: 1 functions, 1 classes*

### [lazy_loader](./torch_geometric/lazy_loader.md)
Module: `torch_geometric.lazy_loader`

*Contains: 1 functions, 3 classes*

### [loader](./torch_geometric/loader.md)
Module: `torch_geometric.loader`

*Contains: 1 functions, 25 classes*

### [nn](./torch_geometric/nn.md)
Module: `torch_geometric.nn`

*Contains: 33 functions, 175 classes*

### [profile](./torch_geometric/profile.md)
Module: `torch_geometric.profile`

*Contains: 15 functions, 1 classes*

### [resolver](./torch_geometric/resolver.md)
Module: `torch_geometric.resolver`

*Contains: 2 functions, 1 classes*

### [sampler](./torch_geometric/sampler.md)
Module: `torch_geometric.sampler`

*Contains: 9 classes*

### [seed](./torch_geometric/seed.md)
Module: `torch_geometric.seed`

*Contains: 1 functions*

### [template](./torch_geometric/template.md)
Module: `torch_geometric.template`

*Contains: 1 functions, 3 classes*

### [torch_geometric](./torch_geometric/torch_geometric.md)
Module: `torch_geometric`

*Contains: 12 functions, 8 classes*

### [transforms](./torch_geometric/transforms.md)
Module: `torch_geometric.transforms`

*Contains: 1 functions, 66 classes*

### [utils](./torch_geometric/utils.md)
Module: `torch_geometric.utils`

*Contains: 89 functions*

### [visualization](./torch_geometric/visualization.md)
Module: `torch_geometric.visualization`

*Contains: 2 functions*

### [warnings](./torch_geometric/warnings.md)
Module: `torch_geometric.warnings`

*Contains: 2 functions*
