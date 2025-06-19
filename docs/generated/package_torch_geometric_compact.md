# Torch_Geometric Package Documentation

Auto-generated documentation for installed package `torch_geometric`

## Package Information

- **Version**: 2.6.1
- **Location**: D:\astro-lab\.venv\Lib\site-packages
- **Summary**: Graph Neural Network Library for PyTorch

## Submodules

### backend
Module: `torch_geometric.backend`

### contrib
Module: `torch_geometric.contrib`

### data
Module: `torch_geometric.data`

### datasets
Module: `torch_geometric.datasets`

### deprecation
Module: `torch_geometric.deprecation`

### edge_index
Module: `torch_geometric.edge_index`

### experimental
Module: `torch_geometric.experimental`

### explain
Module: `torch_geometric.explain`

### graphgym
Module: `torch_geometric.graphgym`

### home
Module: `torch_geometric.home`

### index
Module: `torch_geometric.index`

### inspector
Module: `torch_geometric.inspector`

### io
Module: `torch_geometric.io`

### isinstance
Module: `torch_geometric.isinstance`

### lazy_loader
Module: `torch_geometric.lazy_loader`

### loader
Module: `torch_geometric.loader`

### nn
Module: `torch_geometric.nn`

### profile
Module: `torch_geometric.profile`

GNN profiling package.

### resolver
Module: `torch_geometric.resolver`

### sampler
Module: `torch_geometric.sampler`

Graph sampler package.

### seed
Module: `torch_geometric.seed`

### template
Module: `torch_geometric.template`

### torch_geometric
Module: `torch_geometric`

### transforms
Module: `torch_geometric.transforms`

### typing
Module: `torch_geometric.typing`

### utils
Module: `torch_geometric.utils`

Utility package.

### visualization
Module: `torch_geometric.visualization`

Visualization package.

### warnings
Module: `torch_geometric.warnings`

## Functions

### compile(model: Optional[torch.nn.modules.module.Module] = None, *args: Any, **kwargs: Any) -> Union[torch.nn.modules.module.Module, Callable[[torch.nn.modules.module.Module], torch.nn.modules.module.Module]]
Module: `torch_geometric._compile`

Optimizes the given :pyg:`PyG` model/function via
:meth:`torch.compile`.
This function has the same signature as :meth:`torch.compile` (see
`here <https://pytorch.org/docs/stable/generated/torch.compile.html>`__).

.. note::
    :meth:`torch_geometric.compile` is deprecated in favor of
    :meth:`torch.compile`.

### device(device: Any) -> torch.device
Module: `torch_geometric.device`

Returns a :class:`torch.device`.

If :obj:`"auto"` is specified, returns the optimal device depending on
available hardware.

### get_home_dir() -> str
Module: `torch_geometric.home`

Get the cache directory used for storing all :pyg:`PyG`-related data.

If :meth:`set_home_dir` is not called, the path is given by the environment
variable :obj:`$PYG_HOME` which defaults to :obj:`"~/.cache/pyg"`.

### is_compiling() -> bool
Module: `torch_geometric._compile`

Returns :obj:`True` in case :pytorch:`PyTorch` is compiling via
:meth:`torch.compile`.

### is_debug_enabled() -> bool
Module: `torch_geometric.debug`

Returns :obj:`True` if the debug mode is enabled.

### is_experimental_mode_enabled(options: Union[str, List[str], NoneType] = None) -> bool
Module: `torch_geometric.experimental`

Returns :obj:`True` if the experimental mode is enabled. See
:class:`torch_geometric.experimental_mode` for a list of (optional)
options.

### is_in_onnx_export() -> bool
Module: `torch_geometric._onnx`

Returns :obj:`True` in case :pytorch:`PyTorch` is exporting to ONNX via
:meth:`torch.onnx.export`.

### is_mps_available() -> bool
Module: `torch_geometric.device`

Returns a bool indicating if MPS is currently available.

### is_torch_instance(obj: Any, cls: Union[Type, Tuple[Type]]) -> bool
Module: `torch_geometric.isinstance`

Checks if the :obj:`obj` is an instance of a :obj:`cls`.

This function extends :meth:`isinstance` to be applicable during
:meth:`torch.compile` usage by checking against the original class of
compiled models.

### is_xpu_available() -> bool
Module: `torch_geometric.device`

Returns a bool indicating if XPU is currently available.

### seed_everything(seed: int) -> None
Module: `torch_geometric.seed`

Sets the seed for generating random numbers in :pytorch:`PyTorch`,
:obj:`numpy` and :python:`Python`.

Args:
    seed (int): The desired seed.

### set_home_dir(path: str) -> None
Module: `torch_geometric.home`

Set the cache directory used for storing all :pyg:`PyG`-related data.

Args:
    path (str): The path to a local folder.

## Classes

### EdgeIndex
Module: `torch_geometric.edge_index`

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

**`validate(self) -> 'EdgeIndex'`**

Validates the :class:`EdgeIndex` representation.

In particular, it ensures that

* it only holds valid indices.
* the sort order is correctly set.
* indices are bidirectional in case it is specified as undirected.

**`sparse_size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], int, NoneType]`**

The size of the underlying sparse matrix.
If :obj:`dim` is specified, returns an integer holding the size of that
sparse dimension.

Args:
dim (int, optional): The dimension for which to retrieve the size.
(default: :obj:`None`)

**`get_sparse_size(self, dim: Optional[int] = None) -> Union[torch.Size, int]`**

The size of the underlying sparse matrix.
Automatically computed and cached when not explicitly set.
If :obj:`dim` is specified, returns an integer holding the size of that
sparse dimension.

Args:
dim (int, optional): The dimension for which to retrieve the size.
(default: :obj:`None`)

**`sparse_resize_(self, num_rows: Optional[int], num_cols: Optional[int]) -> 'EdgeIndex'`**

Assigns or re-assigns the size of the underlying sparse matrix.

Args:
num_rows (int, optional): The number of rows.
num_cols (int, optional): The number of columns.

**`get_num_rows(self) -> int`**

The number of rows of the underlying sparse matrix.
Automatically computed and cached when not explicitly set.

**`get_num_cols(self) -> int`**

The number of columns of the underlying sparse matrix.
Automatically computed and cached when not explicitly set.

**`get_indptr(self) -> torch.Tensor`**

Returns the compressed index representation in case
:class:`EdgeIndex` is sorted.

**`get_csr(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]`**

Returns the compressed CSR representation
:obj:`(rowptr, col), perm` in case :class:`EdgeIndex` is sorted.

**`get_csc(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]`**

Returns the compressed CSC representation
:obj:`(colptr, row), perm` in case :class:`EdgeIndex` is sorted.

**`fill_cache_(self, no_transpose: bool = False) -> 'EdgeIndex'`**

Fills the cache with (meta)data information.

Args:
no_transpose (bool, optional): If set to :obj:`True`, will not fill
the cache with information about the transposed
:class:`EdgeIndex`. (default: :obj:`False`)

**`share_memory_(self) -> 'EdgeIndex'`**

*No documentation available.*

**`is_shared(self) -> bool`**

*No documentation available.*

**`as_tensor(self) -> torch.Tensor`**

Zero-copies the :class:`EdgeIndex` representation back to a
:class:`torch.Tensor` representation.

**`sort_by(self, sort_order: Union[str, torch_geometric.edge_index.SortOrder], stable: bool = False) -> 'SortReturnType'`**

Sorts the elements by row or column indices.

Args:
sort_order (str): The sort order, either :obj:`"row"` or
:obj:`"col"`.
stable (bool, optional): Makes the sorting routine stable, which
guarantees that the order of equivalent elements is preserved.
(default: :obj:`False`)

**`to_dense(self, value: Optional[torch.Tensor] = None, fill_value: float = 0.0, dtype: Optional[torch.dtype] = None) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a dense :class:`torch.Tensor`.

.. warning::

In case of duplicated edges, the behavior is non-deterministic (one
of the values from :obj:`value` will be picked arbitrarily). For
deterministic behavior, consider calling
:meth:`~torch_geometric.utils.coalesce` beforehand.

Args:
value (torch.Tensor, optional): The values for non-zero elements.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)
fill_value (float, optional): The fill value for remaining elements
in the dense matrix. (default: :obj:`0.0`)
dtype (torch.dtype, optional): The data type of the returned
tensor. (default: :obj:`None`)

**`to_sparse_coo(self, value: Optional[torch.Tensor] = None) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a :pytorch:`null`
:class:`torch.sparse_coo_tensor`.

Args:
value (torch.Tensor, optional): The values for non-zero elements.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)

**`to_sparse_csr(self, value: Optional[torch.Tensor] = None) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a :pytorch:`null`
:class:`torch.sparse_csr_tensor`.

Args:
value (torch.Tensor, optional): The values for non-zero elements.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)

**`to_sparse_csc(self, value: Optional[torch.Tensor] = None) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a :pytorch:`null`
:class:`torch.sparse_csc_tensor`.

Args:
value (torch.Tensor, optional): The values for non-zero elements.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)

**`to_sparse(self, *, layout: torch.layout = torch.sparse_coo, value: Optional[torch.Tensor] = None) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a
:pytorch:`null` :class:`torch.sparse` tensor.

Args:
layout (torch.layout, optional): The desired sparse layout. One of
:obj:`torch.sparse_coo`, :obj:`torch.sparse_csr`, or
:obj:`torch.sparse_csc`. (default: :obj:`torch.sparse_coo`)
value (torch.Tensor, optional): The values for non-zero elements.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)

**`to_sparse_tensor(self, value: Optional[torch.Tensor] = None) -> torch_geometric.typing.SparseTensor`**

Converts :class:`EdgeIndex` into a
:class:`torch_sparse.SparseTensor`.
Requires that :obj:`torch-sparse` is installed.

Args:
value (torch.Tensor, optional): The values for non-zero elements.
(default: :obj:`None`)

**`matmul(self, other: Union[torch.Tensor, ForwardRef('EdgeIndex')], input_value: Optional[torch.Tensor] = None, other_value: Optional[torch.Tensor] = None, reduce: Literal['sum', 'mean', 'amin', 'amax', 'add', 'min', 'max'] = 'sum', transpose: bool = False) -> Union[torch.Tensor, Tuple[ForwardRef('EdgeIndex'), torch.Tensor]]`**

Performs a matrix multiplication of the matrices :obj:`input` and
:obj:`other`.
If :obj:`input` is a :math:`(n \times m)` matrix and :obj:`other` is a
:math:`(m \times p)` tensor, then the output will be a
:math:`(n \times p)` tensor.
See :meth:`torch.matmul` for more information.

:obj:`input` is a sparse matrix as denoted by the indices in
:class:`EdgeIndex`, and :obj:`input_value` corresponds to the values
of non-zero elements in :obj:`input`.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`.

:obj:`other` can either be a dense :class:`torch.Tensor` or a sparse
:class:`EdgeIndex`.
if :obj:`other` is a sparse :class:`EdgeIndex`, then :obj:`other_value`
corresponds to the values of its non-zero elements.

This function additionally accepts an optional :obj:`reduce` argument
that allows specification of an optional reduction operation.
See :meth:`torch.sparse.mm` for more information.

Lastly, the :obj:`transpose` option allows to perform matrix
multiplication where :obj:`input` will be first transposed, *i.e.*:

.. math::

\textrm{input}^{\top} \cdot \textrm{other}

Args:
other (torch.Tensor or EdgeIndex): The second matrix to be
multiplied, which can be sparse or dense.
input_value (torch.Tensor, optional): The values for non-zero
elements of :obj:`input`.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)
other_value (torch.Tensor, optional): The values for non-zero
elements of :obj:`other` in case it is sparse.
If not specified, non-zero elements will be assigned a value of
:obj:`1.0`. (default: :obj:`None`)
reduce (str, optional): The reduce operation, one of
:obj:`"sum"`/:obj:`"add"`, :obj:`"mean"`,
:obj:`"min"`/:obj:`amin` or :obj:`"max"`/:obj:`amax`.
(default: :obj:`"sum"`)
transpose (bool, optional): If set to :obj:`True`, will perform
matrix multiplication based on the transposed :obj:`input`.
(default: :obj:`False`)

**`sparse_narrow(self, dim: int, start: Union[int, torch.Tensor], length: int) -> 'EdgeIndex'`**

Returns a new :class:`EdgeIndex` that is a narrowed version of
itself. Narrowing is performed by interpreting :class:`EdgeIndex` as a
sparse matrix of shape :obj:`(num_rows, num_cols)`.

In contrast to :meth:`torch.narrow`, the returned tensor does not share
the same underlying storage anymore.

Args:
dim (int): The dimension along which to narrow.
start (int or torch.Tensor): Index of the element to start the
narrowed dimension from.
length (int): Length of the narrowed dimension.

**`to_vector(self) -> torch.Tensor`**

Converts :class:`EdgeIndex` into a one-dimensional index
vector representation.

### Index
Module: `torch_geometric.index`

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

**`validate(self) -> 'Index'`**

Validates the :class:`Index` representation.

In particular, it ensures that

* it only holds valid indices.
* the sort order is correctly set.

**`get_dim_size(self) -> int`**

The size of the underlying sparse vector.
Automatically computed and cached when not explicitly set.

**`dim_resize_(self, dim_size: Optional[int]) -> 'Index'`**

Assigns or re-assigns the size of the underlying sparse vector.

**`get_indptr(self) -> torch.Tensor`**

Returns the compressed index representation in case :class:`Index`
is sorted.

**`fill_cache_(self) -> 'Index'`**

Fills the cache with (meta)data information.

**`share_memory_(self) -> 'Index'`**

*No documentation available.*

**`is_shared(self) -> bool`**

*No documentation available.*

**`as_tensor(self) -> torch.Tensor`**

Zero-copies the :class:`Index` representation back to a
:class:`torch.Tensor` representation.

### LazyLoader
Module: `torch_geometric.lazy_loader`

Create a module object.

The name must be a string; the optional doc argument can have any type.

### debug
Module: `torch_geometric.debug`

Context-manager that enables the debug mode to help track down errors
and separate usage errors from real bugs.

.. code-block:: python

    with torch_geometric.debug():
        out = model(data.x, data.edge_index)

### experimental_mode
Module: `torch_geometric.experimental`

Context-manager that enables the experimental mode to test new but
potentially unstable features.

.. code-block:: python

    with torch_geometric.experimental_mode():
        out = model(data.x, data.edge_index)

Args:
    options (str or list, optional): Currently there are no experimental
        features.

### set_debug
Module: `torch_geometric.debug`

Context-manager that sets the debug mode on or off.

:class:`set_debug` will enable or disable the debug mode based on its
argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`debug` above for more details.

### set_experimental_mode
Module: `torch_geometric.experimental`

Context-manager that sets the experimental mode on or off.

:class:`set_experimental_mode` will enable or disable the experimental mode
based on its argument :attr:`mode`.
It can be used as a context-manager or as a function.

See :class:`experimental_mode` above for more details.
