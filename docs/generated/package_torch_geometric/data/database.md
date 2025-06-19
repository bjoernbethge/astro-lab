# database

Part of `torch_geometric.data`
Module: `torch_geometric.data.database`

## Functions (3)

### `abstractmethod(funcobj)`

A decorator indicating abstract methods.

Requires that the metaclass is ABCMeta or derived from it.  A
class that has a metaclass derived from ABCMeta cannot be
instantiated unless all of its abstract methods are overridden.
The abstract methods can be called using any of the normal
'super' call mechanisms.  abstractmethod() may be used to declare
abstract methods for properties and descriptors.

Usage:

    class C(metaclass=ABCMeta):
        @abstractmethod
        def my_abstract_method(self, arg1, arg2, argN):
            ...

### `dataclass(cls=None, /, *, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, match_args=True, kw_only=False, slots=False, weakref_slot=False)`

Add dunder methods based on the fields defined in the class.

Examines PEP 526 __annotations__ to determine fields.

If init is true, an __init__() method is added to the class. If repr
is true, a __repr__() method is added. If order is true, rich
comparison dunder methods are added. If unsafe_hash is true, a
__hash__() method is added. If frozen is true, fields may not be
assigned to after instance creation. If match_args is true, the
__match_args__ tuple is added. If kw_only is true, then by default
all fields are keyword-only. If slots is true, a new class with a
__slots__ attribute is returned.

### `maybe_cast_to_tensor_info(value: Any) -> Union[Any, torch_geometric.data.database.TensorInfo]`

## Classes (13)

### `ABC`

Helper class that provides a standard way to create an ABC using
inheritance.

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `CastMixin`

### `Database`

Base class for inserting and retrieving data from a database.

A database acts as a persisted, out-of-memory and index-based key/value
store for tensor and custom data:

.. code-block:: python

    db = Database()
    db[0] = Data(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> Data(x=[5, 16], y=0, z='id_0')

To improve efficiency, it is recommended to specify the underlying
:obj:`schema` of the data:

.. code-block:: python

    db = Database(schema={  # Custom schema:
        # Tensor information can be specified through a dictionary:
        'x': dict(dtype=torch.float, size=(-1, 16)),
        'y': int,
        'z': str,
    })
    db[0] = dict(x=torch.randn(5, 16), y=0, z='id_0')
    print(db[0])
    >>> {'x': torch.tensor(...), 'y': 0, 'z': 'id_0'}

In addition, databases support batch-wise insert and get, and support
syntactic sugar known from indexing :python:`Python` lists, *e.g.*:

.. code-block:: python

    db = Database()
    db[2:5] = torch.randn(3, 16)
    print(db[torch.tensor([2, 3])])
    >>> [torch.tensor(...), torch.tensor(...)]

Args:
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

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

### `RocksDatabase`

An index-based key/value database based on :obj:`RocksDB`.

.. note::
    This database implementation requires the :obj:`rocksdict` package.

.. warning::
    :class:`RocksDatabase` is currently less optimized than
    :class:`SQLiteDatabase`.

Args:
    path (str): The path to where the database should be saved.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`to_key(index: int) -> bytes`**

### `SQLiteDatabase`

An index-based key/value database based on :obj:`sqlite3`.

.. note::
    This database implementation requires the :obj:`sqlite3` package.

Args:
    path (str): The path to where the database should be saved.
    name (str): The name of the table to save the data to.
    schema (Any or Tuple[Any] or Dict[str, Any], optional): The schema of
        the input data.
        Can take :obj:`int`, :obj:`float`, :obj:`str`, :obj:`object`, or a
        dictionary with :obj:`dtype` and :obj:`size` keys (for specifying
        tensor data) as input, and can be nested as a tuple or dictionary.
        Specifying the schema will improve efficiency, since by default the
        database will use python pickling for serializing and
        deserializing. (default: :obj:`object`)

#### Methods

- **`connect(self) -> None`**
  Connects to the database.

- **`close(self) -> None`**
  Closes the connection to the database.

- **`insert(self, index: int, data: Any) -> None`**
  Inserts data at the specified index.

### `SortOrder`

Create a collection of name/value pairs.

Example enumeration:

>>> class Color(Enum):
...     RED = 1
...     BLUE = 2
...     GREEN = 3

Access them by:

- attribute access::

>>> Color.RED
<Color.RED: 1>

- value lookup:

>>> Color(1)
<Color.RED: 1>

- name lookup:

>>> Color['RED']
<Color.RED: 1>

Enumerations can be iterated over, and know how many members they have:

>>> len(Color)
3

>>> list(Color)
[<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

Methods can be added to enumerations, and members can have their own
attributes -- see the documentation for details.

### `Tensor`

#### Methods

- **`storage(self)`**
  storage() -> torch.TypedStorage

- **`backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None)`**
  Computes the gradient of current tensor wrt graph leaves.

- **`register_hook(self, hook)`**
  Registers a backward hook.

### `TensorInfo`

TensorInfo(dtype: torch.dtype, size: Tuple[int, ...] = (-1,), is_index: bool = False, is_edge_index: bool = False)

### `cached_property`

### `tqdm`

Decorate an iterable object, returning an iterator which acts exactly
like the original iterable, but prints a dynamically updating
progressbar every time a value is requested.

Parameters
----------
iterable  : iterable, optional
    Iterable to decorate with a progressbar.
    Leave blank to manually manage the updates.
desc  : str, optional
    Prefix for the progressbar.
total  : int or float, optional
    The number of expected iterations. If unspecified,
    len(iterable) is used if possible. If float("inf") or as a last
    resort, only basic progress statistics are displayed
    (no ETA, no progressbar).
    If `gui` is True and this parameter needs subsequent updating,
    specify an initial arbitrary large positive number,
    e.g. 9e9.
leave  : bool, optional
    If [default: True], keeps all traces of the progressbar
    upon termination of iteration.
    If `None`, will leave only if `position` is `0`.
file  : `io.TextIOWrapper` or `io.StringIO`, optional
    Specifies where to output the progress messages
    (default: sys.stderr). Uses `file.write(str)` and `file.flush()`
    methods.  For encoding, see `write_bytes`.
ncols  : int, optional
    The width of the entire output message. If specified,
    dynamically resizes the progressbar to stay within this bound.
    If unspecified, attempts to use environment width. The
    fallback is a meter width of 10 and no limit for the counter and
    statistics. If 0, will not print any meter (only stats).
mininterval  : float, optional
    Minimum progress display update interval [default: 0.1] seconds.
maxinterval  : float, optional
    Maximum progress display update interval [default: 10] seconds.
    Automatically adjusts `miniters` to correspond to `mininterval`
    after long display update lag. Only works if `dynamic_miniters`
    or monitor thread is enabled.
miniters  : int or float, optional
    Minimum progress display update interval, in iterations.
    If 0 and `dynamic_miniters`, will automatically adjust to equal
    `mininterval` (more CPU efficient, good for tight loops).
    If > 0, will skip display of specified number of iterations.
    Tweak this and `mininterval` to get very efficient loops.
    If your progress is erratic with both fast and slow iterations
    (network, skipping items, etc) you should set miniters=1.
ascii  : bool or str, optional
    If unspecified or False, use unicode (smooth blocks) to fill
    the meter. The fallback is to use ASCII characters " 123456789#".
disable  : bool, optional
    Whether to disable the entire progressbar wrapper
    [default: False]. If set to None, disable on non-TTY.
unit  : str, optional
    String that will be used to define the unit of each iteration
    [default: it].
unit_scale  : bool or int or float, optional
    If 1 or True, the number of iterations will be reduced/scaled
    automatically and a metric prefix following the
    International System of Units standard will be added
    (kilo, mega, etc.) [default: False]. If any other non-zero
    number, will scale `total` and `n`.
dynamic_ncols  : bool, optional
    If set, constantly alters `ncols` and `nrows` to the
    environment (allowing for window resizes) [default: False].
smoothing  : float, optional
    Exponential moving average smoothing factor for speed estimates
    (ignored in GUI mode). Ranges from 0 (average speed) to 1
    (current/instantaneous speed) [default: 0.3].
bar_format  : str, optional
    Specify a custom bar string formatting. May impact performance.
    [default: '{l_bar}{bar}{r_bar}'], where
    l_bar='{desc}: {percentage:3.0f}%|' and
    r_bar='| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, '
        '{rate_fmt}{postfix}]'
    Possible vars: l_bar, bar, r_bar, n, n_fmt, total, total_fmt,
        percentage, elapsed, elapsed_s, ncols, nrows, desc, unit,
        rate, rate_fmt, rate_noinv, rate_noinv_fmt,
        rate_inv, rate_inv_fmt, postfix, unit_divisor,
        remaining, remaining_s, eta.
    Note that a trailing ": " is automatically removed after {desc}
    if the latter is empty.
initial  : int or float, optional
    The initial counter value. Useful when restarting a progress
    bar [default: 0]. If using float, consider specifying `{n:.3f}`
    or similar in `bar_format`, or specifying `unit_scale`.
position  : int, optional
    Specify the line offset to print this bar (starting from 0)
    Automatic if unspecified.
    Useful to manage multiple bars at once (eg, from threads).
postfix  : dict or *, optional
    Specify additional stats to display at the end of the bar.
    Calls `set_postfix(**postfix)` if possible (dict).
unit_divisor  : float, optional
    [default: 1000], ignored unless `unit_scale` is True.
write_bytes  : bool, optional
    Whether to write bytes. If (default: False) will write unicode.
lock_args  : tuple, optional
    Passed to `refresh` for intermediate output
    (initialisation, iterating, and updating).
nrows  : int, optional
    The screen height. If specified, hides nested bars outside this
    bound. If unspecified, attempts to use environment height.
    The fallback is 20.
colour  : str, optional
    Bar colour (e.g. 'green', '#00ff00').
delay  : float, optional
    Don't display until [default: 0] seconds have elapsed.
gui  : bool, optional
    WARNING: internal parameter - do not use.
    Use tqdm.gui.tqdm(...) instead. If set, will attempt to use
    matplotlib animations for a graphical output [default: False].

Returns
-------
out  : decorated iterator.

#### Methods

- **`format_sizeof(num, suffix='', divisor=1000)`**
  Formats a number (greater than unity) with SI Order of Magnitude

- **`format_interval(t)`**
  Formats a number of seconds as a clock time, [H:]MM:SS

- **`format_num(n)`**
  Intelligent scientific notation (.3g).
