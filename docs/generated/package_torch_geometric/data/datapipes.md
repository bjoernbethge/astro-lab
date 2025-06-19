# datapipes

Part of `torch_geometric.data`
Module: `torch_geometric.data.datapipes`

## Functions (2)

### `from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data'`

Converts a SMILES string to a :class:`torch_geometric.data.Data`
instance.

Args:
    smiles (str): The SMILES string.
    with_hydrogen (bool, optional): If set to :obj:`True`, will store
        hydrogens in the molecule graph. (default: :obj:`False`)
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

### `functional_transform(name: str) -> Callable`

## Classes (8)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.

### `Batch`

A data object describing a batch of graphs as one big (disconnected)
graph.
Inherits from :class:`torch_geometric.data.Data` or
:class:`torch_geometric.data.HeteroData`.
In addition, single graphs can be identified via the assignment vector
:obj:`batch`, which maps each node to its respective graph identifier.

:pyg:`PyG` allows modification to the underlying batching procedure by
overwriting the :meth:`~Data.__inc__` and :meth:`~Data.__cat_dim__`
functionalities.
The :meth:`~Data.__inc__` method defines the incremental count between two
consecutive graph attributes.
By default, :pyg:`PyG` increments attributes by the number of nodes
whenever their attribute names contain the substring :obj:`index`
(for historical reasons), which comes in handy for attributes such as
:obj:`edge_index` or :obj:`node_index`.
However, note that this may lead to unexpected behavior for attributes
whose names contain the substring :obj:`index` but should not be
incremented.
To make sure, it is best practice to always double-check the output of
batching.
Furthermore, :meth:`~Data.__cat_dim__` defines in which dimension graph
tensors of the same attribute should be concatenated together.

#### Methods

- **`get_example(self, idx: int) -> torch_geometric.data.data.BaseData`**
  Gets the :class:`~torch_geometric.data.Data` or

- **`index_select(self, idx: Union[slice, torch.Tensor, numpy.ndarray, collections.abc.Sequence]) -> List[torch_geometric.data.data.BaseData]`**
  Creates a subset of :class:`~torch_geometric.data.Data` or

- **`to_data_list(self) -> List[torch_geometric.data.data.BaseData]`**
  Reconstructs the list of :class:`~torch_geometric.data.Data` or

### `Batcher`

Creates mini-batches of data (functional name: ``batch``).

An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
last batch if ``drop_last`` is set to ``False``.

Args:
    datapipe: Iterable DataPipe being batched
    batch_size: The size of each batch
    drop_last: Option to drop the last batch if it's not full
    wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
        defaults to ``DataChunk``

Example:
    >>> # xdoctest: +SKIP
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> dp = IterableWrapper(range(10))
    >>> dp = dp.batch(batch_size=3, drop_last=True)
    >>> list(dp)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

### `DatasetAdapter`

Iterable-style DataPipe.

All DataPipes that represent an iterable of data samples should subclass this.
This style of DataPipes is particularly useful when data come from a stream, or
when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

All subclasses should overwrite :meth:`__iter__`, which would return an
iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
and various state variables within the custom ``IterDataPipe``.

Note:
    Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
    and the creation a second iterator will invalidate the first one. This constraint is necessary because
    some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
    The code example below presents details on how this constraint looks in practice.
    If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

These DataPipes can be invoked in two ways, using the class constructor or applying their
functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
operations in succession.

.. _GitHub IterDataPipe Single Iterator Issue:
    https://github.com/pytorch/data/issues/45

Note:
    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the DataPipe object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

Examples:
    General Usage:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
        >>> list(filter_dp)
        [2, 4, 6, 8, 10]
    Single Iterator Constraint Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> source_dp = IterableWrapper(range(10))
        >>> it1 = iter(source_dp)
        >>> list(it1)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> it1 = iter(source_dp)
        >>> it2 = iter(source_dp)  # The creation of a new iterator invalidates `it1`
        >>> next(it2)
        0
        >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`

#### Methods

- **`is_shardable(self) -> bool`**

- **`apply_sharding(self, num_shards: int, shard_idx: int) -> None`**

### `IterBatcher`

Creates mini-batches of data (functional name: ``batch``).

An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
last batch if ``drop_last`` is set to ``False``.

Args:
    datapipe: Iterable DataPipe being batched
    batch_size: The size of each batch
    drop_last: Option to drop the last batch if it's not full
    wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
        defaults to ``DataChunk``

Example:
    >>> # xdoctest: +SKIP
    >>> from torchdata.datapipes.iter import IterableWrapper
    >>> dp = IterableWrapper(range(10))
    >>> dp = dp.batch(batch_size=3, drop_last=True)
    >>> list(dp)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

### `IterDataPipe`

Iterable-style DataPipe.

All DataPipes that represent an iterable of data samples should subclass this.
This style of DataPipes is particularly useful when data come from a stream, or
when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

All subclasses should overwrite :meth:`__iter__`, which would return an
iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
and various state variables within the custom ``IterDataPipe``.

Note:
    Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
    and the creation a second iterator will invalidate the first one. This constraint is necessary because
    some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
    The code example below presents details on how this constraint looks in practice.
    If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

These DataPipes can be invoked in two ways, using the class constructor or applying their
functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
operations in succession.

.. _GitHub IterDataPipe Single Iterator Issue:
    https://github.com/pytorch/data/issues/45

Note:
    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the DataPipe object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

Examples:
    General Usage:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
        >>> list(filter_dp)
        [2, 4, 6, 8, 10]
    Single Iterator Constraint Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> source_dp = IterableWrapper(range(10))
        >>> it1 = iter(source_dp)
        >>> list(it1)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> it1 = iter(source_dp)
        >>> it2 = iter(source_dp)  # The creation of a new iterator invalidates `it1`
        >>> next(it2)
        0
        >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`

#### Methods

- **`reset(self) -> None`**
  Reset the `IterDataPipe` to the initial state.

### `SMILESParser`

Iterable-style DataPipe.

All DataPipes that represent an iterable of data samples should subclass this.
This style of DataPipes is particularly useful when data come from a stream, or
when the number of samples is too large to fit them all in memory. ``IterDataPipe`` is lazily initialized and its
elements are computed only when ``next()`` is called on the iterator of an ``IterDataPipe``.

All subclasses should overwrite :meth:`__iter__`, which would return an
iterator of samples in this DataPipe. Calling ``__iter__`` of an ``IterDataPipe`` automatically invokes its
method ``reset()``, which by default performs no operation. When writing a custom ``IterDataPipe``, users should
override ``reset()`` if necessary. The common usages include resetting buffers, pointers,
and various state variables within the custom ``IterDataPipe``.

Note:
    Only `one` iterator can be valid for each ``IterDataPipe`` at a time,
    and the creation a second iterator will invalidate the first one. This constraint is necessary because
    some ``IterDataPipe`` have internal buffers, whose states can become invalid if there are multiple iterators.
    The code example below presents details on how this constraint looks in practice.
    If you have any feedback related to this constraint, please see `GitHub IterDataPipe Single Iterator Issue`_.

These DataPipes can be invoked in two ways, using the class constructor or applying their
functional form onto an existing ``IterDataPipe`` (recommended, available to most but not all DataPipes).
You can chain multiple `IterDataPipe` together to form a pipeline that will perform multiple
operations in succession.

.. _GitHub IterDataPipe Single Iterator Issue:
    https://github.com/pytorch/data/issues/45

Note:
    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the DataPipe will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the DataPipe object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

Examples:
    General Usage:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> dp = IterableWrapper(range(10))
        >>> map_dp_1 = Mapper(dp, lambda x: x + 1)  # Using class constructor
        >>> map_dp_2 = dp.map(lambda x: x + 1)  # Using functional form (recommended)
        >>> list(map_dp_1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(map_dp_2)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> filter_dp = map_dp_1.filter(lambda x: x % 2 == 0)
        >>> list(filter_dp)
        [2, 4, 6, 8, 10]
    Single Iterator Constraint Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, Mapper
        >>> source_dp = IterableWrapper(range(10))
        >>> it1 = iter(source_dp)
        >>> list(it1)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> it1 = iter(source_dp)
        >>> it2 = iter(source_dp)  # The creation of a new iterator invalidates `it1`
        >>> next(it2)
        0
        >>> next(it1)  # Further usage of `it1` will raise a `RunTimeError`

### `functional_datapipe`
